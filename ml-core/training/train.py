import argparse
import random
from pathlib import Path

import sys
import torch
import torch.nn as nn
import safetensors.torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.recursive_student import RecursiveHasher


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatImageDataset(Dataset):
    def __init__(self, root, transform, list_file=None, max_images=None, seed=0, cache_ram=False):
        self.paths = self._load_paths(root, list_file)
        if max_images and max_images > 0 and len(self.paths) > max_images:
            rng = random.Random(seed)
            rng.shuffle(self.paths)
            self.paths = self.paths[:max_images]
        
        self.transform = transform
        self.cache_ram = cache_ram
        self.images = [None] * len(self.paths)
        
        if self.cache_ram:
            self._cache_images()

    def _load_paths(self, root, list_file):
        """Load image paths from list file or directory scan."""
        if list_file:
            return self._load_from_list_file(list_file)
        return self._scan_directory(root)

    def _load_from_list_file(self, list_file):
        """Load paths from a text file."""
        base = Path(list_file).resolve().parent
        raw_paths = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
        paths = []
        for p in raw_paths:
            path = Path(p)
            if not path.is_absolute():
                path = (base / path).resolve()
            paths.append(path)
        if not paths:
            raise ValueError(f"No images found in {list_file}")
        return paths

    def _scan_directory(self, root):
        """Scan directory for image files."""
        paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not paths:
            raise ValueError(f"No images found under {root}")
        return paths

    def _cache_images(self):
        """Load all images into RAM."""
        print(f"Loading {len(self.paths)} images into RAM...")
        if sys.platform.startswith('win'):
            print("⚠️ Warning: --cache-ram on Windows duplicates memory per worker. Consider --workers 0 if OOM occurs.")
        
        from tqdm import tqdm
        for i, p in enumerate(tqdm(self.paths, desc="Caching")):
            with Image.open(p) as img:
                self.images[i] = img.convert("RGB")
        print("Finished caching.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache_ram:
            img = self.images[idx]
        else:
            img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_transforms(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_cuda_optimizations(args, device):
    """Configure CUDA-specific optimizations."""
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _create_dataloader(args, transform):
    """Create the training data loader."""
    dataset = FlatImageDataset(
        args.data,
        transform,
        list_file=args.data_list,
        max_images=args.max_images,
        seed=args.seed,
        cache_ram=args.cache_ram,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )


def _create_models(args, device):
    """Create teacher, student, and projection head."""
    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True).to(device)
    teacher.eval()
    
    # Get teacher embedding dimension from config
    teacher_dim = teacher.config.hidden_size  # 1024 for ViT-L

    student = RecursiveHasher(state_dim=args.state_dim, hash_dim=args.hash_dim).to(device)
    
    # Projection head: maps teacher embedding (1024) → student hash dim (96)
    # This is trained alongside the student so we compare in the same space
    proj_head = nn.Sequential(
        nn.Linear(teacher_dim, args.hash_dim),
    ).to(device)
    
    if args.channels_last and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
    
    if args.resume:
        safetensors.torch.load_model(student, args.resume)
    
    return teacher, student, proj_head


def _save_checkpoint(student, checkpoint_dir, name):
    """Save a model checkpoint."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    out = ckpt_path / name
    safetensors.torch.save_model(student, out.as_posix())


def _train_step(images, teacher, student, proj_head, args, device):
    """Run a single training step and return the loss."""
    use_amp = args.amp and device.type == "cuda"
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            teacher_out = teacher(images)
            teacher_emb = teacher_out.last_hidden_state[:, 0]  # [B, 1024]

    state = torch.zeros(images.size(0), args.state_dim, device=device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        for _ in range(args.recursion_steps):
            state, student_hash = student(images, state)
        
        # Project teacher embedding to student hash space and normalize
        target = F.normalize(proj_head(teacher_emb), dim=1)  # [B, 96]
        student_hash = F.normalize(student_hash, dim=1)      # [B, 96]
        
        loss = 1.0 - F.cosine_similarity(student_hash, target).mean()
        loss = loss / args.grad_accum

    return loss


def _train_epoch(epoch, loader, teacher, student, proj_head, optimizer, scaler, args, device, global_step):
    """Run training for one epoch and return updated global_step."""
    student.train()
    proj_head.train()
    total_loss = 0.0
    
    for step, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if args.channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        loss = _train_step(images, teacher, student, proj_head, args, device)
        scaler.scale(loss).backward()
        
        if step % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.grad_accum
        global_step += 1
        
        if step % args.log_interval == 0:
            print(f"epoch={epoch+1} step={step} loss={total_loss / step:.4f}")
        
        if args.checkpoint_dir and args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
            _save_checkpoint(student, args.checkpoint_dir, f"student_step_{global_step}.safetensors")
        
        if args.max_steps and global_step >= args.max_steps:
            break
    
    return global_step


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _configure_cuda_optimizations(args, device)

    transform = build_transforms(args.image_size)
    loader = _create_dataloader(args, transform)
    teacher, student, proj_head = _create_models(args, device)

    # Train both student and projection head
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    for epoch in range(args.epochs):
        global_step = _train_epoch(epoch, loader, teacher, student, proj_head, optimizer, scaler, args, device, global_step)
        
        if args.max_steps and global_step >= args.max_steps:
            break

        if args.checkpoint_dir:
            _save_checkpoint(student, args.checkpoint_dir, f"student_epoch_{epoch+1}.safetensors")


def main():
    parser = argparse.ArgumentParser(description="Train the recursive hasher via DINOv3 distillation.")
    parser.add_argument("--data", default="", help="Root folder containing images.")
    parser.add_argument("--data-list", default="", help="Optional text file with one image path per line.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images.")
    parser.add_argument("--teacher", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hash-dim", type=int, default=96)
    parser.add_argument("--recursion-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-ram", action="store_true", help="Cache all images in RAM (decodes to RGB). Use with caution on Windows (high memory usage per worker).")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save every N steps (0 disables).")
    parser.add_argument("--resume", default="", help="Path to safetensors checkpoint.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N total steps.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 on CUDA.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format for images.")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark for conv layers.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.data and not args.data_list:
        parser.error("either --data or --data-list is required")

    train(args)


if __name__ == "__main__":
    main()
