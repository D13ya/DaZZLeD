import argparse
import random
from pathlib import Path

import sys
import torch
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
        if list_file:
            base = Path(list_file).resolve().parent
            raw_paths = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
            self.paths = []
            for p in raw_paths:
                path = Path(p)
                if not path.is_absolute():
                    path = (base / path).resolve()
                self.paths.append(path)
        else:
            self.paths = [
                p for p in Path(root).rglob("*")
                if p.suffix.lower() in IMAGE_EXTS
            ]
        if not self.paths:
            source = list_file or root
            raise ValueError(f"No images found under {source}")
        if max_images and max_images > 0 and len(self.paths) > max_images:
            rng = random.Random(seed)
            rng.shuffle(self.paths)
            self.paths = self.paths[:max_images]
        
        self.transform = transform
        self.cache_ram = cache_ram
        self.images = [None] * len(self.paths)
        
        if self.cache_ram:
            print(f"Loading {len(self.paths)} images into RAM...")
            # Detect excessive memory usage risk
            is_windows = sys.platform.startswith('win')
            if is_windows:
                print("⚠️ Warning: --cache-ram on Windows duplicates memory per worker. Consider --workers 0 if OOM occurs.")
            
            from tqdm import tqdm
            for i, p in enumerate(tqdm(self.paths, desc="Caching")):
                # Store as decoded RGB to save decoding time during training
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


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = build_transforms(args.image_size)
    dataset = FlatImageDataset(
        args.data,
        transform,
        list_file=args.data_list,
        max_images=args.max_images,
        seed=args.seed,
        cache_ram=args.cache_ram,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True).to(device)
    teacher.eval()

    student = RecursiveHasher(state_dim=args.state_dim, hash_dim=args.hash_dim).to(device)
    if args.channels_last and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if args.resume:
        safetensors.torch.load_model(student, args.resume)

    global_step = 0
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        for step, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if args.channels_last and images.dim() == 4:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    teacher_out = teacher(images)
                    target = F.normalize(teacher_out.last_hidden_state[:, 0], dim=1)

            state = torch.zeros(images.size(0), args.state_dim, device=device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                for _ in range(args.recursion_steps):
                    state, student_hash = student(images, state)
                loss = 1.0 - F.cosine_similarity(student_hash, target).mean()
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum
            global_step += 1
            if step % args.log_interval == 0:
                avg = total_loss / step
                print(f"epoch={epoch+1} step={step} loss={avg:.4f}")
            if args.checkpoint_dir and args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                ckpt_path = Path(args.checkpoint_dir)
                ckpt_path.mkdir(parents=True, exist_ok=True)
                out = ckpt_path / f"student_step_{global_step}.safetensors"
                safetensors.torch.save_model(student, out.as_posix())
            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.checkpoint_dir:
            ckpt_path = Path(args.checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            out = ckpt_path / f"student_epoch_{epoch+1}.safetensors"
            safetensors.torch.save_model(student, out.as_posix())


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
