"""
TRM Training Script with Deep Supervision

Based on: "Less is More: Recursive Reasoning with Tiny Networks"
https://arxiv.org/abs/2510.04871

Key training features:
1. Deep supervision: N_sup=16 supervision steps per sample
2. Loss at EACH supervision step (not just final)
3. Carry (y, z) across supervision steps (detached)
4. EMA for stability
5. ACT (Adaptive Computational Time) for efficiency
"""

import argparse
import random
from pathlib import Path
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.trm_hasher import TRMHasher


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
        if list_file:
            return self._load_from_list_file(list_file)
        return self._scan_directory(root)

    def _load_from_list_file(self, list_file):
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
        paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not paths:
            raise ValueError(f"No images found under {root}")
        return paths

    def _cache_images(self):
        print(f"Loading {len(self.paths)} images into RAM...")
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


def build_transforms(image_size, teacher_name=None):
    \"\"\"Build transforms. If teacher_name is provided, use teacher's processor for normalization.\"\"\"
    # Get teacher's expected normalization if available
    if teacher_name:
        try:
            processor = AutoImageProcessor.from_pretrained(teacher_name, trust_remote_code=True)
            mean = processor.image_mean
            std = processor.image_std
            # Use teacher's expected size if available
            if hasattr(processor, 'size'):
                if isinstance(processor.size, dict):
                    image_size = processor.size.get('shortest_edge', image_size)
                else:
                    image_size = processor.size
        except Exception:
            # Fallback to ImageNet defaults
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average of model weights (as used in TRM paper)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_step_deep_supervision(
    images,
    teacher,
    student,
    proj_head,
    optimizer,
    scaler,
    args,
    device,
):
    """
    TRM training with deep supervision.
    
    Key differences from standard distillation:
    1. Run N_sup supervision steps per batch
    2. Compute loss at EACH step
    3. Carry (y, z) across steps (detached between steps)
    4. Accumulate losses and do ONE backward+step per batch
    
    This avoids:
    - Stale encoder outputs (x is in the accumulated graph)
    - Scheduler/optimizer mismatch (one step per batch)
    """
    batch_size = images.size(0)
    use_amp = args.amp and device.type == "cuda"
    
    optimizer.zero_grad(set_to_none=True)

    # Get teacher embedding (frozen, no grad)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            teacher_out = teacher(images)
            teacher_emb = teacher_out.last_hidden_state[:, 0]  # [B, 1024]

    # Project teacher embedding to hash space (WITH gradient - proj_head trains)
    with torch.amp.autocast('cuda', enabled=use_amp):
        target = F.normalize(proj_head(teacher_emb), dim=1)  # [B, hash_dim]

        # Encode image (part of trainable graph)
        x = student.image_encoder(images)

        # Initialize y and z
        y = student.y_init.expand(batch_size, -1).to(device)
        z = student.z_init.expand(batch_size, -1).to(device)

        accumulated_loss = 0.0
        step_losses = []

        # Deep supervision loop - accumulate losses
        for _ in range(args.n_sup):
            # Deep recursion: T-1 without grad, 1 with grad
            with torch.no_grad():
                for _ in range(student.t - 1):
                    y_tmp, z_tmp = student.latent_recursion(x, y, z)
                    y, z = y_tmp, z_tmp

            # Final recursion with gradient
            y_new, z_new = student.latent_recursion(x, y, z)

            # Compute hash
            hash_out = student.output_head(y_new)
            hash_out = F.normalize(hash_out, p=2, dim=-1)

            # Loss: cosine similarity to teacher
            step_loss = 1.0 - F.cosine_similarity(hash_out, target).mean()
            accumulated_loss = accumulated_loss + step_loss
            step_losses.append(step_loss.item())

            # Carry forward (detached for next supervision step)
            y = y_new.detach()
            z = z_new.detach()

            # ACT: Early stopping if loss is very low
            if args.use_act and step_loss.item() < 0.01:
                break

        # Average the accumulated loss
        n_steps = len(step_losses)
        accumulated_loss = accumulated_loss / n_steps

    # Single backward + step per batch
    scaler.scale(accumulated_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return sum(step_losses) / n_steps


def _configure_backends(args, device):
    """Configure CUDA backends for performance."""
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _create_dataloader(args):
    """Create training dataloader."""
    # Use teacher's preprocessing for correct normalization
    transform = build_transforms(args.image_size, teacher_name=args.teacher)
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
    print("Loading teacher model...")
    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True).to(device)
    teacher.eval()
    teacher_dim = teacher.config.hidden_size

    print("Creating TRM student...")
    student = TRMHasher(
        embed_dim=args.embed_dim,
        hash_dim=args.hash_dim,
        n_layers=args.n_layers,
        n_latent=args.n_latent,
        t=args.t,
    ).to(device)

    if args.channels_last and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)

    # Fixed orthogonal projection: teacher_dim -> hash_dim
    # Using orthogonal init preserves distances better than random (JL-lemma)
    proj_head = nn.Linear(teacher_dim, args.hash_dim, bias=False).to(device)
    nn.init.orthogonal_(proj_head.weight)
    proj_head.requires_grad_(False)  # Freeze - fixed target prevents drift

    if args.resume:
        safetensors.torch.load_model(student, args.resume)
        # Note: proj_head is fixed, no need to load/save it

    return teacher, student, proj_head


def _create_optimizer_and_scheduler(args, student, proj_head):
    """Create optimizer and LR scheduler."""
    # Only train student - proj_head is frozen (fixed orthogonal projection)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _run_epoch(epoch, loader, teacher, student, proj_head, optimizer, scheduler, scaler, ema, args, device, global_step):
    """Run one training epoch."""
    student.train()
    epoch_loss = 0.0

    for step, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if args.channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        loss = train_step_deep_supervision(
            images, teacher, student, proj_head, optimizer, scaler, args, device
        )

        scheduler.step()
        epoch_loss += loss
        global_step += 1

        if ema is not None:
            ema.update(student)

        if step % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"epoch={epoch+1} step={step} loss={epoch_loss/step:.4f} lr={lr:.2e}")

        if args.checkpoint_dir and args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
            save_checkpoint(student, proj_head, ema, args.checkpoint_dir, f"student_step_{global_step}.safetensors")

        if args.max_steps and global_step >= args.max_steps:
            break

    return epoch_loss, global_step


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _configure_backends(args, device)

    loader = _create_dataloader(args)
    teacher, student, proj_head = _create_models(args, device)
    optimizer, scheduler = _create_optimizer_and_scheduler(args, student, proj_head)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = EMA(student, decay=args.ema_decay) if args.use_ema else None

    global_step = 0
    print(f"\nStarting TRM training with deep supervision (N_sup={args.n_sup})...")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    for epoch in range(args.epochs):
        epoch_loss, global_step = _run_epoch(
            epoch, loader, teacher, student, proj_head,
            optimizer, scheduler, scaler, ema, args, device, global_step
        )

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.checkpoint_dir:
            save_checkpoint(student, proj_head, ema, args.checkpoint_dir, f"student_epoch_{epoch+1}.safetensors")

        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss/len(loader):.4f}")


def save_checkpoint(student, proj_head, ema, checkpoint_dir, name):
    """Save student weights (optionally with EMA). proj_head is fixed, no need to save."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    student_out = ckpt_path / name

    if ema is not None:
        # Save EMA weights for student
        ema.apply_shadow(student)
        safetensors.torch.save_model(student, student_out.as_posix())
        ema.restore(student)
    else:
        safetensors.torch.save_model(student, student_out.as_posix())


def main():
    parser = argparse.ArgumentParser(description="TRM training with deep supervision")

    # Data
    parser.add_argument("--data", default="", help="Root folder containing images")
    parser.add_argument("--data-list", default="", help="Text file with image paths")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--cache-ram", action="store_true")

    # Model
    parser.add_argument("--teacher", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--embed-dim", type=int, default=256, help="TRM embedding dimension")
    parser.add_argument("--hash-dim", type=int, default=96, help="Output hash dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="TRM network layers (paper uses 2)")
    parser.add_argument("--n-latent", type=int, default=6, help="Latent recursion steps per cycle")
    parser.add_argument("--t", type=int, default=3, help="Number of cycles (T-1 no grad, 1 with grad)")
    parser.add_argument("--n-sup", type=int, default=16, help="Deep supervision steps")

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--use-ema", action="store_true", help="Use EMA (recommended)")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-act", action="store_true", help="Early stop supervision if loss low")

    # Optimization
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if not args.data and not args.data_list:
        parser.error("either --data or --data-list is required")

    train(args)


if __name__ == "__main__":
    main()
