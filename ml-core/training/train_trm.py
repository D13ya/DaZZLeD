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
        
        # Apply transform (handles both single-view and two-view)
        return self.transform(img)


class TwoViewTransform:
    """
    Wrapper that returns two augmented views of the same image.
    Used for SimCLR-style contrastive learning.
    """
    def __init__(self, base_transform):
        self.transform = base_transform
    
    def __call__(self, img):
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


def get_normalization_config(teacher_name=None):
    """Get normalization mean/std - returns consistent values for training and eval."""
    if teacher_name:
        try:
            processor = AutoImageProcessor.from_pretrained(teacher_name, trust_remote_code=True)
            mean = list(processor.image_mean)
            std = list(processor.image_std)
            # Use teacher's expected size if available
            image_size = 224
            if hasattr(processor, 'size'):
                if isinstance(processor.size, dict):
                    image_size = processor.size.get('shortest_edge', 224)
                else:
                    image_size = processor.size
            return mean, std, image_size
        except Exception:
            pass
    # Fallback to ImageNet defaults
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224


def build_transforms(image_size, teacher_name=None, two_view=False):
    """Build transforms. If teacher_name is provided, use teacher's processor for normalization."""
    mean, std, teacher_size = get_normalization_config(teacher_name)
    if teacher_size != 224:
        image_size = teacher_size
    
    # Augmentations based on arXiv:2406.00918 findings:
    # - Rotation: PHAs vulnerable to rotation attacks (±15° provides robustness)
    # - Filtering: GaussianBlur helps with filter-based attacks
    # - Color/Crop: Standard augmentations for general robustness
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),  # ±15° rotation (arXiv:2406.00918)
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    if two_view:
        return TwoViewTransform(base_transform), mean, std
    return base_transform, mean, std


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


def info_nce_loss(embeddings, temperature=0.07):
    """
    InfoNCE contrastive loss (in-batch negatives) - REPULSION ONLY.
    
    For a batch of N embeddings from different images, pushes them apart.
    Used when we have single-view batch.
    
    Args:
        embeddings: [B, D] normalized embeddings
        temperature: τ, controls sharpness (lower = sharper)
    
    Returns:
        Scalar loss (lower when different images are dissimilar)
    """
    # Similarity matrix [B, B]
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    
    # Labels: each sample is its own positive (diagonal)
    batch_size = embeddings.size(0)
    labels = torch.arange(batch_size, device=embeddings.device)
    
    # Cross-entropy: maximize diagonal, minimize off-diagonal
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def simclr_loss(z1, z2, temperature=0.07):
    """
    SimCLR contrastive loss with two views.
    
    For each image, we have two augmented views (z1, z2).
    - Positive pairs: (z1[i], z2[i]) - same image, different augmentation
    - Negative pairs: all other combinations
    
    This PULLS same-image pairs together AND PUSHES different images apart.
    
    Args:
        z1: [B, D] normalized embeddings from view 1
        z2: [B, D] normalized embeddings from view 2
        temperature: τ, controls sharpness
    
    Returns:
        Scalar loss
    """
    batch_size = z1.size(0)
    
    # Concatenate both views: [2B, D]
    z = torch.cat([z1, z2], dim=0)
    
    # Similarity matrix: [2B, 2B]
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    # For row i in [0, B): positive is at column i+B
    # For row i in [B, 2B): positive is at column i-B
    pos_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])
    
    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    # = -pos + log(sum(exp(all)))
    # = cross_entropy(sim_matrix, pos_indices)
    loss = F.cross_entropy(sim_matrix, pos_indices)
    
    return loss


def train_step_deep_supervision(
    view1,
    view2,
    teacher,
    student,
    proj_head,
    optimizer,
    scaler,
    args,
    device,
    effective_contrast_weight,
):
    """
    TRM training with deep supervision + SimCLR contrastive loss (Option A).
    
    Loss function:
        L = (1/N_sup) * Σ_{k=1}^{N_sup} L_align(y_k, teacher) 
            + λ * L_SimCLR(y_N^1, y_N^2) / log(2B)
    
    Key properties:
    - Distillation at EVERY step k (core TRM deep supervision)
    - SimCLR ONLY at final step N (on most refined hash)
    - Contrast loss normalized by log(2B) for stable λ tuning
    - Two augmented views per image (true SimCLR)
    """
    batch_size = view1.size(0)
    use_amp = args.amp and device.type == "cuda"
    
    optimizer.zero_grad(set_to_none=True)

    # Get teacher embedding from view1 only (frozen, no grad)
    # INTENTIONAL: Both augmented views align to the SAME teacher target.
    # This enforces augmentation invariance - different crops/colors of the
    # same image should produce the same hash (matching the teacher's view).
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            teacher_out = teacher(view1)
            teacher_emb = teacher_out.last_hidden_state[:, 0]  # [B, 1024]

    with torch.amp.autocast('cuda', enabled=use_amp):
        # Project teacher embedding to hash space
        target = F.normalize(proj_head(teacher_emb), dim=1)  # [B, hash_dim]

        # Encode BOTH views
        x1 = student.image_encoder(view1)
        x2 = student.image_encoder(view2)

        # Initialize y and z for both views
        y1 = student.y_init.expand(batch_size, -1).to(device)
        z1 = student.z_init.expand(batch_size, -1).to(device)
        y2 = student.y_init.expand(batch_size, -1).to(device)
        z2 = student.z_init.expand(batch_size, -1).to(device)

        accumulated_align_loss = 0.0
        step_losses = []

        # Deep supervision loop
        # L = Σ_k L_align(y_k, target) + λ * L_SimCLR(y_N^1, y_N^2)
        for _ in range(args.n_sup):
            # Deep recursion: T-1 without grad, 1 with grad (for both views)
            with torch.no_grad():
                for _ in range(student.t - 1):
                    y1, z1 = student.latent_recursion(x1, y1, z1)
                    y2, z2 = student.latent_recursion(x2, y2, z2)

            # Final recursion with gradient
            y1_new, z1_new = student.latent_recursion(x1, y1, z1)
            y2_new, z2_new = student.latent_recursion(x2, y2, z2)

            # Compute hashes for both views
            hash1 = F.normalize(student.output_head(y1_new), p=2, dim=-1)
            hash2 = F.normalize(student.output_head(y2_new), p=2, dim=-1)

            # Loss 1: Alignment with teacher at EVERY step (core TRM deep supervision)
            # Apply to BOTH views - each view should match the teacher embedding
            align_loss1 = 1.0 - F.cosine_similarity(hash1, target).mean()
            align_loss2 = 1.0 - F.cosine_similarity(hash2, target).mean()
            align_loss = (align_loss1 + align_loss2) / 2.0  # Average over views
            accumulated_align_loss = accumulated_align_loss + align_loss
            step_losses.append(align_loss.item())

            # Carry forward (detached for next supervision step)
            y1, z1 = y1_new.detach(), z1_new.detach()
            y2, z2 = y2_new.detach(), z2_new.detach()

            # ACT: Early stopping if loss is very low
            if args.use_act and align_loss.item() < 0.01:
                break

        # Loss 2: SimCLR contrastive loss ONLY at final step (on most refined hash)
        # This is Option A: L = Σ_k L_align(y_k) + λ * L_SimCLR(y_N^1, y_N^2)
        #
        # LOSS SCALE NOTE:
        # - align_loss ≈ 0.2-0.6 (cosine distance)
        # - contrast_loss ≈ log(2B) ≈ 5-6 for B=192
        # We normalize by log(2B) so λ is interpretable (λ=1 means equal weight)
        if effective_contrast_weight > 0:
            contrast_loss = simclr_loss(hash1, hash2, temperature=args.nce_temperature)
            # Normalize by log(2B) to make λ more interpretable
            log_2b = torch.log(torch.tensor(2.0 * batch_size, device=device))
            contrast_loss_normalized = contrast_loss / log_2b
        else:
            contrast_loss_normalized = torch.tensor(0.0, device=device)
            contrast_loss = torch.tensor(0.0, device=device)

        # Total loss: deep supervision alignment + final-step contrastive
        n_steps = len(step_losses)
        total_loss = accumulated_align_loss / n_steps + effective_contrast_weight * contrast_loss_normalized

    # Single backward + step per batch
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Return combined loss for logging (use same formula as training)
    return (sum(step_losses) / n_steps) + effective_contrast_weight * contrast_loss_normalized.item()


def _configure_backends(args, device):
    """Configure CUDA backends for performance."""
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _create_dataloader(args):
    """Create training dataloader with two-view transform for SimCLR."""
    # Use teacher's preprocessing for correct normalization
    # two_view=True returns TwoViewTransform that gives (view1, view2) per image
    use_two_view = args.contrast_weight > 0
    transform, mean, std = build_transforms(
        args.image_size, 
        teacher_name=args.teacher,
        two_view=use_two_view
    )
    
    # Store normalization config for eval consistency
    args._norm_mean = mean
    args._norm_std = std
    
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
    ), use_two_view


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


def _run_epoch(epoch, loader, teacher, student, proj_head, optimizer, scheduler, scaler, ema, args, device, global_step, use_two_view):
    """Run one training epoch."""
    student.train()
    epoch_loss = 0.0
    
    # Contrast warmup: λ=0 for first N epochs
    effective_contrast_weight = 0.0 if epoch < args.contrast_warmup_epochs else args.contrast_weight
    if epoch == args.contrast_warmup_epochs and args.contrast_weight > 0:
        print(f"  📈 Enabling SimCLR contrastive loss (λ={args.contrast_weight})")

    for step, batch in enumerate(loader, start=1):
        if use_two_view:
            # batch is (view1, view2) tuple
            view1, view2 = batch
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if args.channels_last:
                view1 = view1.contiguous(memory_format=torch.channels_last)
                view2 = view2.contiguous(memory_format=torch.channels_last)
        else:
            # Single view (no contrastive loss)
            view1 = batch.to(device, non_blocking=True)
            view2 = view1  # Same view (SimCLR loss will be identity, but contrast_weight=0 anyway)
            if args.channels_last and view1.dim() == 4:
                view1 = view1.contiguous(memory_format=torch.channels_last)
                view2 = view1

        loss = train_step_deep_supervision(
            view1, view2, teacher, student, proj_head, optimizer, scaler, args, device,
            effective_contrast_weight
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
            save_checkpoint(student, proj_head, ema, args.checkpoint_dir, f"student_step_{global_step}.safetensors", args)

        if args.max_steps and global_step >= args.max_steps:
            break

    return epoch_loss, global_step


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _configure_backends(args, device)

    loader, use_two_view = _create_dataloader(args)
    teacher, student, proj_head = _create_models(args, device)
    optimizer, scheduler = _create_optimizer_and_scheduler(args, student, proj_head)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = EMA(student, decay=args.ema_decay) if args.use_ema else None

    global_step = 0
    mode = "SimCLR two-view" if use_two_view else "single-view"
    print(f"\nStarting TRM training with deep supervision (N_sup={args.n_sup}, mode={mode})...")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")
    if hasattr(args, '_norm_mean'):
        print(f"Normalization: mean={args._norm_mean}, std={args._norm_std}")

    for epoch in range(args.epochs):
        epoch_loss, global_step = _run_epoch(
            epoch, loader, teacher, student, proj_head,
            optimizer, scheduler, scaler, ema, args, device, global_step, use_two_view
        )

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.checkpoint_dir:
            save_checkpoint(student, proj_head, ema, args.checkpoint_dir, f"student_epoch_{epoch+1}.safetensors", args)

        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss/len(loader):.4f}")


def save_checkpoint(student, proj_head, ema, checkpoint_dir, name, args=None):
    """Save student weights (optionally with EMA) + normalization config for eval."""
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
    
    # Save normalization config for eval consistency
    if args is not None and hasattr(args, '_norm_mean'):
        import json
        config = {
            "norm_mean": args._norm_mean,
            "norm_std": args._norm_std,
            "hash_dim": args.hash_dim,
            "embed_dim": args.embed_dim,
            "n_layers": args.n_layers,
            "n_latent": args.n_latent,
            "t": args.t,
            "image_size": args.image_size,
        }
        config_path = ckpt_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


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
    parser.add_argument("--hash-dim", type=int, default=128, help="Output hash dimension (96=compact, 128=balanced, 192=high-cap)")
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
    
    # Contrastive loss (SimCLR NT-Xent) - applied only at final supervision step
    # Loss is normalized by log(2B) so λ=1.0 means equal weight to align and contrast
    parser.add_argument("--contrast-weight", type=float, default=0.3, help="λ for SimCLR (normalized, 0.1-0.5 recommended)")
    parser.add_argument("--nce-temperature", type=float, default=0.07, help="SimCLR temperature τ (lower=sharper)")
    parser.add_argument("--contrast-warmup-epochs", type=int, default=1, help="Epochs with λ=0 before enabling SimCLR")

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
