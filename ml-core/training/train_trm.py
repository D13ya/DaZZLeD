"""
TRM Training Script with Deep Supervision - Paper-Exact Version

Based on: "Less is More: Recursive Reasoning with Tiny Networks"
https://arxiv.org/abs/2510.04871

Implementation notes:
1. Per-step backward: loss.backward() at each supervision step
2. Scheduler and EMA step at each supervision step
3. total_steps = epochs * batches_per_epoch * n_sup
4. ACT triggers final-step losses on last executed step
5. Separate batch_step vs update_step tracking
6. Optional frozen encoder mode for faster training
7. Single-view mode when contrast_weight=0 (fast)
8. LOGIC2 losses: BCE alignment + hash centers + quantization + margin + SimCLR
"""

import argparse
import hashlib
import random
import re
from collections import Counter
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


# ============================================================================
# Loss Functions
# ============================================================================

def bit_balance_loss(hash_batch):
    """Penalize non-uniform bit distribution in the hash."""
    dim_means = hash_batch.mean(dim=0)
    return (dim_means ** 2).mean()


def hash_entropy_estimate(hash_batch):
    """Estimate effective entropy in bits (for monitoring)."""
    binary = (hash_batch > 0.5).float()
    p = binary.mean(dim=0).clamp(min=1e-7, max=1 - 1e-7)
    entropy_per_bit = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
    return entropy_per_bit.sum()


def count_hash_collisions(hash_batch, threshold=0.99):
    """Count near-collisions in a batch (for monitoring)."""
    centered = F.normalize(hash_batch - 0.5, p=2, dim=-1)
    sim_matrix = torch.mm(centered, centered.t())
    batch_size = hash_batch.size(0)
    mask = torch.triu(
        torch.ones(batch_size, batch_size, device=hash_batch.device, dtype=torch.bool),
        diagonal=1,
    )
    collisions = ((sim_matrix > threshold) & mask).sum()
    return collisions.item()


def simclr_loss(z1, z2, temperature=0.07, hard_neg_k=0):
    """
    SimCLR NT-Xent loss with optional hard negative mining.

    Args:
        z1, z2: [B, D] continuous hash outputs in (0,1)
        temperature: softmax temperature
        hard_neg_k: if > 0, only use top-k hardest negatives per sample
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1 - 0.5, p=2, dim=-1)
    z2 = F.normalize(z2 - 0.5, p=2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

    pos_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    if hard_neg_k > 0 and hard_neg_k < 2 * batch_size - 2:
        row_idx = torch.arange(2 * batch_size, device=z.device)
        sim_for_topk = sim_matrix.clone()
        sim_for_topk[row_idx, pos_indices] = -float("inf")
        topk = torch.topk(sim_for_topk, k=hard_neg_k, dim=1).indices

        keep = torch.zeros_like(sim_matrix, dtype=torch.bool)
        keep[row_idx, pos_indices] = True
        keep[row_idx.unsqueeze(1), topk] = True
        sim_matrix = sim_matrix.masked_fill(~keep, -float("inf"))

    return F.cross_entropy(sim_matrix, pos_indices)


def quantization_loss(hash_batch):
    """Quantization loss (log cosh) to push hashes toward {0,1}."""
    return torch.log(torch.cosh(torch.abs(2 * hash_batch - 1) - 1)).mean()


def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Generate a Hadamard matrix of size (size x size). size must be power-of-two."""
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError("hash_dim must be power-of-two for Hadamard centers")
    h = torch.tensor([[1.0]], device=device)
    while h.size(0) < size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def build_hash_centers(num_centers: int, hash_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    """Build hash centers using a Hadamard matrix; returns [num_centers, hash_dim] in {0,1}."""
    if num_centers <= 0:
        raise ValueError("num_centers must be > 0")
    h = hadamard_matrix(hash_dim, device=device)
    if num_centers > hash_dim:
        h = torch.cat([h, -h], dim=0)
    if num_centers > h.size(0):
        raise ValueError("num_centers exceeds available Hadamard rows")
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        perm = torch.randperm(h.size(0), generator=g, device=device)
        h = h[perm]
    centers = (h[:num_centers] + 1.0) / 2.0
    return centers


def hash_center_losses(hash_batch, labels, centers, neg_k: int):
    """Compute L_C and L_D for hash centers. Returns (center_loss, distinct_loss)."""
    if labels is None or centers is None:
        return 0.0, 0.0
    labels = labels.long()
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0, 0.0

    labels = labels.clone()
    labels[labels < 0] = 0
    labels = labels % centers.size(0)
    centers_for_batch = centers[labels]

    hash_batch = hash_batch[valid]
    centers_for_batch = centers_for_batch[valid]

    center_loss = F.binary_cross_entropy(hash_batch, centers_for_batch, reduction="none").mean()

    if neg_k <= 0:
        return center_loss, 0.0

    num_centers = centers.size(0)
    if num_centers <= 1:
        return center_loss, 0.0

    batch_size = hash_batch.size(0)
    neg_k = min(neg_k, num_centers-1)
    neg_indices = []
    for label in labels[valid]:
        label_idx = int(label.item())
        pool = torch.cat([
            torch.arange(0, label_idx, device=centers.device),
            torch.arange(label_idx + 1, num_centers, device=centers.device),
        ])
        if pool.numel() > neg_k:
            perm = torch.randperm(pool.numel(), device=centers.device)[:neg_k]
            pool = pool[perm]
        neg_indices.append(pool)
    neg_indices = torch.stack(neg_indices, dim=0)
    neg_centers = centers[neg_indices]

    hash_exp = hash_batch.unsqueeze(1).expand(-1, neg_k, -1)
    neg_loss = F.binary_cross_entropy(hash_exp, neg_centers, reduction="none").mean(dim=2)
    distinct_loss = -neg_loss.mean()

    return center_loss, distinct_loss


def hamming_margin_loss(hash1, hash2, delta: float, mu: float):
    """Hamming margin loss using L1 distance on continuous hashes."""
    if hash1 is None or hash2 is None:
        return 0.0
    if delta <= 0:
        return 0.0
    if mu <= 0:
        mu = 2 * delta

    pos_dist = torch.abs(hash1 - hash2).sum(dim=1)
    intra = F.relu(pos_dist - delta).pow(2).mean()

    if hash1.size(0) < 2:
        return intra
    pairwise = torch.abs(hash1.unsqueeze(1) - hash1.unsqueeze(0)).sum(dim=2)
    mask = ~torch.eye(hash1.size(0), device=hash1.device, dtype=torch.bool)
    neg_dist = pairwise[mask]
    inter = F.relu(mu - neg_dist).pow(2).mean()

    return intra + inter


# ============================================================================
# Dataset
# ============================================================================

def _stable_label_hash(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big", signed=False)
    return value & ((1 << 63) - 1)


class FlatImageDataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        list_file=None,
        max_images=None,
        seed=0,
        cache_ram=False,
        label_mode="none",
        label_regex=None,
        label_regex_group=1,
    ):
        self.label_regex = re.compile(label_regex) if label_regex else None
        self.label_regex_group = label_regex_group
        self.paths, raw_labels = self._load_paths(root, list_file, label_mode)
        self.raw_labels = raw_labels
        if max_images and max_images > 0 and len(self.paths) > max_images:
            rng = random.Random(seed)
            rng.shuffle(self.paths)
            self.paths = self.paths[:max_images]
            raw_labels = raw_labels[:max_images]

        self.transform = transform
        self.cache_ram = cache_ram
        self.images = [None] * len(self.paths)
        self.label_map = {}
        self.labels = self._encode_labels(raw_labels, label_mode)
        self.num_classes = len(self.label_map)
        if self.cache_ram:
            self._cache_images()

    def _extract_label(self, path, label_mode, label):
        if label is None and label_mode in ("list", "auto") and self.label_regex:
            match = self.label_regex.search(path.name)
            if match:
                try:
                    label = match.group(self.label_regex_group)
                except IndexError:
                    label = None
        if label is None and label_mode in ("parent", "auto"):
            label = path.parent.name
        return label

    def _load_paths(self, root, list_file, label_mode):
        paths = []
        labels = []
        if list_file:
            base = Path(list_file).resolve().parent
            raw_lines = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
            for line in raw_lines:
                if line.startswith("#"):
                    continue
                parts = line.split()
                p = parts[0]
                path = Path(p)
                if not path.is_absolute():
                    path = (base / path).resolve()
                label = None
                if label_mode in ("list", "auto") and len(parts) > 1:
                    label = parts[1]
                label = self._extract_label(path, label_mode, label)
                paths.append(path)
                labels.append(label)

        if not paths and root:
            root_path = Path(root)
            if root_path.exists():
                for ext in IMAGE_EXTS:
                    for p in root_path.rglob(f"*{ext}"):
                        label = None
                        label = self._extract_label(p, label_mode, label)
                        paths.append(p)
                        labels.append(label)

        if not paths:
            raise ValueError("No images found. Provide --data or --data-list")

        return [str(p) for p in paths], labels

    def _encode_labels(self, raw_labels, label_mode):
        if label_mode == "none":
            return [-1 for _ in raw_labels]
        labels = []
        for label in raw_labels:
            if label is None:
                labels.append(-1)
                continue
            label_str = str(label)
            if label_str not in self.label_map:
                self.label_map[label_str] = len(self.label_map)
            labels.append(_stable_label_hash(label_str))
        return labels

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
        return self.transform(img), self.labels[idx]


class SingleViewTransform:
    """Returns a single augmented view."""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, img):
        return self.transform(img)


class TwoViewTransform:
    """Returns two augmented views of the same image for SimCLR."""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Training Steps
# ============================================================================

def train_step_trainable_two_view(
    view1, view2, teacher, student, proj_head,
    optimizer, scheduler, scaler, ema,
    args, device, effective_contrast_weight, update_step,
    labels, centers,
):
    """Paper-exact training with TRAINABLE encoder and TWO views."""
    batch_size = view1.size(0)
    use_amp = args.amp and device.type == "cuda"

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            teacher_out = teacher(view1)
            teacher_emb = teacher_out.last_hidden_state[:, 0]
            target = torch.sigmoid(proj_head(teacher_emb))

    y1 = student.y_init.expand(batch_size, -1).to(device)
    z1 = student.z_init.expand(batch_size, -1).to(device)
    y2 = student.y_init.expand(batch_size, -1).to(device)
    z2 = student.z_init.expand(batch_size, -1).to(device)

    align_losses = []
    total_losses = []
    contrast_loss_val = 0.0
    center_loss_val = 0.0
    distinct_loss_val = 0.0
    quant_loss_val = 0.0
    margin_loss_val = 0.0
    final_step_executed = -1

    for sup_step in range(args.n_sup):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            x1 = student.image_encoder(view1)
            x2 = student.image_encoder(view2)

            y1_curr, z1_curr = y1, z1
            y2_curr, z2_curr = y2, z2

            with torch.no_grad():
                for _ in range(student.t - 1):
                    y1_curr, z1_curr = student.latent_recursion(x1.detach(), y1_curr, z1_curr)
                    y2_curr, z2_curr = student.latent_recursion(x2.detach(), y2_curr, z2_curr)

            y1_new, z1_new = student.latent_recursion(x1, y1_curr, z1_curr)
            y2_new, z2_new = student.latent_recursion(x2, y2_curr, z2_curr)

            hash1 = torch.sigmoid(student.output_head(y1_new))
            hash2 = torch.sigmoid(student.output_head(y2_new))

            align_loss = 0.5 * (
                F.binary_cross_entropy(hash1, target)
                + F.binary_cross_entropy(hash2, target)
            )

            step_loss = align_loss

            if args.center_weight > 0 or args.distinct_weight > 0:
                c1, d1 = hash_center_losses(hash1, labels, centers, args.center_neg_k)
                c2, d2 = hash_center_losses(hash2, labels, centers, args.center_neg_k)
                center_loss = 0.5 * (c1 + c2)
                distinct_loss = 0.5 * (d1 + d2)
                if args.center_weight > 0:
                    step_loss = step_loss + args.center_weight * center_loss
                    center_loss_val = float(center_loss)
                if args.distinct_weight > 0:
                    step_loss = step_loss + args.distinct_weight * distinct_loss
                    distinct_loss_val = float(distinct_loss)

            if args.quant_weight > 0:
                quant_loss = 0.5 * (quantization_loss(hash1) + quantization_loss(hash2))
                step_loss = step_loss + args.quant_weight * quant_loss
                quant_loss_val = float(quant_loss)

            is_last_scheduled = (sup_step == args.n_sup - 1)
            act_triggered = args.use_act and align_loss.item() < args.act_threshold
            is_final = is_last_scheduled or act_triggered

            if is_final:
                final_step_executed = sup_step
                if effective_contrast_weight > 0:
                    contrast_loss = simclr_loss(
                        hash1,
                        hash2,
                        temperature=args.nce_temperature,
                        hard_neg_k=args.hard_neg_k,
                    )
                    log_2b = torch.log(torch.tensor(2.0 * batch_size, device=device))
                    contrast_loss_normalized = contrast_loss / log_2b
                    step_loss = step_loss + effective_contrast_weight * contrast_loss_normalized
                    contrast_loss_val = contrast_loss_normalized.item()

                if args.margin_weight > 0:
                    margin_loss = hamming_margin_loss(
                        hash1, hash2, args.hamming_delta, args.hamming_mu,
                    )
                    step_loss = step_loss + args.margin_weight * margin_loss
                    margin_loss_val = float(margin_loss)

        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        if ema is not None:
            ema.update(student)

        align_losses.append(align_loss.item())
        total_losses.append(step_loss.item())
        update_step += 1

        y1, z1 = y1_new.detach(), z1_new.detach()
        y2, z2 = y2_new.detach(), z2_new.detach()

        if act_triggered:
            break

    n_steps = len(align_losses)
    return {
        "total": sum(total_losses) / n_steps,
        "align": sum(align_losses) / n_steps,
        "contrast": contrast_loss_val,
        "center": center_loss_val,
        "distinct": distinct_loss_val,
        "quant": quant_loss_val,
        "margin": margin_loss_val,
        "n_steps": n_steps,
        "final_step": final_step_executed,
        "update_step": update_step,
    }


def train_step_frozen_two_view(
    view1, view2, teacher, student, proj_head,
    optimizer, scheduler, scaler, ema,
    args, device, effective_contrast_weight, update_step,
    labels, centers,
):
    """Paper-exact training with FROZEN encoder and TWO views."""
    batch_size = view1.size(0)
    use_amp = args.amp and device.type == "cuda"

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            teacher_out = teacher(view1)
            teacher_emb = teacher_out.last_hidden_state[:, 0]
            target = torch.sigmoid(proj_head(teacher_emb))
            x1 = student.image_encoder(view1)
            x2 = student.image_encoder(view2)

    y1 = student.y_init.expand(batch_size, -1).to(device)
    z1 = student.z_init.expand(batch_size, -1).to(device)
    y2 = student.y_init.expand(batch_size, -1).to(device)
    z2 = student.z_init.expand(batch_size, -1).to(device)

    align_losses = []
    total_losses = []
    contrast_loss_val = 0.0
    center_loss_val = 0.0
    distinct_loss_val = 0.0
    quant_loss_val = 0.0
    margin_loss_val = 0.0
    final_step_executed = -1

    for sup_step in range(args.n_sup):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            y1_curr, z1_curr = y1, z1
            y2_curr, z2_curr = y2, z2

            with torch.no_grad():
                for _ in range(student.t - 1):
                    y1_curr, z1_curr = student.latent_recursion(x1, y1_curr, z1_curr)
                    y2_curr, z2_curr = student.latent_recursion(x2, y2_curr, z2_curr)

            y1_new, z1_new = student.latent_recursion(x1, y1_curr, z1_curr)
            y2_new, z2_new = student.latent_recursion(x2, y2_curr, z2_curr)

            hash1 = torch.sigmoid(student.output_head(y1_new))
            hash2 = torch.sigmoid(student.output_head(y2_new))

            align_loss = 0.5 * (
                F.binary_cross_entropy(hash1, target)
                + F.binary_cross_entropy(hash2, target)
            )

            step_loss = align_loss

            if args.center_weight > 0 or args.distinct_weight > 0:
                c1, d1 = hash_center_losses(hash1, labels, centers, args.center_neg_k)
                c2, d2 = hash_center_losses(hash2, labels, centers, args.center_neg_k)
                center_loss = 0.5 * (c1 + c2)
                distinct_loss = 0.5 * (d1 + d2)
                if args.center_weight > 0:
                    step_loss = step_loss + args.center_weight * center_loss
                    center_loss_val = float(center_loss)
                if args.distinct_weight > 0:
                    step_loss = step_loss + args.distinct_weight * distinct_loss
                    distinct_loss_val = float(distinct_loss)

            if args.quant_weight > 0:
                quant_loss = 0.5 * (quantization_loss(hash1) + quantization_loss(hash2))
                step_loss = step_loss + args.quant_weight * quant_loss
                quant_loss_val = float(quant_loss)

            is_last_scheduled = (sup_step == args.n_sup - 1)
            act_triggered = args.use_act and align_loss.item() < args.act_threshold
            is_final = is_last_scheduled or act_triggered

            if is_final:
                final_step_executed = sup_step
                if effective_contrast_weight > 0:
                    contrast_loss = simclr_loss(
                        hash1,
                        hash2,
                        temperature=args.nce_temperature,
                        hard_neg_k=args.hard_neg_k,
                    )
                    log_2b = torch.log(torch.tensor(2.0 * batch_size, device=device))
                    contrast_loss_normalized = contrast_loss / log_2b
                    step_loss = step_loss + effective_contrast_weight * contrast_loss_normalized
                    contrast_loss_val = contrast_loss_normalized.item()

                if args.margin_weight > 0:
                    margin_loss = hamming_margin_loss(
                        hash1, hash2, args.hamming_delta, args.hamming_mu,
                    )
                    step_loss = step_loss + args.margin_weight * margin_loss
                    margin_loss_val = float(margin_loss)

        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        if ema is not None:
            ema.update(student)

        align_losses.append(align_loss.item())
        total_losses.append(step_loss.item())
        update_step += 1

        y1, z1 = y1_new.detach(), z1_new.detach()
        y2, z2 = y2_new.detach(), z2_new.detach()

        if act_triggered:
            break

    n_steps = len(align_losses)
    return {
        "total": sum(total_losses) / n_steps,
        "align": sum(align_losses) / n_steps,
        "contrast": contrast_loss_val,
        "center": center_loss_val,
        "distinct": distinct_loss_val,
        "quant": quant_loss_val,
        "margin": margin_loss_val,
        "n_steps": n_steps,
        "final_step": final_step_executed,
        "update_step": update_step,
    }


def train_step_single_view(
    view1, teacher, student, proj_head,
    optimizer, scheduler, scaler, ema,
    args, device, update_step, freeze_encoder,
    labels, centers,
):
    """Single-view training (no SimCLR)."""
    batch_size = view1.size(0)
    use_amp = args.amp and device.type == "cuda"

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            teacher_out = teacher(view1)
            teacher_emb = teacher_out.last_hidden_state[:, 0]
            target = torch.sigmoid(proj_head(teacher_emb))
            if freeze_encoder:
                x1 = student.image_encoder(view1)

    y1 = student.y_init.expand(batch_size, -1).to(device)
    z1 = student.z_init.expand(batch_size, -1).to(device)

    align_losses = []
    total_losses = []
    center_loss_val = 0.0
    distinct_loss_val = 0.0
    quant_loss_val = 0.0
    margin_loss_val = 0.0
    final_step_executed = -1

    for sup_step in range(args.n_sup):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if not freeze_encoder:
                x1 = student.image_encoder(view1)

            y1_curr, z1_curr = y1, z1

            with torch.no_grad():
                for _ in range(student.t - 1):
                    if freeze_encoder:
                        y1_curr, z1_curr = student.latent_recursion(x1, y1_curr, z1_curr)
                    else:
                        y1_curr, z1_curr = student.latent_recursion(x1.detach(), y1_curr, z1_curr)

            y1_new, z1_new = student.latent_recursion(x1, y1_curr, z1_curr)
            hash1 = torch.sigmoid(student.output_head(y1_new))

            align_loss = F.binary_cross_entropy(hash1, target)
            step_loss = align_loss

            if args.center_weight > 0 or args.distinct_weight > 0:
                center_loss, distinct_loss = hash_center_losses(
                    hash1, labels, centers, args.center_neg_k,
                )
                if args.center_weight > 0:
                    step_loss = step_loss + args.center_weight * center_loss
                    center_loss_val = float(center_loss)
                if args.distinct_weight > 0:
                    step_loss = step_loss + args.distinct_weight * distinct_loss
                    distinct_loss_val = float(distinct_loss)

            if args.quant_weight > 0:
                quant_loss = quantization_loss(hash1)
                step_loss = step_loss + args.quant_weight * quant_loss
                quant_loss_val = float(quant_loss)

            is_last_scheduled = (sup_step == args.n_sup - 1)
            act_triggered = args.use_act and align_loss.item() < args.act_threshold
            is_final = is_last_scheduled or act_triggered

            if is_final:
                final_step_executed = sup_step
                if args.margin_weight > 0:
                    margin_loss_val = 0.0

        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        if ema is not None:
            ema.update(student)

        align_losses.append(align_loss.item())
        total_losses.append(step_loss.item())
        update_step += 1

        y1, z1 = y1_new.detach(), z1_new.detach()

        if act_triggered:
            break

    n_steps = len(align_losses)
    return {
        "total": sum(total_losses) / n_steps,
        "align": sum(align_losses) / n_steps,
        "contrast": 0.0,
        "center": center_loss_val,
        "distinct": distinct_loss_val,
        "quant": quant_loss_val,
        "margin": margin_loss_val,
        "n_steps": n_steps,
        "final_step": final_step_executed,
        "update_step": update_step,
    }


# ============================================================================
# Helpers
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_backends(args, device):
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _get_base_transform(args, processor):
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    return transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize,
    ])


def _create_dataloader(args, use_two_view):
    processor = AutoImageProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    args._norm_mean = list(processor.image_mean)
    args._norm_std = list(processor.image_std)

    base_transform = _get_base_transform(args, processor)
    if use_two_view:
        transform = TwoViewTransform(base_transform)
    else:
        transform = SingleViewTransform(base_transform)

    dataset = FlatImageDataset(
        root=args.data if args.data else None,
        transform=transform,
        list_file=args.data_list if args.data_list else None,
        max_images=args.max_images if args.max_images > 0 else None,
        seed=args.seed,
        cache_ram=args.cache_ram,
        label_mode=args.label_mode,
        label_regex=args.label_regex if args.label_regex else None,
        label_regex_group=args.label_regex_group,
    )

    if args.label_mode != "none":
        total = len(dataset.raw_labels)
        known_labels = [str(label) for label in dataset.raw_labels if label is not None]
        unlabeled = total - len(known_labels)
        unique = len(set(known_labels))
        pct_unlabeled = (unlabeled / total * 100.0) if total > 0 else 0.0
        print(
            f"Label stats: {unique} unique, "
            f"{unlabeled}/{total} unlabeled ({pct_unlabeled:.1f}%)."
        )
        if known_labels:
            top = Counter(known_labels).most_common(5)
            top_str = ", ".join(f"{label}:{count}" for label, count in top)
            print(f"Top labels: {top_str}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    print(f"Dataset: {len(dataset)} images, {len(loader)} batches")
    return loader, dataset


def _create_models(args, device):
    print(f"Loading teacher: {args.teacher}")
    teacher = AutoModel.from_pretrained(args.teacher, trust_remote_code=True).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher_dim = teacher.config.hidden_size

    print("Creating TRM student...")
    student = TRMHasher(
        embed_dim=args.embed_dim,
        hash_dim=args.hash_dim,
        n_layers=args.n_layers,
        n_latent=args.n_latent,
        t=args.t,
    ).to(device)

    if args.freeze_encoder:
        print(">>> Image encoder FROZEN <<<")
        for param in student.image_encoder.parameters():
            param.requires_grad = False

    if args.channels_last:
        student = student.to(memory_format=torch.channels_last)

    proj_head = nn.Linear(teacher_dim, args.hash_dim, bias=False).to(device)
    nn.init.orthogonal_(proj_head.weight)
    proj_head.requires_grad_(False)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        safetensors.torch.load_model(student, args.resume)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"Student: {trainable:,} trainable / {total:,} total params")

    return teacher, student, proj_head


def _create_optimizer_and_scheduler(args, student, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_update_steps = args.epochs * steps_per_epoch * args.n_sup
    warmup_batches = args.warmup_batches if args.warmup_batches > 0 else args.warmup_steps
    warmup_updates = warmup_batches * args.n_sup

    print(f"Scheduler: {total_update_steps:,} total updates, {warmup_updates:,} warmup")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_update_steps,
        pct_start=min(warmup_updates / total_update_steps, 0.1) if total_update_steps > 0 else 0.1,
    )

    return optimizer, scheduler


def save_checkpoint(model, proj_head, ema, save_dir, filename, args):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename
    if ema is not None:
        ema.apply_shadow(model)
    safetensors.torch.save_model(model, str(save_path))
    if ema is not None:
        ema.restore(model)
    print(f"Saved: {save_path}")

    if args is not None and hasattr(args, "_norm_mean"):
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
        config_path = Path(save_dir) / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


# ============================================================================
# Epoch Runner
# ============================================================================

def _run_epoch(
    epoch, loader, teacher, student, proj_head,
    optimizer, scheduler, scaler, ema,
    args, device, batch_step, update_step, use_two_view,
    centers,
):
    student.train()
    epoch_total_loss = 0.0
    epoch_align_loss = 0.0
    last_collision = None

    effective_contrast_weight = (
        args.contrast_weight if epoch >= args.contrast_warmup_epochs else 0.0
    )

    if use_two_view:
        train_fn = train_step_frozen_two_view if args.freeze_encoder else train_step_trainable_two_view
    else:
        train_fn = None

    for batch_idx, batch in enumerate(loader, start=1):
        if use_two_view:
            labels = None
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]):
                views, labels = batch
            else:
                views = batch
            view1, view2 = views
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            if args.channels_last:
                view1 = view1.contiguous(memory_format=torch.channels_last)
                view2 = view2.contiguous(memory_format=torch.channels_last)

            losses = train_fn(
                view1, view2, teacher, student, proj_head,
                optimizer, scheduler, scaler, ema,
                args, device, effective_contrast_weight, update_step,
                labels, centers,
            )
        else:
            labels = None
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]):
                view1, labels = batch
            else:
                view1 = batch
            view1 = view1.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            if args.channels_last:
                view1 = view1.contiguous(memory_format=torch.channels_last)

            losses = train_step_single_view(
                view1, teacher, student, proj_head,
                optimizer, scheduler, scaler, ema,
                args, device, update_step, args.freeze_encoder,
                labels, centers,
            )

        update_step = losses["update_step"]
        batch_step += 1

        epoch_total_loss += losses["total"]
        epoch_align_loss += losses["align"]

        if args.collision_interval > 0 and batch_step % args.collision_interval == 0:
            with torch.no_grad():
                test_view = view1
                hashes = student.inference(test_view, n_sup=args.n_sup)
                collisions = count_hash_collisions(hashes, threshold=args.collision_threshold)
                entropy = hash_entropy_estimate(hashes).item()
                last_collision = (collisions, entropy)

        if batch_step % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            line = (
                f"E{epoch+1} B{batch_idx} U{update_step} "
                f"tot={losses['total']:.4f} ali={losses['align']:.4f} "
                f"ctr={losses['center']:.4f} dst={losses['distinct']:.4f} "
                f"q={losses['quant']:.4f} m={losses['margin']:.4f} "
                f"con={losses['contrast']:.4f} "
                f"sup={losses['n_steps']} lr={lr:.2e}"
            )
            if last_collision is not None:
                line += f" col={last_collision[0]} ent={last_collision[1]:.1f}"
                last_collision = None
            print(line)

        if args.checkpoint_dir and args.checkpoint_every > 0:
            if update_step % args.checkpoint_every == 0:
                save_checkpoint(
                    student, proj_head, ema, args.checkpoint_dir,
                    f"student_u{update_step}.safetensors", args,
                )

        if args.max_updates and update_step >= args.max_updates:
            break

    return epoch_total_loss, epoch_align_loss, batch_step, update_step


# ============================================================================
# Main
# ============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _configure_backends(args, device)

    use_two_view = args.contrast_weight > 0 or args.margin_weight > 0 or args.force_two_view
    mode_str = "two-view" if use_two_view else "single-view"

    loader, dataset = _create_dataloader(args, use_two_view)
    steps_per_epoch = len(loader)

    teacher, student, proj_head = _create_models(args, device)
    optimizer, scheduler = _create_optimizer_and_scheduler(args, student, steps_per_epoch)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = EMA(student, decay=args.ema_decay) if args.use_ema else None

    print("\n" + "=" * 70)
    print("TRM Training - Paper-Exact")
    print("=" * 70)
    print(f"Mode: {mode_str}")
    print(f"Encoder: {'FROZEN' if args.freeze_encoder else 'TRAINABLE'}")
    print(f"EMA: {'ON' if args.use_ema else 'OFF'} (decay={args.ema_decay})")
    print(f"Hash: {args.hash_dim}d, Embed: {args.embed_dim}d")
    print(f"TRM: n_sup={args.n_sup}, t={args.t}, n_latent={args.n_latent}")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Updates/batch: {args.n_sup}, Total updates: ~{args.epochs * steps_per_epoch * args.n_sup:,}")
    print(
        "Losses: "
        f"align(BCE) + C={args.center_weight} D={args.distinct_weight} "
        f"Q={args.quant_weight} M={args.margin_weight} "
        f"con={args.contrast_weight}"
    )
    print(f"ACT: {args.use_act} (thr={args.act_threshold})")
    print(f"Hard negatives: k={args.hard_neg_k}")
    print("=" * 70 + "\n")

    centers = None
    if args.center_weight > 0 or args.distinct_weight > 0:
        if dataset.num_classes <= 0:
            print("WARNING: No labels available; hash-center losses will be disabled.")
        else:
            max_centers = 2 * args.hash_dim
            num_centers = args.num_centers if args.num_centers > 0 else dataset.num_classes
            if num_centers > max_centers:
                print(
                    f"WARNING: num_centers={num_centers} exceeds Hadamard limit "
                    f"({max_centers}); using {max_centers} with hash(label)%num_centers mapping."
                )
                num_centers = max_centers
            try:
                centers = build_hash_centers(
                    num_centers=num_centers,
                    hash_dim=args.hash_dim,
                    device=device,
                    seed=args.center_seed,
                )
                print(f"Hash centers: {num_centers} (classes={dataset.num_classes})")
            except ValueError as exc:
                print(f"WARNING: hash-center disabled: {exc}")
                centers = None

    batch_step = 0
    update_step = 0

    for epoch in range(args.epochs):
        epoch_total, epoch_align, batch_step, update_step = _run_epoch(
            epoch, loader, teacher, student, proj_head,
            optimizer, scheduler, scaler, ema,
            args, device, batch_step, update_step, use_two_view, centers,
        )

        n_batches = len(loader)
        print(
            f"\n>>> Epoch {epoch+1}/{args.epochs} done. "
            f"Avg total={epoch_total/n_batches:.4f} align={epoch_align/n_batches:.4f}\n"
        )

        if args.checkpoint_dir:
            save_checkpoint(
                student, proj_head, ema, args.checkpoint_dir,
                f"student_e{epoch+1}.safetensors", args,
            )

        if args.max_updates and update_step >= args.max_updates:
            print(f"Reached max_updates={args.max_updates}, stopping.")
            break

    if args.checkpoint_dir:
        save_checkpoint(student, proj_head, ema, args.checkpoint_dir, "student_final.safetensors", args)

    print(f"\nTraining complete! Batches={batch_step}, Updates={update_step}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRM Training - Paper Exact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", default="", help="Image folder path")
    data_group.add_argument("--data-list", default="", help="Text file with image paths (optional: path label)")
    data_group.add_argument("--max-images", type=int, default=0, help="Limit images (0=all)")
    data_group.add_argument("--cache-ram", action="store_true", help="Cache images in RAM")
    data_group.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    data_group.add_argument(
        "--label-mode",
        choices=["none", "parent", "list", "auto"],
        default="list",
        help="Label source for hash centers (manifest labels/regex, parent dir, or none)",
    )
    data_group.add_argument(
        "--label-regex",
        default=r"^((?:ffhq|openimages|openimg|mobileview)_\d+|\d+)",
        help="Regex to extract image_id from filename when manifest labels are omitted",
    )
    data_group.add_argument(
        "--label-regex-group",
        type=int,
        default=1,
        help="Capture group index to use for --label-regex",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--teacher", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    model_group.add_argument("--embed-dim", type=int, default=256)
    model_group.add_argument("--hash-dim", type=int, default=128)
    model_group.add_argument("--n-layers", type=int, default=2)
    model_group.add_argument("--n-latent", type=int, default=6)
    model_group.add_argument("--t", type=int, default=3)
    model_group.add_argument("--n-sup", type=int, default=16)
    model_group.add_argument("--freeze-encoder", action="store_true", help="Freeze image encoder")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=10)
    train_group.add_argument("--batch-size", type=int, default=128)
    train_group.add_argument("--image-size", type=int, default=224)
    train_group.add_argument("--lr", type=float, default=3e-5)
    train_group.add_argument("--weight-decay", type=float, default=0.1)
    train_group.add_argument("--warmup-steps", type=int, default=2000,
                             help="Warmup in batches (scaled by n_sup internally)")
    train_group.add_argument("--warmup-batches", type=int, default=0,
                             help="Override warmup steps (0 uses warmup-steps)")
    train_group.add_argument("--use-ema", action="store_true", help="Enable EMA")
    train_group.add_argument("--ema-decay", type=float, default=0.999)

    act_group = parser.add_argument_group("ACT")
    act_group.add_argument("--use-act", action="store_true", help="Enable early stopping")
    act_group.add_argument("--act-threshold", type=float, default=0.01)

    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--contrast-weight", type=float, default=0.3,
                            help="SimCLR weight (0=single-view mode)")
    loss_group.add_argument("--nce-temperature", type=float, default=0.07)
    loss_group.add_argument("--contrast-warmup-epochs", type=int, default=1)
    loss_group.add_argument("--center-weight", type=float, default=1.0,
                            help="Hash center loss weight (L_C)")
    loss_group.add_argument("--distinct-weight", type=float, default=0.5,
                            help="Distinct center loss weight (L_D)")
    loss_group.add_argument("--quant-weight", type=float, default=0.1,
                            help="Quantization loss weight (L_Q)")
    loss_group.add_argument("--margin-weight", type=float, default=0.3,
                            help="Hamming margin loss weight (L_M)")
    loss_group.add_argument("--hamming-delta", type=float, default=7.0,
                            help="Intra-class Hamming threshold (delta)")
    loss_group.add_argument("--hamming-mu", type=float, default=0.0,
                            help="Inter-class Hamming margin (mu); 0 uses 2*delta")
    loss_group.add_argument("--center-neg-k", type=int, default=8,
                            help="Negative centers sampled per batch item")
    loss_group.add_argument("--num-centers", type=int, default=0,
                            help="Override number of hash centers (0=auto)")
    loss_group.add_argument("--center-seed", type=int, default=42,
                            help="Seed for Hadamard center permutation")
    loss_group.add_argument("--hard-neg-k", type=int, default=0,
                            help="Top-k hard negatives for SimCLR (0=all)")
    loss_group.add_argument("--force-two-view", action="store_true",
                            help="Force two-view even if contrast_weight=0")

    mon_group = parser.add_argument_group("Monitoring")
    mon_group.add_argument("--collision-interval", type=int, default=200,
                           help="Check collisions every N batches (0=off)")
    mon_group.add_argument("--collision-threshold", type=float, default=0.99)

    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--device", default="cuda")
    infra_group.add_argument("--amp", action="store_true")
    infra_group.add_argument("--channels-last", action="store_true")
    infra_group.add_argument("--allow-tf32", action="store_true")
    infra_group.add_argument("--cudnn-benchmark", action="store_true")
    infra_group.add_argument("--pin-memory", action="store_true")
    infra_group.add_argument("--prefetch-factor", type=int, default=2)
    infra_group.add_argument("--seed", type=int, default=42)
    infra_group.add_argument("--log-interval", type=int, default=50)
    infra_group.add_argument("--checkpoint-dir", default="./checkpoints")
    infra_group.add_argument("--checkpoint-every", type=int, default=10000,
                             help="Checkpoint every N updates")
    infra_group.add_argument("--max-updates", type=int, default=0,
                             help="Stop after N updates (0=off)")
    infra_group.add_argument("--max-steps", type=int, default=0,
                             help="Alias for max-updates")
    infra_group.add_argument("--resume", default="", help="Resume from checkpoint")

    args = parser.parse_args()

    if not args.data and not args.data_list:
        parser.error("Must provide --data or --data-list (or both)")

    if args.max_updates <= 0 and args.max_steps > 0:
        args.max_updates = args.max_steps

    train(args)


if __name__ == "__main__":
    main()
