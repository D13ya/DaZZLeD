"""
Hash Center Training with ResNet Backbone (LOGIC2)

This script trains a ResNet encoder + projection head + hash layer to map images
to fixed hash centers using center + distinct + quantization losses. The model
emits logits; apply sigmoid to obtain hashes in (0,1) for inference.
"""

import argparse
import hashlib
import io
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.counterfactual_vae import CounterfactualVAE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_backends(args, device: torch.device) -> None:
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _normalize_batch(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _denormalize_batch(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return batch * std + mean


def quantization_loss(hash_batch: torch.Tensor) -> torch.Tensor:
    """Quantization loss (log cosh) to push hashes toward {0,1}."""
    return (1 - torch.log(torch.cosh(torch.abs(2 * hash_batch - 1)))).mean()


def simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent loss for counterfactual contrastive learning."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -float("inf"))
    pos = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(sim, pos)


def dhd_loss(hash1: torch.Tensor, hash2: torch.Tensor) -> torch.Tensor:
    """Deep hash distillation loss (cosine distance)."""
    h1 = F.normalize(hash1, dim=1)
    h2 = F.normalize(hash2, dim=1)
    return 1.0 - (h1 * h2).sum(dim=1).mean()


def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError("hash_dim must be power-of-two for Hadamard centers")
    h = torch.tensor([[1.0]], device=device)
    while h.size(0) < size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def build_hash_centers(num_centers: int, hash_dim: int, device: torch.device, seed: int) -> torch.Tensor:
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
    return (h[:num_centers] + 1.0) / 2.0


def _label_seed(label: int) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _random_centers_for_labels(labels: torch.Tensor, hash_dim: int, device: torch.device) -> torch.Tensor:
    centers = []
    for label in labels.tolist():
        g = torch.Generator()
        g.manual_seed(_label_seed(int(label)))
        vec = torch.rand(hash_dim, generator=g)
        centers.append((vec > 0.5).float())
    return torch.stack(centers, dim=0).to(device)


def build_ghost_centers(num_centers: int, hash_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    if num_centers <= 0:
        return torch.empty((0, hash_dim), device=device)
    g = torch.Generator()
    g.manual_seed(seed)
    vec = torch.rand((num_centers, hash_dim), generator=g)
    return (vec > 0.5).float().to(device)


def _sample_negative_indices(label_indices, num_centers: int, neg_k: int, device):
    neg_indices = []
    for label_idx in label_indices.tolist():
        pool = torch.cat([
            torch.arange(0, label_idx, device=device),
            torch.arange(label_idx + 1, num_centers, device=device),
        ])
        if pool.numel() > neg_k:
            perm = torch.randperm(pool.numel(), device=device)[:neg_k]
            pool = pool[perm]
        neg_indices.append(pool)
    return torch.stack(neg_indices, dim=0)


def _center_loss_and_indices(hash_logits, labels_valid, centers, center_mode):
    if center_mode == "random":
        unique_labels, inv = torch.unique(labels_valid, return_inverse=True, dim=0)
        centers_unique = _random_centers_for_labels(unique_labels.cpu(), hash_logits.size(1), hash_logits.device)
        centers_for_batch = centers_unique[inv]
        center_loss = F.binary_cross_entropy_with_logits(
            hash_logits, centers_for_batch, reduction="none",
        ).mean()
        return center_loss, inv, centers_unique
    if centers is None:
        return None, None, None
    labels_mod = labels_valid % centers.size(0)
    centers_for_batch = centers[labels_mod]
    center_loss = F.binary_cross_entropy_with_logits(
        hash_logits, centers_for_batch, reduction="none",
    ).mean()
    return center_loss, labels_mod, centers


def _distinct_center_loss(hash_logits, label_indices, centers_for_neg, neg_k: int, ghost_centers: Optional[torch.Tensor]):
    num_centers = centers_for_neg.size(0)
    if num_centers <= 1:
        return torch.zeros((), device=hash_logits.device)
    neg_k = min(neg_k, num_centers - 1)

    neg_losses = []
    if neg_k == 0:
        logits_exp = hash_logits.unsqueeze(1).expand(-1, num_centers, -1)
        centers_exp = centers_for_neg.unsqueeze(0).expand(hash_logits.size(0), -1, -1)
        full_loss = F.binary_cross_entropy_with_logits(
            logits_exp, centers_exp, reduction="none",
        ).mean(dim=2)
        mask = torch.ones_like(full_loss, dtype=torch.bool)
        mask.scatter_(1, label_indices.unsqueeze(1), False)
        neg_loss = full_loss[mask].view(hash_logits.size(0), num_centers - 1)
        neg_losses.append(neg_loss)
    elif neg_k > 0:
        neg_indices = _sample_negative_indices(label_indices, num_centers, neg_k, hash_logits.device)
        neg_centers = centers_for_neg[neg_indices]
        logits_exp = hash_logits.unsqueeze(1).expand(-1, neg_k, -1)
        neg_loss = F.binary_cross_entropy_with_logits(
            logits_exp, neg_centers, reduction="none",
        ).mean(dim=2)
        neg_losses.append(neg_loss)

    if ghost_centers is not None and ghost_centers.numel() > 0:
        ghost_centers = ghost_centers.to(hash_logits.device)
        ghost_exp = hash_logits.unsqueeze(1).expand(-1, ghost_centers.size(0), -1)
        ghost_targets = ghost_centers.unsqueeze(0).expand(hash_logits.size(0), -1, -1)
        ghost_loss = F.binary_cross_entropy_with_logits(
            ghost_exp, ghost_targets, reduction="none",
        ).mean(dim=2)
        neg_losses.append(ghost_loss)

    if not neg_losses:
        return torch.zeros((), device=hash_logits.device)

    neg_loss = torch.cat(neg_losses, dim=1)
    return neg_loss.mean()


def hash_center_losses(hash_logits, labels, centers, neg_k: int, center_mode: str, ghost_centers: Optional[torch.Tensor]):
    if labels is None:
        zero = torch.zeros((), device=hash_logits.device)
        return zero, zero
    labels = labels.long()
    valid = labels >= 0
    if valid.sum() == 0:
        zero = torch.zeros((), device=hash_logits.device)
        return zero, zero

    labels = labels.clone()
    labels[labels < 0] = 0
    hash_logits = hash_logits[valid]
    labels_valid = labels[valid]

    center_loss, label_indices, centers_for_neg = _center_loss_and_indices(
        hash_logits, labels_valid, centers, center_mode,
    )
    if center_loss is None:
        zero = torch.zeros((), device=hash_logits.device)
        return zero, zero

    distinct_loss = _distinct_center_loss(
        hash_logits, label_indices, centers_for_neg, neg_k, ghost_centers,
    )

    return center_loss, distinct_loss


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
        domain_mode="none",
        domain_regex=None,
        domain_regex_group=1,
    ):
        self.label_regex = re.compile(label_regex) if label_regex else None
        self.label_regex_group = label_regex_group
        self.domain_mode = domain_mode
        self.domain_regex = re.compile(domain_regex) if domain_regex else None
        self.domain_regex_group = domain_regex_group
        paths, raw_labels, raw_domains = self._load_paths(root, list_file, label_mode)

        if max_images and max_images > 0 and len(paths) > max_images:
            rng = random.Random(seed)
            combined = list(zip(paths, raw_labels, raw_domains))
            rng.shuffle(combined)
            combined = combined[:max_images]
            paths, raw_labels, raw_domains = zip(*combined)

        self.paths = list(paths)
        self.raw_labels = list(raw_labels)
        self.raw_domains = list(raw_domains)
        self.transform = transform
        self.cache_ram = cache_ram
        self.images = [None] * len(self.paths)
        self.label_map = {}
        self.labels = self._encode_labels(self.raw_labels, label_mode)
        self.num_classes = len(self.label_map)
        self.domain_map = {}
        self.domain_ids = self._encode_domains(self.raw_domains)
        self.num_domains = len(self.domain_map)
        if self.cache_ram:
            self._cache_images()

    def _extract_label(self, path: Path, label_mode: str, label):
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

    def _extract_domain(self, path: Path, domain):
        if self.domain_mode == "none":
            return None
        if domain is None and self.domain_mode in ("regex", "auto") and self.domain_regex:
            match = self.domain_regex.search(str(path))
            if match:
                try:
                    domain = match.group(self.domain_regex_group)
                except IndexError:
                    domain = None
        if domain is None and self.domain_mode in ("parent", "auto"):
            domain = path.parent.name
        return domain

    def _load_paths_from_list(self, list_file, label_mode):
        paths = []
        labels = []
        domains = []
        base = Path(list_file).resolve().parent
        raw_lines = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
        for line in raw_lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            path = Path(parts[0])
            if not path.is_absolute():
                path = (base / path).resolve()
            label = None
            if label_mode in ("list", "auto") and len(parts) > 1:
                label = parts[1]
            label = self._extract_label(path, label_mode, label)
            domain = self._extract_domain(path, None)
            paths.append(path)
            labels.append(label)
            domains.append(domain)
        return paths, labels, domains

    def _load_paths_from_root(self, root, label_mode):
        paths = []
        labels = []
        domains = []
        root_path = Path(root)
        if not root_path.exists():
            return paths, labels, domains
        for ext in IMAGE_EXTS:
            for p in root_path.rglob(f"*{ext}"):
                label = self._extract_label(p, label_mode, None)
                domain = self._extract_domain(p, None)
                paths.append(p)
                labels.append(label)
                domains.append(domain)
        return paths, labels, domains

    def _load_paths(self, root, list_file, label_mode):
        paths = []
        labels = []
        domains = []
        if list_file:
            paths, labels, domains = self._load_paths_from_list(list_file, label_mode)
        if not paths and root:
            paths, labels, domains = self._load_paths_from_root(root, label_mode)
        if not paths:
            raise ValueError("No images found. Provide --data or --data-list")
        return [str(p) for p in paths], labels, domains

    def _encode_labels(self, raw_labels, label_mode):
        if label_mode == "none":
            return [-1 for _ in raw_labels]
        label_strings = [str(label) if label is not None else None for label in raw_labels]
        unique_labels = sorted({label for label in label_strings if label is not None})
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = []
        for label in label_strings:
            if label is None:
                labels.append(-1)
                continue
            labels.append(self.label_map[label])
        return labels

    def _encode_domains(self, raw_domains):
        domain_strings = [str(domain) if domain is not None else None for domain in raw_domains]
        unique_domains = sorted({domain for domain in domain_strings if domain is not None})
        self.domain_map = {domain: idx for idx, domain in enumerate(unique_domains)}
        domains = []
        for domain in domain_strings:
            if domain is None:
                domains.append(-1)
                continue
            domains.append(self.domain_map[domain])
        return domains

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
        return self.transform(img), self.labels[idx], self.domain_ids[idx]


class SingleViewTransform:
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, img):
        return self.transform(img)


class RandomJPEGCompression:
    def __init__(self, quality_range=(50, 95), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")
        out.load()
        buffer.close()
        return out


class TwoViewTransform:
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class ResNetHashNet(nn.Module):
    def __init__(self, backbone: str, hash_dim: int, proj_dim: int, pretrained: bool):
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            if pretrained:
                try:
                    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                except Exception:
                    resnet = models.resnet18(pretrained=True)
            else:
                try:
                    resnet = models.resnet18(weights=None)
                except Exception:
                    resnet = models.resnet18(pretrained=False)
        elif backbone == "resnet50":
            if pretrained:
                try:
                    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except Exception:
                    resnet = models.resnet50(pretrained=True)
            else:
                try:
                    resnet = models.resnet50(weights=None)
                except Exception:
                    resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = resnet.fc.in_features
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.hash_head = nn.Linear(proj_dim, hash_dim)
        self._init_heads()

    def _init_heads(self) -> None:
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.hash_head.weight, mean=0.0, std=0.01)
        if self.hash_head.bias is not None:
            nn.init.zeros_(self.hash_head.bias)

    def forward(self, x: torch.Tensor, return_proj: bool = False):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        z = self.proj(x)
        logits = self.hash_head(z)
        if return_proj:
            return logits, z
        return logits


def build_backbone(name: str, hash_dim: int, proj_dim: int, pretrained: bool) -> nn.Module:
    return ResNetHashNet(name, hash_dim, proj_dim, pretrained)


def _get_base_transform(args):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if args.no_aug:
        return transforms.Compose([
            transforms.Resize(args.image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
    jpeg = RandomJPEGCompression(p=0.3, quality_range=(50, 95))
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    return transforms.Compose([
        transforms.Resize(args.image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
        jpeg,
        transforms.RandomApply([blur], p=0.3),
        transforms.ToTensor(),
        normalize,
    ])


def _create_dataloader(args, use_two_view: bool):
    base_transform = _get_base_transform(args)
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
        domain_mode=args.domain_mode,
        domain_regex=args.domain_regex if args.domain_regex else None,
        domain_regex_group=args.domain_regex_group,
    )

    if args.label_mode != "none":
        total = len(dataset.raw_labels)
        known_labels = [str(label) for label in dataset.raw_labels if label is not None]
        unlabeled = total - len(known_labels)
        unique = len(set(known_labels))
        pct_unlabeled = (unlabeled / total * 100.0) if total > 0 else 0.0
        print(
            f"Label stats: {unique} unique, {unlabeled}/{total} unlabeled ({pct_unlabeled:.1f}%)."
        )
        if known_labels:
            top = Counter(known_labels).most_common(5)
            top_str = ", ".join(f"{label}:{count}" for label, count in top)
            print(f"Top labels: {top_str}")
    if args.domain_mode != "none":
        total = len(dataset.raw_domains)
        known_domains = [str(domain) for domain in dataset.raw_domains if domain is not None]
        unknown = total - len(known_domains)
        unique = len(set(known_domains))
        pct_unknown = (unknown / total * 100.0) if total > 0 else 0.0
        print(f"Domain stats: {unique} unique, {unknown}/{total} unlabeled ({pct_unknown:.1f}%).")
        if known_domains:
            top = Counter(known_domains).most_common(5)
            top_str = ", ".join(f"{domain}:{count}" for domain, count in top)
            print(f"Top domains: {top_str}")

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


def _validate_labels(args, dataset: FlatImageDataset) -> None:
    if args.center_weight <= 0 and args.distinct_weight <= 0:
        raise ValueError("Center mode requires --center-weight > 0 or --distinct-weight > 0")
    if args.label_mode != "none" and (args.center_weight > 0 or args.distinct_weight > 0):
        if dataset.num_classes < 2:
            raise ValueError(
                "CRITICAL: fewer than 2 classes found. Center/Distinct losses cannot separate images. "
                "Check your manifest, regex, or label_mode."
            )


def _validate_domains(args, dataset: FlatImageDataset) -> None:
    if args.counterfactual_mode != "vae":
        return
    if dataset.num_domains < 2:
        raise ValueError(
            "Counterfactual VAE requires domain labels with at least 2 domains. "
            "Provide --domain-mode regex/parent with a matching --domain-regex."
        )


def _build_centers(args, dataset, device: torch.device):
    centers = None
    if args.center_weight > 0 or args.distinct_weight > 0:
        if dataset.num_classes <= 0:
            print("WARNING: No labels available; hash-center losses will be disabled.")
        else:
            if args.center_mode == "random":
                print(f"Hash centers: random (classes={dataset.num_classes})")
            else:
                max_centers = 2 * args.hash_dim
                num_centers = args.num_centers if args.num_centers > 0 else dataset.num_classes
                if num_centers > max_centers:
                    print(
                        f"WARNING: num_centers={num_centers} exceeds Hadamard limit "
                        f"({max_centers}); using {max_centers} with hash(label)%num_centers mapping."
                    )
                    num_centers = max_centers
                centers = build_hash_centers(
                    num_centers=num_centers,
                    hash_dim=args.hash_dim,
                    device=device,
                    seed=args.center_seed,
                )
                print(f"Hash centers: {num_centers} (classes={dataset.num_classes})")
    return centers


def _build_ghost_centers(args, device: torch.device):
    if args.distinct_weight <= 0 or args.extra_negatives <= 0:
        return None
    if args.center_mode != "random":
        print("WARNING: extra negatives ignored unless --center-mode=random")
        return None
    ghost_seed = args.seed + 1337
    ghost_centers = build_ghost_centers(
        num_centers=args.extra_negatives,
        hash_dim=args.hash_dim,
        device=device,
        seed=ghost_seed,
    )
    print(f"Ghost centers: {ghost_centers.size(0)}")
    return ghost_centers


def _build_counterfactual_vae(args, dataset: FlatImageDataset, device: torch.device) -> Optional[CounterfactualVAE]:
    if args.counterfactual_mode != "vae":
        return None
    if not args.counterfactual_weights:
        raise ValueError("counterfactual_mode=vae requires --counterfactual-weights")
    model = CounterfactualVAE(num_domains=dataset.num_domains)
    ckpt_path = Path(args.counterfactual_weights)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Counterfactual VAE checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        safetensors.torch.load_model(model, str(ckpt_path))
    else:
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _sample_target_domains(domain_ids: torch.Tensor, num_domains: int) -> torch.Tensor:
    if num_domains <= 1:
        return domain_ids
    rand = torch.randint(1, num_domains, size=domain_ids.shape, device=domain_ids.device)
    return (domain_ids + rand) % num_domains


def _generate_counterfactual(
    view1: torch.Tensor,
    view2: Optional[torch.Tensor],
    domain_ids: torch.Tensor,
    cf_model: Optional[CounterfactualVAE],
    args,
    device: torch.device,
    use_two_view: bool,
) -> Optional[torch.Tensor]:
    if args.counterfactual_mode == "none":
        return None
    if args.counterfactual_mode == "aug":
        if not use_two_view or view2 is None:
            raise ValueError("counterfactual_mode=aug requires two-view inputs")
        return view2
    if args.counterfactual_mode == "vae":
        if cf_model is None:
            raise ValueError("counterfactual_mode=vae requires a loaded VAE model")
        if (domain_ids < 0).any():
            raise ValueError("Counterfactual VAE requires valid domain labels")
        target_domains = _sample_target_domains(domain_ids, cf_model.num_domains)
        raw_view = _denormalize_batch(view1, device).clamp(0, 1)
        with torch.no_grad():
            cf_raw = cf_model.generate_counterfactual(raw_view, domain_ids, target_domains)
        cf_view = _normalize_batch(cf_raw, device)
        return cf_view
    raise ValueError(f"Unsupported counterfactual_mode: {args.counterfactual_mode}")


def _pgd_attack(
    model: ResNetHashNet,
    x: torch.Tensor,
    labels: torch.Tensor,
    centers: Optional[torch.Tensor],
    ghost_centers: Optional[torch.Tensor],
    args,
    device: torch.device,
) -> torch.Tensor:
    if args.pgd_steps <= 0 or args.adv_weight <= 0:
        return x
    eps = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    eps = (args.pgd_epsilon / eps).clamp_min(1e-6)
    alpha = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    alpha = (args.pgd_alpha / alpha).clamp_min(1e-6)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x_adv = x.detach()
    x_orig = x.detach()

    for _ in range(args.pgd_steps):
        x_adv.requires_grad_(True)
        with torch.amp.autocast("cuda", enabled=False):
            logits_adv = model(x_adv.float())
            center_loss, distinct_loss = hash_center_losses(
                logits_adv, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
            )
            q_loss = quantization_loss(torch.sigmoid(logits_adv)) if args.quant_weight > 0 else torch.zeros((), device=device)
            hash_loss = (
                args.center_weight * center_loss +
                args.distinct_weight * distinct_loss +
                args.quant_weight * q_loss
            )
        grad = torch.autograd.grad(hash_loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.max(torch.min(x_adv, upper), lower)
        x_adv = x_adv.detach()
    return x_adv


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _ttc_transforms(img: Image.Image):
    rng = random.Random(42)
    w, h = img.size
    crop_pct = 0.95
    crop_w, crop_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.BICUBIC)
    angle = rng.choice([-1.0, 1.0])
    brightness = rng.choice([0.95, 1.05])
    noise_sigma = 0.02
    noise_tensor = transforms.ToTensor()(img)
    noise = torch.randn_like(noise_tensor) * noise_sigma
    noisy = (noise_tensor + noise).clamp(0, 1)
    noisy_img = transforms.ToPILImage()(noisy)

    return [
        img,
        img.rotate(angle, resample=Image.BICUBIC),
        _jpeg_compress(img, 90),
        img.filter(ImageFilter.GaussianBlur(radius=0.5)),
        cropped,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(img).enhance(brightness),
        noisy_img,
    ]


def _run_ttc_check(model: ResNetHashNet, dataset: FlatImageDataset, device: torch.device, args) -> None:
    if not args.ttc_check or args.ttc_samples <= 0:
        return
    base_transform = transforms.Compose([
        transforms.Resize(args.image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    indices = list(range(min(args.ttc_samples, len(dataset.paths))))
    total_pairs = 0
    consistent_pairs = 0
    for idx in indices:
        img = Image.open(dataset.paths[idx]).convert("RGB")
        views = _ttc_transforms(img)
        hashes = []
        for view in views[: args.ttc_views]:
            tensor = base_transform(view).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)
                hashes.append((probs > 0.5).float())
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                total_pairs += 1
                dist = (hashes[i] != hashes[j]).float().mean().item() * args.hash_dim
                if dist <= args.ttc_threshold:
                    consistent_pairs += 1
    if total_pairs > 0:
        ratio = consistent_pairs / total_pairs
        print(f"TTC consistency: {ratio:.3f} ({consistent_pairs}/{total_pairs} pairs <= {args.ttc_threshold})")


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _configure_backends(args, device)

    use_two_view = not args.single_view
    mode_str = "two-view" if use_two_view else "single-view"

    loader, dataset = _create_dataloader(args, use_two_view)
    _validate_labels(args, dataset)
    _validate_domains(args, dataset)

    if args.counterfactual_mode != "none" and args.single_view:
        raise ValueError("counterfactual_mode requires two-view inputs (disable --single-view)")
    if args.counterfactual_mode != "vae":
        print("WARNING: resnet.tex specifies counterfactual VAE; using approximate mode.")

    model = build_backbone(args.backbone, args.hash_dim, args.proj_dim, args.pretrained).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    cf_model = _build_counterfactual_vae(args, dataset, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = max(args.epochs * len(loader), 1)
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))
    print(f"Scheduler: {total_steps:,} total steps, {warmup_steps:,} warmup")
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            total_iters=warmup_steps,
        )
        cosine_steps = max(total_steps - warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print("\n" + "=" * 70)
    print("Hash Center Training - ResNet")
    print("=" * 70)
    print(f"Mode: {mode_str}")
    print(f"Backbone: {args.backbone} (pretrained={args.pretrained})")
    print(f"Hash: {args.hash_dim}d, Proj: {args.proj_dim}d")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(
        "Losses: "
        f"center(C)={args.center_weight} distinct(D)={args.distinct_weight} "
        f"quant(Q)={args.quant_weight}"
    )
    print(f"Center mode: {args.center_mode}")
    print(f"Extra negatives: {args.extra_negatives}")
    print(f"Counterfactual mode: {args.counterfactual_mode}")
    print(
        "Extras: "
        f"CF={args.cf_weight} DHD={args.dhd_weight} ADV={args.adv_weight} "
        f"PGD={args.pgd_steps} TTC={args.ttc_check}"
    )
    print("=" * 70 + "\n")

    centers = _build_centers(args, dataset, device)
    ghost_centers = _build_ghost_centers(args, device)

    update_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_total = 0.0
        epoch_align = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            if use_two_view:
                views, labels, domain_ids = batch
                view1, view2 = views
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
            else:
                view1, labels, domain_ids = batch
                view1 = view1.to(device, non_blocking=True)
                view2 = None

            labels = labels.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)
            if args.channels_last:
                view1 = view1.contiguous(memory_format=torch.channels_last)
                if view2 is not None:
                    view2 = view2.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits1, proj1 = model(view1, return_proj=True)
                if use_two_view:
                    logits2, proj2 = model(view2, return_proj=True)
                else:
                    logits2, proj2 = None, None

                if args.counterfactual_mode == "none":
                    if use_two_view and logits2 is not None:
                        c1, d1 = hash_center_losses(
                            logits1, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
                        )
                        c2, d2 = hash_center_losses(
                            logits2, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
                        )
                        center_loss = 0.5 * (c1 + c2)
                        distinct_loss = 0.5 * (d1 + d2)
                        if args.quant_weight > 0:
                            q_loss = 0.5 * (
                                quantization_loss(torch.sigmoid(logits1)) +
                                quantization_loss(torch.sigmoid(logits2))
                            )
                        else:
                            q_loss = torch.zeros((), device=device)
                    else:
                        center_loss, distinct_loss = hash_center_losses(
                            logits1, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
                        )
                        q_loss = quantization_loss(torch.sigmoid(logits1)) if args.quant_weight > 0 else torch.zeros((), device=device)
                    cf_loss = torch.zeros((), device=device)
                    dhd = torch.zeros((), device=device)
                else:
                    center_loss, distinct_loss = hash_center_losses(
                        logits1, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
                    )
                    q_loss = quantization_loss(torch.sigmoid(logits1)) if args.quant_weight > 0 else torch.zeros((), device=device)

                    cf_view = _generate_counterfactual(
                        view1, view2, domain_ids, cf_model, args, device, use_two_view,
                    )
                    cf_logits, cf_proj = model(cf_view, return_proj=True)
                    cf_loss = simclr_loss(proj1, cf_proj, args.cf_temperature) if args.cf_weight > 0 else torch.zeros((), device=device)
                    if args.dhd_weight > 0:
                        h1 = 2 * torch.sigmoid(logits1) - 1
                        h2 = 2 * torch.sigmoid(cf_logits) - 1
                        dhd = dhd_loss(h1, h2)
                    else:
                        dhd = torch.zeros((), device=device)

                base_hash = (
                    args.center_weight * center_loss -
                    args.distinct_weight * distinct_loss +
                    args.quant_weight * q_loss
                )

                adv_loss = torch.zeros((), device=device)
                if args.adv_weight > 0:
                    adv_view = _pgd_attack(
                        model, view1, labels, centers, ghost_centers, args, device,
                    )
                    adv_logits = model(adv_view)
                    adv_center, adv_distinct = hash_center_losses(
                        adv_logits, labels, centers, args.center_neg_k, args.center_mode, ghost_centers,
                    )
                    adv_q = quantization_loss(torch.sigmoid(adv_logits)) if args.quant_weight > 0 else torch.zeros((), device=device)
                    adv_hash = (
                        args.center_weight * adv_center -
                        args.distinct_weight * adv_distinct +
                        args.quant_weight * adv_q
                    )
                    adv_loss = adv_hash

                total_loss = (
                    base_hash +
                    args.cf_weight * cf_loss +
                    args.dhd_weight * dhd +
                    args.adv_weight * adv_loss
                )

            scaler.scale(total_loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            prev_step = getattr(optimizer, "_step_count", None)
            scaler.step(optimizer)
            scaler.update()
            if prev_step is None:
                scheduler.step()
            elif getattr(optimizer, "_step_count", 0) > prev_step:
                # Only advance the scheduler if the optimizer actually stepped.
                scheduler.step()

            update_step += 1
            epoch_total += total_loss.item()
            epoch_align += center_loss.item()

            if update_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"E{epoch+1} B{batch_idx} U{update_step} "
                    f"tot={total_loss.item():.4f} ctr={center_loss.item():.4f} "
                    f"dst={distinct_loss.item():.4f} q={q_loss.item():.4f} "
                    f"cf={cf_loss.item():.4f} dhd={dhd.item():.4f} adv={adv_loss.item():.4f} "
                    f"lr={lr:.2e}"
                )

            if args.checkpoint_dir and args.checkpoint_every > 0:
                if update_step % args.checkpoint_every == 0:
                    _save_checkpoint(model, args.checkpoint_dir, f"student_u{update_step}.safetensors")

        n_batches = len(loader)
        print(
            f"\n>>> Epoch {epoch+1}/{args.epochs} done. "
            f"Avg total={epoch_total/n_batches:.4f} align={epoch_align/n_batches:.4f}\n"
        )
        _run_ttc_check(model, dataset, device, args)
        if args.checkpoint_dir:
            _save_checkpoint(model, args.checkpoint_dir, f"student_e{epoch+1}.safetensors")

    if args.checkpoint_dir:
        _save_checkpoint(model, args.checkpoint_dir, "student_final.safetensors")

    print(f"\nTraining complete! Updates={update_step}")


def _save_checkpoint(model, checkpoint_dir: str, filename: str) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(checkpoint_dir) / filename
    safetensors.torch.save_model(model, str(save_path))
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hash Center Training (ResNet)",
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
    data_group.add_argument(
        "--domain-mode",
        choices=["none", "regex", "parent", "auto"],
        default="regex",
        help="Domain label source for counterfactual VAE conditioning",
    )
    data_group.add_argument(
        "--domain-regex",
        default=r"^(ffhq|openimages|openimg|mobileviews?)",
        help="Regex to extract domain from filename when --domain-mode=regex",
    )
    data_group.add_argument(
        "--domain-regex-group",
        type=int,
        default=1,
        help="Capture group index to use for --domain-regex",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    model_group.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    model_group.add_argument("--hash-dim", type=int, default=128)
    model_group.add_argument("--proj-dim", type=int, default=512)
    model_group.add_argument("--image-size", type=int, default=224)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=10)
    train_group.add_argument("--batch-size", type=int, default=256)
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--warmup-steps", type=int, default=2000)
    train_group.add_argument("--center-weight", type=float, default=1.0)
    train_group.add_argument("--distinct-weight", type=float, default=0.5)
    train_group.add_argument("--quant-weight", type=float, default=0.1)
    train_group.add_argument("--clip-grad", type=float, default=1.0)
    train_group.add_argument("--cf-weight", type=float, default=1.0)
    train_group.add_argument("--cf-temperature", type=float, default=0.07)
    train_group.add_argument("--dhd-weight", type=float, default=0.5)
    train_group.add_argument("--adv-weight", type=float, default=0.5)
    train_group.add_argument("--pgd-epsilon", type=float, default=8/255)
    train_group.add_argument("--pgd-alpha", type=float, default=2/255)
    train_group.add_argument("--pgd-steps", type=int, default=7)
    train_group.add_argument("--center-neg-k", type=int, default=0)
    train_group.add_argument("--num-centers", type=int, default=0)
    train_group.add_argument("--center-seed", type=int, default=42)
    train_group.add_argument("--center-mode", choices=["hadamard", "random"], default="hadamard")
    train_group.add_argument("--extra-negatives", type=int, default=0)
    train_group.add_argument("--single-view", action="store_true", help="Disable two-view augmentation")
    train_group.add_argument("--no-aug", action="store_true", help="Disable data augmentation (resize+crop only)")
    train_group.add_argument(
        "--counterfactual-mode",
        choices=["none", "aug", "vae"],
        default="vae",
        help="Counterfactual source (aug uses view2, vae uses CounterfactualVAE)",
    )
    train_group.add_argument(
        "--counterfactual-weights",
        default="",
        help="Path to CounterfactualVAE checkpoint (required for --counterfactual-mode=vae)",
    )

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
    infra_group.add_argument("--checkpoint-every", type=int, default=0)
    infra_group.add_argument("--ttc-check", action="store_true", help="Run TTC consistency check after each epoch")
    infra_group.add_argument("--ttc-samples", type=int, default=16)
    infra_group.add_argument("--ttc-views", type=int, default=8)
    infra_group.add_argument("--ttc-threshold", type=int, default=10)

    args = parser.parse_args()

    if not args.data and not args.data_list:
        parser.error("Must provide --data or --data-list (or both)")
    if args.counterfactual_mode == "none" and (args.cf_weight > 0 or args.dhd_weight > 0):
        print("WARNING: cf/dhd weights ignored when counterfactual_mode=none")

    train(args)


if __name__ == "__main__":
    main()
