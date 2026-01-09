"""
Counterfactual VAE Training

Trains a conditional autoencoder to reconstruct images given a domain label.
The model can then generate counterfactuals by swapping the domain embedding.
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models.counterfactual_vae import CounterfactualVAE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DomainImageDataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        list_file=None,
        max_images=None,
        seed=0,
        cache_ram=False,
        domain_mode="regex",
        domain_regex=None,
        domain_regex_group=1,
    ):
        self.domain_mode = domain_mode
        self.domain_regex = re.compile(domain_regex) if domain_regex else None
        self.domain_regex_group = domain_regex_group
        paths, raw_domains = self._load_paths(root, list_file)

        if max_images and max_images > 0 and len(paths) > max_images:
            rng = random.Random(seed)
            combined = list(zip(paths, raw_domains))
            rng.shuffle(combined)
            combined = combined[:max_images]
            paths, raw_domains = zip(*combined)

        self.paths = list(paths)
        self.raw_domains = list(raw_domains)
        self.transform = transform
        self.cache_ram = cache_ram
        self.images = [None] * len(self.paths)
        self.domain_map = {}
        self.domain_ids = self._encode_domains(self.raw_domains)
        self.num_domains = len(self.domain_map)
        if self.cache_ram:
            self._cache_images()

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

    def _load_paths_from_list(self, list_file):
        paths = []
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
            domain = self._extract_domain(path, None)
            paths.append(path)
            domains.append(domain)
        return paths, domains

    def _load_paths_from_root(self, root):
        paths = []
        domains = []
        root_path = Path(root)
        if not root_path.exists():
            return paths, domains
        for ext in IMAGE_EXTS:
            for p in root_path.rglob(f"*{ext}"):
                domain = self._extract_domain(p, None)
                paths.append(p)
                domains.append(domain)
        return paths, domains

    def _load_paths(self, root, list_file):
        paths = []
        domains = []
        if list_file:
            paths, domains = self._load_paths_from_list(list_file)
        if not paths and root:
            paths, domains = self._load_paths_from_root(root)
        if not paths:
            raise ValueError("No images found. Provide --data or --data-list")
        return [str(p) for p in paths], domains

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
        return self.transform(img), self.domain_ids[idx]


def _create_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    dataset = DomainImageDataset(
        root=args.data if args.data else None,
        transform=transform,
        list_file=args.data_list if args.data_list else None,
        max_images=args.max_images if args.max_images > 0 else None,
        seed=args.seed,
        cache_ram=args.cache_ram,
        domain_mode=args.domain_mode,
        domain_regex=args.domain_regex if args.domain_regex else None,
        domain_regex_group=args.domain_regex_group,
    )

    total = len(dataset.raw_domains)
    known = [str(domain) for domain in dataset.raw_domains if domain is not None]
    unlabeled = total - len(known)
    unique = len(set(known))
    pct_unlabeled = (unlabeled / total * 100.0) if total > 0 else 0.0
    print(f"Domain stats: {unique} unique, {unlabeled}/{total} unlabeled ({pct_unlabeled:.1f}%).")
    if known:
        top = Counter(known).most_common(5)
        top_str = ", ".join(f"{domain}:{count}" for domain, count in top)
        print(f"Top domains: {top_str}")

    if dataset.num_domains < 2:
        raise ValueError("Counterfactual VAE requires at least 2 domains.")

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


def _save_checkpoint(model, checkpoint_dir: str, filename: str) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(checkpoint_dir) / filename
    safetensors.torch.save_model(model, str(save_path))
    print(f"Saved: {save_path}")


def _vae_loss(recon, x, mu, logvar, decoder_var, kl_weight):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / (2.0 * decoder_var)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kld
    return total, recon_loss, kld


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    loader, dataset = _create_dataloader(args)
    model = CounterfactualVAE(num_domains=dataset.num_domains).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print("\n" + "=" * 70)
    print("Counterfactual VAE Training")
    print("=" * 70)
    print(f"Domains: {dataset.num_domains}")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"LR: {args.lr}, WD: {args.weight_decay}")
    print("=" * 70 + "\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader, start=1):
            images, domain_ids = batch
            images = images.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                recon, mu, logvar = model(images, domain_ids)
                loss, recon_loss, kld = _vae_loss(
                    recon, images, mu, logvar, args.decoder_var, args.kl_weight
                )

            loss = loss / images.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                avg_recon = recon_loss.item() / images.size(0)
                avg_kld = kld.item() / images.size(0)
                print(
                    f"E{epoch+1} B{batch_idx} loss={loss.item():.4f} "
                    f"recon={avg_recon:.4f} kld={avg_kld:.4f}"
                )

        avg = epoch_loss / len(loader)
        print(f"\n>>> Epoch {epoch+1}/{args.epochs} done. Avg loss={avg:.4f}\n")
        if args.checkpoint_dir:
            _save_checkpoint(model, args.checkpoint_dir, f"cf_vae_e{epoch+1}.safetensors")

    if args.checkpoint_dir:
        _save_checkpoint(model, args.checkpoint_dir, "cf_vae_final.safetensors")


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual VAE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", default="", help="Image folder path")
    data_group.add_argument("--data-list", default="", help="Text file with image paths")
    data_group.add_argument("--max-images", type=int, default=0, help="Limit images (0=all)")
    data_group.add_argument("--cache-ram", action="store_true", help="Cache images in RAM")
    data_group.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    data_group.add_argument(
        "--domain-mode",
        choices=["none", "regex", "parent", "auto"],
        default="regex",
        help="Domain label source for conditioning",
    )
    data_group.add_argument(
        "--domain-regex",
        default=r"^(ffhq|openimages|openimg|mobileview)",
        help="Regex to extract domain from filename",
    )
    data_group.add_argument(
        "--domain-regex-group",
        type=int,
        default=1,
        help="Capture group index to use for --domain-regex",
    )

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=10)
    train_group.add_argument("--batch-size", type=int, default=128)
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--log-interval", type=int, default=50)
    train_group.add_argument("--checkpoint-dir", default="./checkpoints")
    train_group.add_argument("--kl-weight", type=float, default=1.0)
    train_group.add_argument("--decoder-var", type=float, default=1e-2)

    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--device", default="cuda")
    infra_group.add_argument("--amp", action="store_true")
    infra_group.add_argument("--pin-memory", action="store_true")
    infra_group.add_argument("--prefetch-factor", type=int, default=2)
    infra_group.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.data and not args.data_list:
        parser.error("Must provide --data or --data-list (or both)")

    train(args)


if __name__ == "__main__":
    main()
