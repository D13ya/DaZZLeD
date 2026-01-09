"""
HashNet Inference with Test-Time Compute (TTC).

Runs deterministic test-time augmentations, computes hashes, and checks
consistency before accepting a hash.
"""

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import safetensors.torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from training.train_hashnet import ResNetHashNet, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def get_ttc_views(img: Image.Image) -> List[Image.Image]:
    w, h = img.size
    crop_pct = 0.95
    crop_w, crop_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.BICUBIC)

    noise_tensor = transforms.ToTensor()(img)
    noise = torch.randn_like(noise_tensor) * 0.02
    noisy = (noise_tensor + noise).clamp(0, 1)
    noisy_img = transforms.ToPILImage()(noisy)

    return [
        img,
        img.rotate(1.0, resample=Image.BICUBIC),
        img.rotate(-1.0, resample=Image.BICUBIC),
        _jpeg_compress(img, 90),
        img.filter(ImageFilter.GaussianBlur(radius=0.5)),
        cropped,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(img).enhance(1.05),
        noisy_img,
    ]


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(checkpoint: str, backbone: str, hash_dim: int, proj_dim: int, device: torch.device):
    model = ResNetHashNet(backbone, hash_dim, proj_dim, pretrained=False).to(device)
    ckpt_path = Path(checkpoint)
    if ckpt_path.suffix == ".safetensors":
        safetensors.torch.load_model(model, str(ckpt_path))
    else:
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def compute_ttc_hash(
    model,
    image_path: str,
    image_size: int,
    ttc_views: int,
    stability_threshold: float,
    hamming_threshold: Optional[int],
    device: torch.device,
):
    image = Image.open(image_path).convert("RGB")
    views = get_ttc_views(image)
    views = views[:ttc_views]
    transform = build_transform(image_size)
    batch = torch.stack([transform(view) for view in views]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
        binary = (probs > 0.5).float()

    mean_vector = binary.mean(dim=0)
    stability_per_bit = 2 * torch.abs(mean_vector - 0.5)
    stability = stability_per_bit.mean().item()
    final_hash = (mean_vector > 0.5).int()

    accepted = stability >= stability_threshold
    if hamming_threshold is not None:
        hamming = (binary != final_hash).float().mean(dim=1) * final_hash.numel()
        accepted = accepted and (hamming.max().item() <= hamming_threshold)
        return final_hash, stability, hamming.max().item(), accepted

    return final_hash, stability, None, accepted


def main():
    parser = argparse.ArgumentParser(description="HashNet TTC inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.safetensors or .pt)")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--hash-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ttc-views", type=int, default=8)
    parser.add_argument("--stability-threshold", type=float, default=0.9)
    parser.add_argument("--hamming-threshold", type=int, default=10)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.backbone, args.hash_dim, args.proj_dim, device)

    final_hash, stability, max_hamming, accepted = compute_ttc_hash(
        model=model,
        image_path=args.image,
        image_size=args.image_size,
        ttc_views=args.ttc_views,
        stability_threshold=args.stability_threshold,
        hamming_threshold=args.hamming_threshold,
        device=device,
    )

    hash_str = "".join(str(int(bit)) for bit in final_hash.tolist())
    status = "ACCEPT" if accepted else "REJECT"
    print(f"TTC: {status}")
    print(f"Stability: {stability:.4f}")
    if max_hamming is not None:
        print(f"Max Hamming distance: {max_hamming:.2f}")
    print(f"Hash ({args.hash_dim} bits): {hash_str}")


if __name__ == "__main__":
    main()
