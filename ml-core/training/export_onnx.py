import argparse
import sys
from pathlib import Path

import torch
import safetensors.torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.recursive_student import RecursiveHasher


def main():
    parser = argparse.ArgumentParser(description="Export RecursiveHasher to ONNX.")
    parser.add_argument("--checkpoint", default="", help="Path to .pt checkpoint (optional).")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hash-dim", type=int, default=96)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    model = RecursiveHasher(state_dim=args.state_dim, hash_dim=args.hash_dim)
    if args.checkpoint:
        safetensors.torch.load_model(model, args.checkpoint)
    model.eval()

    dummy_img = torch.randn(1, 3, args.image_size, args.image_size)
    dummy_state = torch.zeros(1, args.state_dim)

    torch.onnx.export(
        model,
        (dummy_img, dummy_state),
        args.output,
        input_names=["image", "prev_state"],
        output_names=["next_state", "hash"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch"},
            "prev_state": {0: "batch"},
            "next_state": {0: "batch"},
            "hash": {0: "batch"},
        },
    )


if __name__ == "__main__":
    main()
