"""
DINOv3 Teacher Model Loader

Uses Meta's DINOv3 as a frozen teacher for distillation.
The ViT-L/16 variant provides high-quality dense features for semantic understanding.

NOTE: Teacher output dim is 1024 for ViT-L, which gets projected to hash_dim
via a frozen orthogonal projection in training.
"""
from typing import Optional

import torch
from transformers import AutoModel, AutoImageProcessor


# Available DINOv3 models (true IDs):
# - facebook/dinov3-vits16-pretrain-lvd1689m      (21M params, dim=384)
# - facebook/dinov3-vitb16-pretrain-lvd1689m      (86M params, dim=768)
# - facebook/dinov3-vitl16-pretrain-lvd1689m      (300M params, dim=1024)  <- recommended
# - facebook/dinov3-vith16plus-pretrain-lvd1689m  (840M params, dim=1280)
# - facebook/dinov3-vit7b16-pretrain-lvd1689m     (6.7B params, dim=1536)

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

MODEL_DIMS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 1280,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 1536,
}


def load_teacher(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Load frozen DINOv3 teacher model for feature distillation."""
    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(model_id, **kwargs)
    if device:
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_processor(model_id: str = DEFAULT_MODEL_ID):
    """Load the image processor for DINOv3."""
    return AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)


def get_output_dim(model_id: str = DEFAULT_MODEL_ID) -> int:
    """Get the output dimension for a given model."""
    return MODEL_DIMS.get(model_id, 1024)
