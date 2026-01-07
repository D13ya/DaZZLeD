"""
DINOv3 Teacher Model Loader

Uses Meta's DINOv3 (arXiv:2508.10104) as a frozen teacher for distillation.
The ViT-L/16 variant provides high-quality dense features for semantic understanding.
"""
from transformers import AutoModel, AutoImageProcessor


# Available DINOv3 models (August 2025):
# - facebook/dinov3-vits16-pretrain-lvd1689m      (21M params)
# - facebook/dinov3-vitb16-pretrain-lvd1689m      (86M params)
# - facebook/dinov3-vitl16-pretrain-lvd1689m      (300M params)  <- recommended
# - facebook/dinov3-vith16plus-pretrain-lvd1689m  (840M params)
# - facebook/dinov3-vit7b16-pretrain-lvd1689m     (6.7B params)

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_teacher(model_id: str = DEFAULT_MODEL_ID):
    """Load frozen DINOv3 teacher model for feature distillation."""
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_processor(model_id: str = DEFAULT_MODEL_ID):
    """Load the image processor for DINOv3."""
    return AutoImageProcessor.from_pretrained(model_id)
