from transformers import AutoModel


def load_teacher(model_id="facebook/dinov3-vit-large-14"):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model
