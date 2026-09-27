import torch
from transformers import ViTForImageClassification

def create_vit_model(num_classes : int,model_name : str = "google/vit-base-patch16-224"):
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True,)
    return model

def load_vit_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is incompatible with the configured "
            "Hugging Face ViT implementation. Retrain the model with this environment."
        ) from exc
    return model

def freeze_backbone(model, unfreeze_layers : int = 0):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    if unfreeze_layers > 0:
        encoder_blocks = model.vit.encoder.layer
        for block in encoder_blocks[-unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True
    return model
def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params : {trainable:,}/{total:,}({100 * trainable / total:.2f}%)')
if __name__ == "__main__":
    model = create_vit_model(num_classes=19)
    model = freeze_backbone(model, unfreeze_layers=2)
    print_trainable_params(model)