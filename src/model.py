import torch
from pathlib import Path    
from torch import device, nn
from transformers import ViTForImageClassification
checkpoint_path = Path(__file__).parent / "vit_checkpoint.pth"

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
state = torch.load(checkpoint_path, map_location=device, weights_only=True)
EXPECTED_NUM_CLASSES = 19


def _get_encoder_layers(model):
    encoder = getattr(getattr(model, "vit", None), "encoder", None)
    layers = getattr(encoder, "layer", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise TypeError(
            "Unsupported ViT architecture: expected model.vit.encoder.layer "
            "to be a non-empty torch.nn.ModuleList."
        )
    return layers


def create_vit_model(num_classes: int, model_name: str = "google/vit-base-patch16-224"):
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"This classifier expects exactly {EXPECTED_NUM_CLASSES} outputs; "
            f"received {num_classes}."
        )

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    _get_encoder_layers(model)
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear) or classifier.out_features != num_classes:
        raise TypeError(
            "Unsupported ViT architecture: expected a classifier linear layer "
            f"with {num_classes} outputs."
        )
    return model

def load_vit_checkpoint(model, checkpoint_path, device, expected_metadata=None):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has no metadata-bearing state_dict. "
            "Retrain the model with the current project."
        )
    metadata = state.get("metadata")
    if metadata is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has no class mapping metadata. "
            "It cannot be evaluated safely; retrain the model."
        )
    if expected_metadata is not None and metadata != expected_metadata:
        raise RuntimeError("Loaded checkpoint metadata does not match the expected metadata.")
    try:
        model.load_state_dict(state["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is incompatible with the configured "
            "Hugging Face ViT implementation. Retrain the model with this environment."
        ) from exc
    return metadata

def freeze_backbone(model, unfreeze_layers : int = 0):
    encoder_blocks = _get_encoder_layers(model)
    if not isinstance(unfreeze_layers, int) or not 0 <= unfreeze_layers <= len(encoder_blocks):
        raise ValueError(
            f"unfreeze_layers must be an integer from 0 to {len(encoder_blocks)}; "
            f"received {unfreeze_layers}."
        )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    if unfreeze_layers:
        for block in encoder_blocks[-unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True
    return model


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,}/{total:,} ({percentage:.2f}%)")
    return trainable, total


if __name__ == "__main__":
    model = create_vit_model(num_classes=EXPECTED_NUM_CLASSES)
    model = freeze_backbone(model, unfreeze_layers=2)
    print_trainable_params(model)