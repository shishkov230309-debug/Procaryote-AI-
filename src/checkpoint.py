from enum import Enum

import torch
import transformers


METADATA_VERSION = 1


def _to_safe_metadata(value):
    if isinstance(value, Enum):
        return _to_safe_metadata(value.value)
    if isinstance(value, dict):
        return {str(key): _to_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_safe_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def build_checkpoint_metadata(
    class_mapping,
    model_name,
    image_processor,
    training_csv_path,
):
    return {
        "metadata_version": METADATA_VERSION,
        "class_names": list(class_mapping["class_names"]),
        "class_to_idx": dict(class_mapping["class_to_idx"]),
        "num_classes": int(class_mapping["num_classes"]),
        "model_name": model_name,
        "transformers_version": transformers.__version__,
        "preprocessing_config": _to_safe_metadata(image_processor.to_dict()),
        "training_csv_path": str(training_csv_path),
    }


def save_checkpoint(path, model, metadata):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint_metadata(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint at {path} has no class mapping metadata. "
            "It cannot be evaluated safely; retrain the model."
        )
    metadata = checkpoint["metadata"]
    required = {
        "class_names",
        "class_to_idx",
        "num_classes",
        "model_name",
        "preprocessing_config",
        "training_csv_path",
    }
    missing = required - metadata.keys()
    if missing:
        raise RuntimeError(
            f"Checkpoint at {path} is missing metadata fields: {sorted(missing)}."
        )
    return metadata


def validate_checkpoint_mapping(metadata, class_mapping, csv_path):
    checkpoint_names = list(metadata["class_names"])
    checkpoint_mapping = dict(metadata["class_to_idx"])
    if (
        checkpoint_names != list(class_mapping["class_names"])
        or checkpoint_mapping != dict(class_mapping["class_to_idx"])
        or int(metadata["num_classes"]) != int(class_mapping["num_classes"])
    ):
        raise RuntimeError(
            f"Class mapping mismatch for {csv_path}: the dataset labels do not "
            "match the checkpoint classifier indices."
        )


def validate_checkpoint_model(metadata, model_name):
    if metadata["model_name"] != model_name:
        raise RuntimeError(
            f"Model mismatch: checkpoint was trained with {metadata['model_name']!r}, "
            f"but the current configuration requests {model_name!r}."
        )


def validate_preprocessing(image_processor, metadata):
    if _to_safe_metadata(image_processor.to_dict()) != metadata["preprocessing_config"]:
        raise RuntimeError(
            "Preprocessing configuration mismatch: the processor does not match "
            "the configuration saved with the checkpoint."
        )