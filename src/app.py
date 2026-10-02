from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor

from checkpoint import (
    load_checkpoint_metadata,
    validate_checkpoint_mapping,
    validate_checkpoint_model,
    validate_preprocessing,
)
from dataset import build_class_mapping
from thresholding import apply_global_threshold

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "vit_best.pth"
PROCESSOR_DIR = ROOT_DIR / "checkpoints" / "vit_image_processor"
LABELS_CSV = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"
MODEL_NAME = "google/vit-base-patch16-224"
IGNORED_CONFIG_KEYS = {"id2label", "label2id", "_name_or_path", "transformers_version","dtype","torch_type"}

@dataclass
class ApplicationState:
    model: object
    image_processor: object
    device: torch.device
    class_names: list
    thresholding: dict | None


APPLICATION_STATE = None


def _validate_model_architecture(model, metadata):
    expected_config = metadata.get("model_config")
    if expected_config is None:
        return
    actual = model.config.to_dict()
    diffs = [
        k for k in set(actual) | set(expected_config) 
        if k not in IGNORED_CONFIG_KEYS and actual.get(k) != expected_config.get(k)
    ]
    if diffs:
        raise RuntimeError(
            "The checkpoint model configuration does not match the configured "
            "ViT architecture."
        )


def load_application():
    """Load and validate all inference resources before the app starts."""
    global APPLICATION_STATE
    try:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        if not PROCESSOR_DIR.is_dir():
            raise FileNotFoundError(
                f"Saved image processor not found: {PROCESSOR_DIR}. "
                "Retrain or restore the matching checkpoint package."
            )

        metadata = load_checkpoint_metadata(CHECKPOINT_PATH)
        class_mapping = build_class_mapping(LABELS_CSV)
        validate_checkpoint_mapping(metadata, class_mapping, LABELS_CSV)
        validate_checkpoint_model(metadata, MODEL_NAME)

        image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
        validate_preprocessing(image_processor, metadata)

        from model import create_vit_model, load_vit_checkpoint

        model = create_vit_model(
            num_classes=metadata["num_classes"],
            model_name=metadata["model_name"],
        )
        _validate_model_architecture(model, metadata)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        load_vit_checkpoint(
            model,
            CHECKPOINT_PATH,
            device,
            expected_metadata=metadata,
        )
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "Procaryote AI startup failed while validating the checkpoint package: "
            f"{exc}"
        ) from exc

    APPLICATION_STATE = ApplicationState(
        model=model,
        image_processor=image_processor,
        device=device,
        class_names=list(class_mapping["class_names"]),
        thresholding=metadata.get("thresholding"),
    )
    return APPLICATION_STATE


def _to_rgb_image(image_input):
    if image_input is None or image_input == "":
        raise ValueError("Please upload an image first.")

    if isinstance(image_input, str):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        with Image.open(image_path) as img:
            return img.convert("RGB")

    if hasattr(image_input, "convert"):
        return image_input.convert("RGB")

    if hasattr(image_input, "read"):
        image_obj = Image.open(image_input)
        return image_obj.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_input)}")


@torch.inference_mode()
def predict_species(image_input):
    if APPLICATION_STATE is None:
        return "Application is not initialized; checkpoint validation has not completed."

    try:
        image = _to_rgb_image(image_input)
    except Exception as exc:
        return f"Image error: {exc}"

    inputs = APPLICATION_STATE.image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(APPLICATION_STATE.device) for k, v in inputs.items()}

    outputs = APPLICATION_STATE.model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    threshold = (
        APPLICATION_STATE.thresholding["selected"]["threshold"]
        if APPLICATION_STATE.thresholding
        else 0.0
    )
    pred_indices, predicted_probabilities, uncertain = apply_global_threshold(
        probabilities,
        threshold,
    )
    pred_idx = int(pred_indices[0])
    predicted_probability = float(predicted_probabilities[0] * 100)

    if APPLICATION_STATE.thresholding and bool(uncertain[0]):
        return (
            f"Uncertain prediction: predicted probability {predicted_probability:.1f}% "
            f"is below the validation threshold of {threshold * 100:.1f}%."
        )

    pred_label = APPLICATION_STATE.class_names[pred_idx].replace("_", " ").title()
    return f"Prediction: {pred_label} (predicted probability {predicted_probability:.1f}%)"


demo = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="filepath", label="Upload colony image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Procaryote AI Colony Classifier",
    description="Upload any local image file (.jpg, .png, .jpeg, .bmp, .webp, etc.).",
)


if __name__ == "__main__":
    try:
        load_application()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    demo.launch(debug=True)
