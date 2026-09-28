from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from transformers import AutoImageProcessor

from model import create_vit_model, load_vit_checkpoint
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

CHECKPOINT_METADATA = load_checkpoint_metadata(CHECKPOINT_PATH)
CLASS_MAPPING = build_class_mapping(LABELS_CSV)
validate_checkpoint_mapping(CHECKPOINT_METADATA, CLASS_MAPPING, LABELS_CSV)
validate_checkpoint_model(CHECKPOINT_METADATA, MODEL_NAME)
CLASS_NAMES = CHECKPOINT_METADATA["class_names"]
THRESHOLDING = CHECKPOINT_METADATA.get("thresholding")


def _load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit_model(
        num_classes=CHECKPOINT_METADATA["num_classes"],
        model_name=CHECKPOINT_METADATA["model_name"],
    )
    model.to(device)

    if CHECKPOINT_PATH.exists():
        load_vit_checkpoint(
            model,
            CHECKPOINT_PATH,
            device,
            expected_metadata=CHECKPOINT_METADATA,
        )
    else:
        raise FileNotFoundError(f"No trained model checkpoint found at {CHECKPOINT_PATH}")

    if PROCESSOR_DIR.exists():
        processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        processor = AutoImageProcessor.from_pretrained(CHECKPOINT_METADATA["model_name"])
    validate_preprocessing(processor, CHECKPOINT_METADATA)

    model.eval()
    return model, processor, device


MODEL, IMAGE_PROCESSOR, DEVICE = _load_model_and_processor()


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
    try:
        image = _to_rgb_image(image_input)
    except Exception as exc:
        return f"Image error: {exc}"

    inputs = IMAGE_PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = MODEL(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    probability_array = probs.cpu().numpy()
    pred_indices, predicted_probabilities, uncertain = apply_global_threshold(
        probability_array,
        THRESHOLDING["selected"]["threshold"] if THRESHOLDING else 0.0,
    )
    pred_idx = int(pred_indices[0])
    predicted_probability = float(predicted_probabilities[0] * 100)
    if THRESHOLDING and bool(uncertain[0]):
        threshold = THRESHOLDING["selected"]["threshold"] * 100
        return (
            f"Uncertain prediction: predicted probability {predicted_probability:.1f}% "
            f"is below the validation threshold of {threshold:.1f}%."
        )

    pred_label = CLASS_NAMES[pred_idx].replace("_", " ").title()

    return f"Prediction: {pred_label} (predicted probability {predicted_probability:.1f}%)"


demo = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="filepath", label="Upload colony image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Procaryote AI Colony Classifier",
    description="Upload any local image file (.jpg, .png, .jpeg, .bmp, .webp, etc.).",
)


if __name__ == "__main__":
    demo.launch(debug=True)
