from pathlib import Path

import pandas as pd
import torch
import gradio as gr
from PIL import Image
from transformers import AutoImageProcessor

from model import create_vit_model

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "vit_best.pth"
PROCESSOR_DIR = ROOT_DIR / "checkpoints" / "vit_image_processor"
LABELS_CSV = ROOT_DIR / "data" / "annotated" / "nature_colony_labels_split.csv"
MODEL_NAME = "google/vit-base-patch16-224"


def _load_class_names():
    df = pd.read_csv(LABELS_CSV)
    return sorted(df["genus"].dropna().unique().tolist())


CLASS_NAMES = _load_class_names()


def _load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit_model(num_classes=len(CLASS_NAMES), model_name=MODEL_NAME)
    model.to(device)

    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"No trained model checkpoint found at {CHECKPOINT_PATH}")

    if PROCESSOR_DIR.exists():
        processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

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
    pred_idx = int(probs.argmax(dim=-1).item())
    pred_label = CLASS_NAMES[pred_idx].replace("_", " ").title()
    confidence = float(probs[0, pred_idx].item() * 100)

    return f"Prediction: {pred_label} ({confidence:.1f}% confidence)"


demo = gr.Interface(
    fn=predict_species,
    inputs=gr.Image(type="filepath", label="Upload colony image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Procaryote AI Colony Classifier",
    description="Upload any local image file (.jpg, .png, .jpeg, .bmp, .webp, etc.).",
)


if __name__ == "__main__":
    demo.launch(debug=True)
