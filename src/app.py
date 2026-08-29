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


@torch.inference_mode()
def predict_species(image):
    if image is None:
        return "Please upload an image first."

    image = Image.open(image).convert("RGB")
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
    inputs=gr.Image(type="pil", label="Upload colony image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Procaryote AI Colony Classifier",
    description="Upload a colony image and the model will predict the most likely bacterial genus.",
)


if __name__ == "__main__":
    demo.launch()
