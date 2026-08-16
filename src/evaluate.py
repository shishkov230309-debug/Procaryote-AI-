from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import ColonyDataset
from model import create_vit_model
from sklearn.metrics import confusion_matrix
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "vit_best.pth"
PROCESSOR_DIR = ROOT_DIR / "checkpoints" / "vit_image_processor"
MODEL_NAME = "google/vit-base-patch16-224"
CSV_PATH = ROOT_DIR / "data" / "annotated" / "nature_colony_labels_split.csv"
IMAGES_DIR = ROOT_DIR / "data" / "raw" / "nature_colony" / "images" / "images"


def compute_macro_precision(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())
    cm = confusion_matrix(
            all_preds, all_labels)
    cm_output = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(ROOT_DIR / "confusion_matrix.png")
    plt.close()
    preds = torch.stack(all_preds)
    labels = torch.stack(all_labels)
    num_classes = int(model.config.num_labels)

    true_positives = torch.zeros(num_classes, dtype=torch.float32)
    false_positives = torch.zeros(num_classes, dtype=torch.float32)

    for pred, label in zip(preds.tolist(), labels.tolist()):
        if pred == label:
            true_positives[label] += 1
        else:
            false_positives[pred] += 1

    precision_per_class = torch.zeros(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        denom = true_positives[idx] + false_positives[idx]
        if denom > 0:
            precision_per_class[idx] = true_positives[idx] / denom

    macro_precision = precision_per_class.mean().item()
    return macro_precision, precision_per_class


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No trained checkpoint found at: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit_model(num_classes=19, model_name=MODEL_NAME)
    model.to(device)

    if PROCESSOR_DIR.exists():
        image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    test_dataset = ColonyDataset(
        csv_path=str(CSV_PATH),
        images_dir=str(IMAGES_DIR),
        split="test",
        image_processor=image_processor,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    precision, precision_per_class = compute_macro_precision(model, test_loader, device)
    
    print(f"Test precision over {test_dataset.num_classes} species: {precision:.4f}")
    for idx, species in enumerate(test_dataset.genus_list):
        print(f"{species}: {precision_per_class[idx].item():.4f}")


if __name__ == "__main__":
    main()
