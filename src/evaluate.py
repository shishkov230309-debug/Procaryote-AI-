from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import ColonyDataset, build_class_mapping
from checkpoint import (
    load_checkpoint_metadata,
    validate_checkpoint_mapping,
    validate_checkpoint_model,
    validate_preprocessing,
)
from model import create_vit_model, load_vit_checkpoint
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "vit_best.pth"
PROCESSOR_DIR = ROOT_DIR / "checkpoints" / "vit_image_processor"
MODEL_NAME = "google/vit-base-patch16-224"
CSV_PATH = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"


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
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    genus_labels = loader.dataset.genus_list
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(genus_labels))))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=genus_labels,
        yticklabels=genus_labels,
    )
    plt.xlabel("Predicted genus")
    plt.ylabel("Actual genus")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    preds = torch.tensor(all_preds)
    labels = torch.tensor(all_labels)
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

    metadata = load_checkpoint_metadata(CHECKPOINT_PATH)
    class_mapping = build_class_mapping(CSV_PATH)
    validate_checkpoint_mapping(metadata, class_mapping, CSV_PATH)
    validate_checkpoint_model(metadata, MODEL_NAME)

    if PROCESSOR_DIR.exists():
        image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        image_processor = AutoImageProcessor.from_pretrained(metadata["model_name"])
    validate_preprocessing(image_processor, metadata)

    test_dataset = ColonyDataset(
        csv_path=str(CSV_PATH),
        split="test",
        image_processor=image_processor,
        class_mapping=class_mapping,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = create_vit_model(
        num_classes=metadata["num_classes"], model_name=metadata["model_name"]
    )
    model.to(device)

    load_vit_checkpoint(model, CHECKPOINT_PATH, device, expected_metadata=metadata)

    precision, precision_per_class = compute_macro_precision(model, test_loader, device)
    
    print(f"Test precision over {test_dataset.num_classes} species: {precision:.4f}")
    for idx, species in enumerate(test_dataset.genus_list):
        print(f"{species}: {precision_per_class[idx].item():.4f}")


if __name__ == "__main__":
    main()
