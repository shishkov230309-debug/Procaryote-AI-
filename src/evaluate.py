"""Strict evaluation of the checkpoint on the training test split."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from checkpoint import (
    load_checkpoint_metadata,
    validate_checkpoint_mapping,
    validate_checkpoint_model,
    validate_preprocessing,
)
from dataset import ColonyDataset, build_class_mapping
from model import create_vit_model, load_vit_checkpoint
from thresholding import threshold_tradeoff

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "vit_best.pth"
PROCESSOR_DIR = ROOT_DIR / "checkpoints" / "vit_image_processor"
MODEL_NAME = "google/vit-base-patch16-224"
CSV_PATH = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"
BATCH_SIZE = 8


def _resolve_training_csv(metadata):
    configured_path = Path(metadata["training_csv_path"])
    if not configured_path.is_absolute():
        configured_path = ROOT_DIR / configured_path
    configured_path = configured_path.resolve()
    expected_path = CSV_PATH.resolve()
    if configured_path != expected_path:
        raise RuntimeError(
            "Checkpoint was trained with a different dataset split file: "
            f"{configured_path}; expected {expected_path}."
        )
    return expected_path


def _validate_model_architecture(model, metadata):
    expected_config = metadata.get("model_config")
    if expected_config is None:
        return
    if model.config.to_dict() != expected_config:
        raise RuntimeError(
            "Checkpoint model configuration does not match the configured "
            "architecture. Evaluation stopped before loading weights."
        )


def collect_predictions(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(pixel_values=images).logits
            batch_probabilities = torch.softmax(logits, dim=-1)
            labels.extend(batch_labels.tolist())
            predictions.extend(batch_probabilities.argmax(dim=-1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return labels, predictions, probabilities


def compute_metrics(labels, predictions, num_classes):
    class_indices = list(range(num_classes))
    per_class_precision, per_class_recall, _, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=class_indices,
            average=None,
            zero_division=0,
        )
    )
    macro_precision = float(
        precision_score(
            labels,
            predictions,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    weighted_precision = float(
        precision_score(
            labels,
            predictions,
            labels=class_indices,
            average="weighted",
            zero_division=0,
        )
    )
    macro_recall = float(
        recall_score(
            labels,
            predictions,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    prediction_counts = np.bincount(predictions, minlength=num_classes)
    return {
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "per_class_precision": [float(value) for value in per_class_precision],
        "per_class_recall": [float(value) for value in per_class_recall],
        "support": [int(value) for value in support],
        "prediction_counts": [int(value) for value in prediction_counts],
    }


def write_confusion_matrix(labels, predictions, class_names):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(class_names))),
    )
    figure = plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Test confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(ROOT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(figure)
    return matrix


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_checkpoint_metadata(CHECKPOINT_PATH)
    csv_path = _resolve_training_csv(metadata)
    class_mapping = build_class_mapping(csv_path)
    validate_checkpoint_mapping(metadata, class_mapping, csv_path)
    validate_checkpoint_model(metadata, MODEL_NAME)

    if PROCESSOR_DIR.is_dir():
        image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        image_processor = AutoImageProcessor.from_pretrained(metadata["model_name"])
    validate_preprocessing(image_processor, metadata)

    test_dataset = ColonyDataset(
        csv_path=csv_path,
        split="test",
        image_processor=image_processor,
        class_mapping=class_mapping,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_vit_model(
        num_classes=metadata["num_classes"],
        model_name=metadata["model_name"],
    )
    _validate_model_architecture(model, metadata)
    model.to(device)
    load_vit_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
        expected_metadata=metadata,
    )

    labels, predictions, probabilities = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, predictions, test_dataset.num_classes)
    write_confusion_matrix(labels, predictions, test_dataset.genus_list)

    print(f"Test samples: {len(labels)}")
    print(
        "Macro precision: "
        f"{metrics['macro_precision']:.4f} "
        "(unweighted mean of per-class precision; zero_division=0)"
    )
    print(f"Weighted precision: {metrics['weighted_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print("Class metrics:")
    for class_name, precision, recall, support, prediction_count in zip(
        test_dataset.genus_list,
        metrics["per_class_precision"],
        metrics["per_class_recall"],
        metrics["support"],
        metrics["prediction_counts"],
    ):
        print(
            f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
            f"support={support}, predictions={prediction_count}"
        )

    thresholding = metadata.get("thresholding")
    if thresholding:
        threshold_report = threshold_tradeoff(
            labels,
            predictions,
            np.max(np.asarray(probabilities), axis=1),
            thresholding["selected"]["threshold"],
            test_dataset.num_classes,
        )
        print(
            "Thresholded test metrics using validation-derived raw predicted "
            f"probability threshold {threshold_report['threshold']:.4f}: "
            f"precision={threshold_report['macro_precision']:.4f}, "
            f"recall={threshold_report['macro_recall']:.4f}, "
            f"coverage={threshold_report['coverage']:.4f}, "
            f"rejected={threshold_report['rejected']}"
        )
    else:
        print("No validation-derived threshold stored; reporting raw argmax only.")


if __name__ == "__main__":
    main()
