import argparse
import os
import platform
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import (
    ColonyDataset,
    build_class_mapping,
    conservative_train_augment,
    train_augment,
)
from checkpoint import build_checkpoint_metadata, save_checkpoint
from model import create_vit_model, freeze_backbone, load_vit_checkpoint
from thresholding import select_global_threshold, threshold_tradeoff

MODEL_NAME = "google/vit-base-patch16-224"
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
TRAINING_CSV_PATH = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"
RANDOM_SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
UNFREEZE_LAYERS = 4


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    augmentation: str
    class_weighted: bool
    head_learning_rate: float
    backbone_learning_rate: float
    max_epochs: int
    patience: int
    crop_boxes: bool = False


EXPERIMENTS = {
    "baseline": ExperimentConfig(
        name="baseline",
        augmentation="current",
        class_weighted=False,
        head_learning_rate=LEARNING_RATE,
        backbone_learning_rate=LEARNING_RATE,
        max_epochs=30,
        patience=5,
    ),
    "weighted": ExperimentConfig(
        name="weighted",
        augmentation="current",
        class_weighted=True,
        head_learning_rate=LEARNING_RATE,
        backbone_learning_rate=LEARNING_RATE,
        max_epochs=30,
        patience=5,
    ),
    "finetune": ExperimentConfig(
        name="finetune",
        augmentation="current",
        class_weighted=False,
        head_learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        max_epochs=30,
        patience=5,
    ),
    "conservative": ExperimentConfig(
        name="conservative",
        augmentation="conservative",
        class_weighted=False,
        head_learning_rate=LEARNING_RATE,
        backbone_learning_rate=LEARNING_RATE,
        max_epochs=30,
        patience=5,
    ),
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validation_metrics(labels, predictions, num_classes):
    class_indices = list(range(num_classes))
    return {
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                labels,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class_precision": [
            float(value)
            for value in precision_score(
                labels,
                predictions,
                labels=class_indices,
                average=None,
                zero_division=0,
            )
        ],
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total += labels.size(0)

    if total == 0:
        return None

    metrics = _validation_metrics(
        all_labels,
        all_predictions,
        num_classes=int(model.config.num_labels),
    )
    metrics["loss"] = running_loss / total
    return metrics


def collect_predictions(model, loader, device, use_amp):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(pixel_values=images).logits
            probabilities = torch.softmax(logits, dim=-1)
            all_labels.extend(labels.tolist())
            all_predictions.extend(probabilities.argmax(dim=-1).cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())
    return all_labels, all_predictions, all_probabilities


def _build_optimizer(model, experiment):
    head_parameters = list(model.classifier.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": experiment.head_learning_rate,
            },
            {
                "params": backbone_parameters,
                "lr": experiment.backbone_learning_rate,
            },
        ]
    )


def _build_class_weights(dataset, class_mapping, device):
    counts = dataset.df["genus"].value_counts()
    weights = [
        len(dataset.df) / (len(class_mapping["class_names"]) * counts[name])
        for name in class_mapping["class_names"]
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _checkpoint_path(experiment_name):
    if experiment_name == "baseline":
        return CHECKPOINT_DIR / "vit_best.pth"
    return CHECKPOINT_DIR / f"vit_{experiment_name}_best.pth"


def _promote_checkpoint(source_path, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        shutil.copyfile(source_path, temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_experiment(experiment):
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    augmentation = (
        train_augment
        if experiment.augmentation == "current"
        else conservative_train_augment
    )

    class_mapping = build_class_mapping(TRAINING_CSV_PATH)
    train_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="train",
        image_processor=image_processor,
        transform=augmentation,
        class_mapping=class_mapping,
        crop_boxes=experiment.crop_boxes,
    )
    val_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="val",
        image_processor=image_processor,
        class_mapping=class_mapping,
        crop_boxes=experiment.crop_boxes,
    )
    test_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="test",
        image_processor=image_processor,
        class_mapping=class_mapping,
        crop_boxes=experiment.crop_boxes,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_vit_model(num_classes=train_dataset.num_classes, model_name=MODEL_NAME)
    model = freeze_backbone(model, unfreeze_layers=UNFREEZE_LAYERS)
    model = model.to(device)

    class_weights = None
    if experiment.class_weighted:
        class_weights = _build_class_weights(train_dataset, class_mapping, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _build_optimizer(model, experiment)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    training_config = {
        "experiment": experiment.name,
        "augmentation": experiment.augmentation,
        "class_weighted_loss": experiment.class_weighted,
        "class_weights": class_weights.detach().cpu().tolist()
        if class_weights is not None
        else None,
        "batch_size": BATCH_SIZE,
        "head_learning_rate": experiment.head_learning_rate,
        "backbone_learning_rate": experiment.backbone_learning_rate,
        "max_epochs": experiment.max_epochs,
        "early_stopping_patience": experiment.patience,
        "unfreeze_layers": UNFREEZE_LAYERS,
        "selection_rule": "maximize validation macro precision; break ties with validation loss",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
    }
    checkpoint_metadata = build_checkpoint_metadata(
        class_mapping,
        MODEL_NAME,
        image_processor,
        TRAINING_CSV_PATH,
        model_config=model.config.to_dict(),
        training_config=training_config,
        random_seed=RANDOM_SEED,
    )

    best_selection = None
    epochs_without_improvement = 0
    checkpoint_path = _checkpoint_path(experiment.name)

    for epoch in range(experiment.max_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        if val_metrics is None:
            print("Validation split is empty. Skipping validation metrics for this epoch.")
            continue

        selection_key = (
            val_metrics["macro_precision"],
            -val_metrics["loss"],
        )
        if best_selection is None or selection_key > best_selection["key"]:
            best_selection = {"key": selection_key, "epoch": epoch + 1}
            epochs_without_improvement = 0
            selected_metadata = dict(checkpoint_metadata)
            selected_metadata["selected_validation_metrics"] = val_metrics
            selected_metadata["selected_epoch"] = epoch + 1
            save_checkpoint(checkpoint_path, model, selected_metadata)
            processor_directory = (
                CHECKPOINT_DIR / "vit_image_processor"
                if experiment.name == "baseline"
                else CHECKPOINT_DIR / f"vit_{experiment.name}_image_processor"
            )
            image_processor.save_pretrained(processor_directory)
        else:
            epochs_without_improvement += 1

        print(
            f"[{experiment.name}] Epoch {epoch + 1}/{experiment.max_epochs} - "
            f"train loss: {epoch_loss:.4f} - "
            f"val loss: {val_metrics['loss']:.4f} - "
            f"macro precision: {val_metrics['macro_precision']:.4f} - "
            f"weighted precision: {val_metrics['weighted_precision']:.4f} - "
            f"macro recall: {val_metrics['macro_recall']:.4f} - "
            f"accuracy: {val_metrics['accuracy']:.4f}"
        )
        print(
            "Validation per-class precision: "
            + ", ".join(
                f"{class_name}={precision:.4f}"
                for class_name, precision in zip(
                    class_mapping["class_names"],
                    val_metrics["per_class_precision"],
                )
            )
        )
        if epochs_without_improvement >= experiment.patience:
            print(
                f"Early stopping after {experiment.patience} epochs without "
                "validation macro-precision improvement."
            )
            break

    if best_selection is None:
        raise RuntimeError("No valid validation metrics were produced; no checkpoint selected.")

    selected_metadata = load_vit_checkpoint(model, checkpoint_path, device)
    validation_labels, validation_predictions, validation_probabilities = (
        collect_predictions(model, val_loader, device, use_amp)
    )
    thresholding = select_global_threshold(
        validation_labels,
        validation_predictions,
        np.max(np.asarray(validation_probabilities), axis=1),
        train_dataset.num_classes,
    )
    selected_metadata["thresholding"] = thresholding
    save_checkpoint(checkpoint_path, model, selected_metadata)
    validation_threshold_report = thresholding["selected"]
    print(
        "Validation threshold: "
        f"{validation_threshold_report['threshold']:.4f}; "
        f"precision={validation_threshold_report['macro_precision']:.4f}; "
        f"recall={validation_threshold_report['macro_recall']:.4f}; "
        f"coverage={validation_threshold_report['coverage']:.4f}; "
        f"rejected={validation_threshold_report['rejected']}"
    )
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
    test_labels, test_predictions, test_probabilities = collect_predictions(
        model, test_loader, device, use_amp
    )
    test_threshold_report = None
    if test_labels:
        test_threshold_report = threshold_tradeoff(
            test_labels,
            test_predictions,
            np.max(np.asarray(test_probabilities), axis=1),
            validation_threshold_report["threshold"],
            train_dataset.num_classes,
        )
    if test_metrics is None:
        print("Test split is empty. Skipping final test evaluation.")
    else:
        print(
            f"Selected epoch: {selected_metadata['selected_epoch']} - "
            f"test loss: {test_metrics['loss']:.4f} - "
            f"test macro precision: {test_metrics['macro_precision']:.4f} - "
            f"test weighted precision: {test_metrics['weighted_precision']:.4f} - "
            f"test macro recall: {test_metrics['macro_recall']:.4f} - "
            f"test accuracy: {test_metrics['accuracy']:.4f}"
        )
        print(
            f"Test thresholded: macro precision={test_threshold_report['macro_precision']:.4f}, "
            f"macro recall={test_threshold_report['macro_recall']:.4f}, "
            f"coverage={test_threshold_report['coverage']:.4f}, "
            f"rejected={test_threshold_report['rejected']}"
        )
        for class_name, precision in zip(
            class_mapping["class_names"], test_metrics["per_class_precision"]
        ):
            print(f"Test precision - {class_name}: {precision:.4f}")

    return {
        "experiment": experiment.name,
        "checkpoint_path": str(checkpoint_path),
        "selected_epoch": selected_metadata["selected_epoch"],
        "selected_validation_metrics": selected_metadata["selected_validation_metrics"],
        "test_metrics": test_metrics,
        "thresholding": thresholding,
        "test_threshold_report": test_threshold_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a controlled precision experiment.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="baseline",
        help="Experiment configuration to run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all non-crop experiments with the same seed and split.",
    )
    args = parser.parse_args()
    names = sorted(EXPERIMENTS) if args.all else [args.experiment]
    results = [run_experiment(EXPERIMENTS[name]) for name in names]
    print("\nExperiment summary:")
    for result in results:
        metrics = result["test_metrics"]
        if metrics is None:
            print(f"{result['experiment']}: no test metrics")
            continue
        print(
            f"{result['experiment']}: selected_epoch={result['selected_epoch']}, "
            f"test_macro_precision={metrics['macro_precision']:.4f}, "
            f"test_weighted_precision={metrics['weighted_precision']:.4f}, "
            f"test_macro_recall={metrics['macro_recall']:.4f}, "
            f"test_accuracy={metrics['accuracy']:.4f}"
        )
    if len(results) > 1:
        preferred = max(
            results,
            key=lambda result: (
                result["selected_validation_metrics"]["macro_precision"],
                -result["selected_validation_metrics"]["loss"],
            ),
        )
        print(
            "Preferred experiment by validation macro precision: "
            f"{preferred['experiment']}"
        )
        canonical_path = _checkpoint_path("baseline")
        preferred_path = Path(preferred["checkpoint_path"])
        if preferred_path != canonical_path:
            _promote_checkpoint(preferred_path, canonical_path)
            print(f"Promoted preferred checkpoint to: {canonical_path}")
        weighted = next(
            (result for result in results if result["experiment"] == "weighted"),
            None,
        )
        baseline = next(
            (result for result in results if result["experiment"] == "baseline"),
            None,
        )
        if weighted is not None and baseline is not None:
            weighted_precision = weighted["selected_validation_metrics"][
                "macro_precision"
            ]
            baseline_precision = baseline["selected_validation_metrics"][
                "macro_precision"
            ]
            print(
                "Class-weighted loss retained as preferred: "
                f"{weighted_precision > baseline_precision}"
            )


if __name__ == "__main__":
    main()
