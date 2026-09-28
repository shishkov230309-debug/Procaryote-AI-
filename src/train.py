import platform
import random
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import ColonyDataset, build_class_mapping, train_augment
from checkpoint import build_checkpoint_metadata, save_checkpoint
from model import create_vit_model, freeze_backbone, load_vit_checkpoint

MODEL_NAME = "google/vit-base-patch16-224"
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
TRAINING_CSV_PATH = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"
RANDOM_SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
UNFREEZE_LAYERS = 4


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


def main():
    set_random_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    class_mapping = build_class_mapping(TRAINING_CSV_PATH)
    train_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="train",
        image_processor=image_processor,
        transform=train_augment,
        class_mapping=class_mapping,
    )
    val_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="val",
        image_processor=image_processor,
        class_mapping=class_mapping,
    )
    test_dataset = ColonyDataset(
        csv_path=TRAINING_CSV_PATH,
        split="test",
        image_processor=image_processor,
        class_mapping=class_mapping,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    training_config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
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
    checkpoint_path = CHECKPOINT_DIR / "vit_best.pth"

    for epoch in range(NUM_EPOCHS):
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
            selected_metadata = dict(checkpoint_metadata)
            selected_metadata["selected_validation_metrics"] = val_metrics
            selected_metadata["selected_epoch"] = epoch + 1
            save_checkpoint(checkpoint_path, model, selected_metadata)
            image_processor.save_pretrained(CHECKPOINT_DIR / "vit_image_processor")

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - train loss: {epoch_loss:.4f} - "
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

    if best_selection is None:
        raise RuntimeError("No valid validation metrics were produced; no checkpoint selected.")

    selected_metadata = load_vit_checkpoint(model, checkpoint_path, device)
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
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
        for class_name, precision in zip(
            class_mapping["class_names"], test_metrics["per_class_precision"]
        ):
            print(f"Test precision - {class_name}: {precision:.4f}")


if __name__ == "__main__":
    main()
