import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import ColonyDataset, train_augment
from model import create_vit_model, freeze_backbone
from pathlib import Path

MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_DIR = Path("checkpoints")


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return None, None

    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_dataset = ColonyDataset(
        csv_path="data/annotated/merged_labels_split.csv",
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )
    val_dataset = ColonyDataset(
        csv_path="data/annotated/merged_labels_split.csv",
        split="val",
        image_processor=image_processor,
    )
    test_dataset = ColonyDataset(
        csv_path="data/annotated/merged_labels_split.csv",
        split="test",
        image_processor=image_processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    model = create_vit_model(num_classes=19, model_name=MODEL_NAME)
    model = freeze_backbone(model, unfreeze_layers=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    num_epochs = 5
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
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

        val_epoch_loss, val_accuracy = evaluate(model, val_loader, criterion, device, use_amp)

        if val_accuracy is None:
            print("Validation split is empty. Skipping validation metrics for this epoch.")
            continue

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), CHECKPOINT_DIR / "vit_best.pth")
            image_processor.save_pretrained(CHECKPOINT_DIR / "vit_image_processor")

        print(
            f"Epoch {epoch + 1}/{num_epochs} - train loss: {epoch_loss:.4f} - "
            f"val loss: {val_epoch_loss:.4f} - val acc: {val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, use_amp)
    if test_accuracy is None:
        print("Test split is empty. Skipping final test evaluation.")
    else:
        print(f"Test loss: {test_loss:.4f} - test acc: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
