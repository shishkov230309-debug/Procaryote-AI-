import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import ColonyDataset, train_augment
from model import create_vit_model, freeze_backbone

MODEL_NAME = "google/vit-base-patch16-224"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_dataset = ColonyDataset(
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw/nature_colony/images/images",
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )
    val_dataset = ColonyDataset(
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw/nature_colony/images/images",
        split="val",
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

    model = create_vit_model(num_classes=19, model_name=MODEL_NAME)
    model = freeze_backbone(model, unfreeze_layers=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    num_epochs = 5
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(pixel_values=images)
                    val_loss = criterion(outputs.logits, labels)

                val_running_loss += val_loss.item() * images.size(0)
                predictions = outputs.logits.argmax(dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_accuracy = val_correct / val_total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "checkpoints/vit_best.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} - train loss: {epoch_loss:.4f} - "
            f"val loss: {val_epoch_loss:.4f} - val acc: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
