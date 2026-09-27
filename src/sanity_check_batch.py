import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from pathlib import Path

from dataset import ColonyDataset, train_augment

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "google/vit-base-patch16-224"
CSV_PATH = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"


def main():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    dataset = ColonyDataset(
        csv_path=CSV_PATH,
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    pixel_values, labels = next(iter(loader))

    print("pixel_values shape:", tuple(pixel_values.shape))
    print("pixel_values dtype:", pixel_values.dtype)
    print("pixel_values min:", float(pixel_values.min()))
    print("pixel_values max:", float(pixel_values.max()))
    print("labels shape:", tuple(labels.shape))
    print("labels dtype:", labels.dtype)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
