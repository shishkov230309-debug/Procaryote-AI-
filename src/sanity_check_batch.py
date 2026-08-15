import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from dataset import ColonyDataset, train_augment


MODEL_NAME = "google/vit-base-patch16-224"


def main():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    dataset = ColonyDataset(
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw/nature_colony/images/images",
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
