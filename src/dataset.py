import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent


train_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ]
)


def build_class_mapping(csv_path):
    df = pd.read_csv(csv_path)
    class_names = sorted(df["genus"].dropna().unique().tolist())
    return {
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }


def validate_class_mapping(df, class_mapping):
    expected = build_class_mapping_from_values(df["genus"])
    if expected != class_mapping:
        raise ValueError(
            "Dataset class mapping does not match the checkpoint mapping."
        )


def build_class_mapping_from_values(values):
    class_names = sorted(values.dropna().unique().tolist())
    return {
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }

class ColonyDataset(Dataset):
    def __init__(self, csv_path, split, image_processor=None, transform=None, class_mapping=None):
        self.df = pd.read_csv(csv_path)
        self.class_mapping = class_mapping or build_class_mapping_from_values(self.df["genus"])
        if class_mapping is not None:
            validate_class_mapping(self.df, class_mapping)
        self.genus_list = self.class_mapping["class_names"]
        self.genus_to_idx = self.class_mapping["class_to_idx"]

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        
        self.image_processor = image_processor
        self.transform = transform

    @property
    def num_classes(self):
        return len(self.genus_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["filename"])
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.image_processor is not None:
            encoded = self.image_processor(images=image, return_tensors="pt")
            image = encoded["pixel_values"].squeeze(0)

        label_index = self.genus_to_idx[row["genus"]]
        return image, torch.tensor(label_index, dtype=torch.long)

    def __len__(self):
        return len(self.df)



if __name__ == "__main__":
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dataset = ColonyDataset(
        csv_path=ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv",
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )
    print("Dataset length:", len(dataset))
    image_tensor, label_index = dataset[0]
    print("Image shape:", image_tensor.shape)
    print("Label index:", label_index)
    print("Num classes:", dataset.num_classes)


