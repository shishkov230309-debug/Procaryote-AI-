import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


train_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ]
)

class ColonyDataset(Dataset):
    def __init__(self, csv_path, images_dir, split, image_processor=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.genus_list = sorted(self.df["genus"].dropna().unique())
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.transform = transform

    @property
    def num_classes(self):
        return len(self.genus_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["filename"])
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
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw/nature_colony/images/images",
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )
    print("Dataset length:", len(dataset))
    image_tensor, label_index = dataset[0]
    print("Image shape:", image_tensor.shape)
    print("Label index:", label_index)
    print("Num classes:", dataset.num_classes)


