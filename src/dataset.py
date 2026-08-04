import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ]
)


val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
class ColonyDataset(Dataset):
    def __init__(self, csv_path, images_dir, split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.genus_list = sorted(self.df["genus"].unique())
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}
        self.images_dir = images_dir
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_index = self.genus_to_idx[row["genus"]]
        return image, label_index

    def __len__(self):
        return len(self.df)
    
if __name__ == "__main__":
    dataset = ColonyDataset(
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw",
        split="train",
        transform=train_transform,
    )
    print("Dataset length:", len(dataset))
    image_tensor, label_index = dataset[0]
    print("Image shape:", image_tensor.shape)
    print("Label index:", label_index)


