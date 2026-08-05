import os

import pandas as pd
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]
)


val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]
)
class ColonyDataset(Dataset):
    def __init__(self, csv_path, images_dir, split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.genus_list = sorted(self.df["genus"].dropna().unique())
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.images_dir = images_dir
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

        label_index = self.genus_to_idx[row["genus"]]
        return image, torch.tensor(label_index, dtype=torch.long)

    def __len__(self):
        return len(self.df)


def configure_vit_classifier(model, num_classes):
    """Replace the default ViT classification head with a dataset-sized head."""
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return model

    raise ValueError("Unsupported ViT model structure for classifier replacement")
    
if __name__ == "__main__":
    dataset = ColonyDataset(
        csv_path="data/annotated/nature_colony_labels_split.csv",
        images_dir="data/raw/nature_colony/images/images",
        split="train",
        transform=train_transform,
    )
    print("Dataset length:", len(dataset))
    image_tensor, label_index = dataset[0]
    print("Image shape:", image_tensor.shape)
    print("Label index:", label_index)
    print("Num classes:", dataset.num_classes)


