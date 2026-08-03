import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    
        
