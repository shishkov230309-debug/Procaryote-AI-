import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class ColonyDataset(Dataset):
    def _init__(self, csv_path, images_dir,split, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.genus_list = sorted(self.df["genus"].unique())
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}
        self.images_dir = images_dir
        self.transform = transform
        
        
