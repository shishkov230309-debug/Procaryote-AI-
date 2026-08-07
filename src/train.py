import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import ColonyDataset , train_transform, val_test_transform
from model import create_vit_model, freeze_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ColonyDataset(
    csv_path="data/train.csv",
    images_dir="data/images",
    split="train",
    transform=train_transform,
    split = "train"

)
val_dataset = ColonyDataset(
    csv_path="data/train.csv",
    images_dir="data/images",
    split="val",
    transform=val_test_transform,
    split   = "val"
)