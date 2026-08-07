import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from dataset import ColonyDataset 
from model import create_vit_model, freeze_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

