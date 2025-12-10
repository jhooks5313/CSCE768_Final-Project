# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script handles the dataset + transforms (augmentations)
acts as a preprocessing script
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#img transforms
img_size = (256,256)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(img_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class DefDataset(Dataset):
    def __init__(self, dat_dir, csv_label=None, transform=transform):
        self.dat_dir = dat_dir
        self.transform = transform
        if csv_label is not None:
            pr = pd.read_csv(csv_label)
            self.dat_path = pr["filename"].tolist()
            self.label = pr["label"].astype(int).tolist()
        else:
            self.dat_path = os.listdir(dat_dir)
            self.label = None

    def __len__(self):
        return len(self.dat_path)

    def __getitem__(self, idx):
        fname = self.dat_path[idx]
        img_path = os.path.join(self.dat_dir, fname)
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        if self.label is not None:
            return img, torch.tensor(self.label[idx], dtype=torch.long)
        else:
            return img
