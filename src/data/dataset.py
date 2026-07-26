# This creates a multi-label dataset class for the maize leaf disease dataset.
#  It reads the dataset.csv file and loads the images from the data/raw directory. 
# It also applies the specified transforms to the images.

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

LABELS = [
    "GLS",
    "NCLB",
    "PLS",
    "CR",
    "SR",
    "NoFoliarSymptoms",
    "Other",
    "UnidentifiedDisease"
]

class MaizeMultiLabelDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        relative_path = str(row["filePath"]).lstrip("/")
        img_path = self.image_root / relative_path

        image = Image.open(img_path).convert("RGB")

        labels = torch.tensor(
            row[LABELS].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels