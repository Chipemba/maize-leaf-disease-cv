# This script trains the baseline CNN model on the maize leaf disease dataset. 
# It uses the MaizeMultiLabelDataset class to load the training and validation data, applies the specified transforms and trains the model using binary cross-entropy loss with logits. 
# The model is saved if it achieves a lower validation loss than previously recorded.

import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.data.dataset import MaizeMultiLabelDataset, LABELS
from src.data.transforms import train_transform, val_test_transform
from src.models.baseline_cnn import BaselineCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = "data/splits/train.csv"
    val_csv = "data/splits/val.csv"
    image_root = "data/raw/static/leaf_images/"

    train_dataset = MaizeMultiLabelDataset(
        csv_path=train_csv,
        image_root=image_root,
        transform=train_transform
    )

    val_dataset = MaizeMultiLabelDataset(
        csv_path=val_csv,
        image_root=image_root,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    model = BaselineCNN(num_labels=8).to(device)

    train_df = pd.read_csv(train_csv)
    pos_counts = train_df[LABELS].sum().values
    neg_counts = len(train_df) - pos_counts

    pos_weight = torch.tensor(
        neg_counts / pos_counts,
        dtype=torch.float32
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(1, 11):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/baseline_cnn.pt")

if __name__ == "__main__":
    main()