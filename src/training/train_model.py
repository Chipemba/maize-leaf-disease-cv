"""
train_model.py

Reusable training script for Week 10 transfer learning and lightweight models.

This script trains:
- ResNet18
- EfficientNet-B0
- MobileNetV2

The project is multi-label, so the script uses:
- 8 output logits
- BCEWithLogitsLoss
- optional positive class weighting
"""

import argparse
import json
from pathlib import Path

import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.dataset import MaizeMultiLabelDataset, LABELS
from src.data.transforms import train_transform, val_test_transform
from src.models.model_factory import (
    build_model,
    count_total_parameters,
    count_trainable_parameters,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_pos_weight(train_csv, device):
    """
    Calculates positive class weights for BCEWithLogitsLoss.

    This helps with class imbalance by giving rare labels more importance.
    """

    train_df = pd.read_csv(train_csv)

    pos_counts = train_df[LABELS].sum().values
    neg_counts = len(train_df) - pos_counts

    # Avoid division by zero just in case any label has 0 positives in the split.
    pos_counts = [max(count, 1) for count in pos_counts]

    pos_weight = torch.tensor(
        neg_counts / pos_counts,
        dtype=torch.float32
    ).to(device)

    return pos_weight


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()

    return running_loss / len(loader)


def save_training_curve(history, save_path, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_training_summary(
    save_path,
    config,
    best_val_loss,
    total_params,
    trainable_params,
    best_model_path
):
    summary = f"""# Week 10 Training Summary: {config["model_name"]}

## Model
{config["model_name"]}

## Training Mode
{config.get("fine_tuning_mode", "not specified")}

## Pretrained
{config["pretrained"]}

## Freeze Backbone
{config["freeze_backbone"]}

## Number of Labels
{config["num_labels"]}

## Hyperparameters
- Image size: {config["image_size"]}
- Batch size: {config["batch_size"]}
- Epochs: {config["epochs"]}
- Learning rate: {config["learning_rate"]}
- Weight decay: {config["weight_decay"]}
- Optimizer: {config["optimizer"]}
- Loss: {config["loss"]}
- Threshold for later evaluation: {config["threshold"]}

## Parameters
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}

## Best Validation Loss
{best_val_loss:.6f}

## Checkpoint
{best_model_path}

## Notes
This model was trained using multi-label classification with BCEWithLogitsLoss.
The model outputs 8 raw logits, one for each maize disease or condition label.
Sigmoid activation is applied during evaluation, not during training.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model_name"]
    pretrained = config["pretrained"]
    num_labels = config["num_labels"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    freeze_backbone = config.get("freeze_backbone", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training model: {model_name}")
    print(f"Using device: {device}")
    print(f"Full fine-tuning: {not freeze_backbone}")

    train_csv = Path("data/splits/train.csv")
    val_csv = Path("data/splits/val.csv")
    image_root = Path("data/raw/static/leaf_images/")

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")

    if not val_csv.exists():
        raise FileNotFoundError(f"Missing validation split: {val_csv}")

    if not image_root.exists():
        raise FileNotFoundError(f"Missing image root: {image_root}")

    results_dir = Path(f"results/week10_{model_name}")
    models_dir = Path("models")

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model = build_model(
        model_name=model_name,
        num_labels=num_labels,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    ).to(device)

    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    pos_weight = calculate_pos_weight(train_csv, device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    best_model_path = models_dir / f"{model_name}.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model: {best_model_path}")

    history_path = results_dir / "training_history.json"
    curve_path = results_dir / "training_curve.png"
    config_path = results_dir / "config_used.json"
    summary_path = results_dir / "training_summary.md"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    save_training_curve(
        history=history,
        save_path=curve_path,
        model_name=model_name
    )

    save_training_summary(
        save_path=summary_path,
        config=config,
        best_val_loss=best_val_loss,
        total_params=total_params,
        trainable_params=trainable_params,
        best_model_path=best_model_path
    )

    print("\nTraining complete.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()