"""
robustness.py

Week 12 robustness experiments for the maize leaf disease project.

This script evaluates trained models under controlled image distortions:
- brightness decrease
- brightness increase
- contrast decrease
- contrast increase
- mild blur
- strong blur
- left rotation
- right rotation

The task is multi-label classification, so the script uses sigmoid probabilities,
thresholding, and multi-label metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageFilter

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    classification_report,
)

from src.data.dataset import LABELS
from src.data.transforms import val_test_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import build_model


TEST_CSV = Path("data/splits/test.csv")
IMAGE_ROOT = Path("data/raw/static/leaf_images/")
RESULTS_DIR = Path("results/week12_robustness")

BATCH_SIZE = 32
NUM_WORKERS = 2
THRESHOLD = 0.5

MODELS = {
    "baseline_cnn": "models/baseline_cnn.pt",
    "efficientnet_b0": "models/efficientnet_b0.pt",
    "mobilenet_v2": "models/mobilenet_v2.pt",
    "resnet18": "models/resnet18.pt",
}

DISTORTIONS = [
    "clean",
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "contrast_high",
    "blur_mild",
    "blur_strong",
    "rotation_left",
    "rotation_right",
]


def apply_distortion(image, distortion_name):
    """
    Applies one controlled distortion to a PIL image.
    """

    if distortion_name == "clean":
        return image

    if distortion_name == "brightness_low":
        return ImageEnhance.Brightness(image).enhance(0.6)

    if distortion_name == "brightness_high":
        return ImageEnhance.Brightness(image).enhance(1.4)

    if distortion_name == "contrast_low":
        return ImageEnhance.Contrast(image).enhance(0.7)

    if distortion_name == "contrast_high":
        return ImageEnhance.Contrast(image).enhance(1.3)

    if distortion_name == "blur_mild":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))

    if distortion_name == "blur_strong":
        return image.filter(ImageFilter.GaussianBlur(radius=3.0))

    if distortion_name == "rotation_left":
        return image.rotate(-15)

    if distortion_name == "rotation_right":
        return image.rotate(15)

    raise ValueError(f"Unknown distortion: {distortion_name}")


class RobustnessDataset(Dataset):
    """
    Loads test images and applies one robustness distortion.
    """

    def __init__(self, csv_path, image_root, distortion_name, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_root = Path(image_root)
        self.distortion_name = distortion_name
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        relative_path = str(row["filePath"]).lstrip("/")
        img_path = self.image_root / relative_path

        image = Image.open(img_path).convert("RGB")
        image = apply_distortion(image, self.distortion_name)

        labels = torch.tensor(
            row[LABELS].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels


def load_model(model_name, checkpoint_path, device):
    """
    Loads a trained model checkpoint.
    """

    if model_name == "baseline_cnn":
        model = BaselineCNN(num_labels=len(LABELS))
    else:
        model = build_model(
            model_name=model_name,
            num_labels=len(LABELS),
            pretrained=False,
            freeze_backbone=False
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def create_loader(distortion_name):
    """
    Creates a test DataLoader for one distortion condition.
    """

    dataset = RobustnessDataset(
        csv_path=TEST_CSV,
        image_root=IMAGE_ROOT,
        distortion_name=distortion_name,
        transform=val_test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return loader


def evaluate_model_on_distortion(model, loader, device):
    """
    Evaluates one model under one distortion condition.
    """

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True
    )

    return metrics, report


def save_distortion_examples():
    """
    Saves example images showing how each distortion changes one sample image.
    """

    example_dir = RESULTS_DIR / "distortion_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    sample_row = df.iloc[0]

    relative_path = str(sample_row["filePath"]).lstrip("/")
    img_path = IMAGE_ROOT / relative_path

    image = Image.open(img_path).convert("RGB")

    for distortion in DISTORTIONS:
        distorted = apply_distortion(image, distortion)
        save_path = example_dir / f"{distortion}.png"
        distorted.save(save_path)


def plot_robustness_comparison(df):
    """
    Saves a line chart comparing Macro F1 under each distortion.
    """

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        plt.plot(
            model_df["distortion"],
            model_df["macro_f1"],
            marker="o",
            label=model_name
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1")
    plt.xlabel("Distortion Condition")
    plt.title("Robustness Comparison by Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "robustness_macro_f1_comparison.png", dpi=200)
    plt.close()


def create_drop_table(df):
    """
    Calculates performance drop from clean Macro F1 for each model.
    """

    rows = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]

        clean_row = model_df[model_df["distortion"] == "clean"]

        if clean_row.empty:
            continue

        clean_macro_f1 = clean_row.iloc[0]["macro_f1"]

        for _, row in model_df.iterrows():
            drop = clean_macro_f1 - row["macro_f1"]

            new_row = row.to_dict()
            new_row["clean_macro_f1"] = clean_macro_f1
            new_row["macro_f1_drop_from_clean"] = drop
            rows.append(new_row)

    drop_df = pd.DataFrame(rows)

    return drop_df


def save_summary(drop_df):
    """
    Creates a Markdown summary of robustness findings.
    """

    summary_path = RESULTS_DIR / "robustness_summary.md"

    worst_rows = drop_df[drop_df["distortion"] != "clean"].sort_values(
        by="macro_f1_drop_from_clean",
        ascending=False
    )

    if len(worst_rows) > 0:
        worst_case = worst_rows.iloc[0]
        worst_text = (
            f"The largest Macro F1 drop was observed for "
            f"{worst_case['model']} under {worst_case['distortion']} "
            f"with a drop of {worst_case['macro_f1_drop_from_clean']:.4f}."
        )
    else:
        worst_text = "No robustness drop could be calculated."

    avg_drop = (
        drop_df[drop_df["distortion"] != "clean"]
        .groupby("model")["macro_f1_drop_from_clean"]
        .mean()
        .sort_values()
    )

    summary = "# Week 12 Robustness Summary\n\n"

    summary += "## Purpose\n"
    summary += (
        "The purpose of this experiment was to test whether trained maize leaf "
        "disease models remain stable under controlled image distortions that "
        "simulate realistic field-image variation.\n\n"
    )

    summary += "## Distortions Tested\n"
    for distortion in DISTORTIONS:
        summary += f"- {distortion}\n"

    summary += "\n## Main Robustness Finding\n"
    summary += worst_text + "\n\n"

    summary += "## Average Macro F1 Drop by Model\n"
    for model_name, value in avg_drop.items():
        summary += f"- {model_name}: {value:.4f}\n"

    summary += "\n## Interpretation Template\n"
    summary += (
        "The model with the smallest average Macro F1 drop is considered the most "
        "robust under the tested image distortions. This result should be interpreted "
        "together with Week 11 efficiency metrics to determine which model provides "
        "the best practical trade-off.\n"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting Week 12 robustness experiments...")
    print(f"Device: {device}")

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing test split: {TEST_CSV}")

    save_distortion_examples()

    all_results = []
    all_reports_dir = RESULTS_DIR / "classification_reports"
    all_reports_dir.mkdir(parents=True, exist_ok=True)

    for model_name, checkpoint_path_str in MODELS.items():
        checkpoint_path = Path(checkpoint_path_str)

        if not checkpoint_path.exists():
            print(f"Skipping {model_name}. Missing checkpoint: {checkpoint_path}")
            continue

        print(f"\nLoading model: {model_name}")
        model = load_model(model_name, checkpoint_path, device)

        for distortion in DISTORTIONS:
            print(f"Evaluating {model_name} under {distortion}...")

            loader = create_loader(distortion)
            metrics, report = evaluate_model_on_distortion(
                model=model,
                loader=loader,
                device=device
            )

            row = {
                "model": model_name,
                "distortion": distortion,
                **metrics
            }

            all_results.append(row)

            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(
                all_reports_dir / f"{model_name}_{distortion}_classification_report.csv"
            )

    results_df = pd.DataFrame(all_results)

    raw_results_path = RESULTS_DIR / "robustness_metrics.csv"
    results_df.to_csv(raw_results_path, index=False)

    drop_df = create_drop_table(results_df)
    drop_path = RESULTS_DIR / "robustness_drop_table.csv"
    drop_df.to_csv(drop_path, index=False)

    plot_robustness_comparison(results_df)
    save_summary(drop_df)

    print("\nRobustness experiments complete.")
    print(f"Raw metrics saved to: {raw_results_path}")
    print(f"Drop table saved to: {drop_path}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()