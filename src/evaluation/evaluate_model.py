"""
evaluate_model.py

General evaluation script for Week 11.

Evaluates baseline CNN, transfer learning models, and lightweight models
for multi-label maize leaf disease classification.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    accuracy_score,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data.dataset import MaizeMultiLabelDataset, LABELS
from src.data.transforms import val_test_transform
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import (
    build_model,
    count_total_parameters,
    count_trainable_parameters,
)


TEST_CSV = Path("data/splits/test.csv")
IMAGE_ROOT = Path("data/raw/static/leaf_images/")
BATCH_SIZE = 32
NUM_WORKERS = 2
THRESHOLD = 0.5


def load_model(model_name, checkpoint_path, device):
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


def load_test_loader():
    dataset = MaizeMultiLabelDataset(
        csv_path=TEST_CSV,
        image_root=IMAGE_ROOT,
        transform=val_test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return dataset, loader


def run_inference(model, loader, device):
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

    return y_true, y_prob, y_pred


def measure_inference_time(model, loader, device):
    model.eval()

    # Warm-up
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 3:
                break
            images = images.to(device)
            _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    num_images = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            num_images += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_image = total_time / num_images
    images_per_second = num_images / total_time

    return {
        "total_inference_time_seconds": total_time,
        "avg_inference_time_per_image_seconds": avg_time_per_image,
        "images_per_second": images_per_second,
        "num_images_timed": num_images,
    }


def calculate_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "threshold": THRESHOLD,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
        "num_test_images": int(y_true.shape[0]),
        "num_labels": int(y_true.shape[1]),
    }

    return metrics, report


def save_confusion_matrices(y_true, y_pred, output_dir):
    cms = multilabel_confusion_matrix(y_true, y_pred)

    for i, label in enumerate(LABELS):
        display = ConfusionMatrixDisplay(
            confusion_matrix=cms[i],
            display_labels=[f"No {label}", label]
        )

        display.plot(values_format="d")
        plt.title(f"One-vs-Rest Confusion Matrix: {label}")
        plt.tight_layout()

        save_path = output_dir / f"confusion_matrix_{label}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()


def save_predictions(y_true, y_prob, y_pred, output_dir):
    test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)

    true_df = pd.DataFrame(y_true, columns=[f"{label}_true" for label in LABELS])
    prob_df = pd.DataFrame(y_prob, columns=[f"{label}_prob" for label in LABELS])
    pred_df = pd.DataFrame(y_pred, columns=[f"{label}_pred" for label in LABELS])

    output_df = pd.concat(
        [
            test_df[["imgID_id", "filePath"]],
            true_df,
            prob_df,
            pred_df
        ],
        axis=1
    )

    output_df["num_label_errors"] = np.abs(y_true - y_pred).sum(axis=1)

    output_df.to_csv(output_dir / "predictions.csv", index=False)

    worst_examples = output_df.sort_values(
        by="num_label_errors",
        ascending=False
    ).head(10)

    best_examples = output_df[
        output_df["num_label_errors"] == 0
    ].head(10)

    worst_examples.to_csv(output_dir / "worst_examples.csv", index=False)
    best_examples.to_csv(output_dir / "best_examples.csv", index=False)


def save_summary(model_name, checkpoint_path, metrics, efficiency, output_dir):
    summary = f"""# Week 11 Evaluation Summary: {model_name}

## Model
{model_name}

## Checkpoint
{checkpoint_path}

## Task Type
Multi-label maize leaf disease and condition classification.

## Threshold
{THRESHOLD}

## Classification Metrics
- Micro F1: {metrics["micro_f1"]:.4f}
- Macro F1: {metrics["macro_f1"]:.4f}
- Weighted F1: {metrics["weighted_f1"]:.4f}
- Hamming Loss: {metrics["hamming_loss"]:.4f}
- Exact Match Ratio: {metrics["exact_match_ratio"]:.4f}

## Efficiency Metrics
- Model size MB: {efficiency["model_size_mb"]:.2f}
- Total parameters: {efficiency["total_parameters"]:,}
- Trainable parameters: {efficiency["trainable_parameters"]:,}
- Average inference time per image: {efficiency["avg_inference_time_per_image_seconds"]:.6f} seconds
- Images per second: {efficiency["images_per_second"]:.2f}

## Initial Interpretation
This model should be compared against the baseline CNN, transfer learning model, and lightweight model to identify the best balance between performance and efficiency.
"""

    with open(output_dir / "evaluation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    model_name = args.model_name
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("results/week11_evaluation") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    dataset, loader = load_test_loader()
    model = load_model(model_name, checkpoint_path, device)

    y_true, y_prob, y_pred = run_inference(model, loader, device)

    metrics, report = calculate_metrics(y_true, y_pred)

    timing = measure_inference_time(model, loader, device)

    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    efficiency = {
        "model_size_mb": float(model_size_mb),
        "total_parameters": int(count_total_parameters(model)),
        "trainable_parameters": int(count_trainable_parameters(model)),
        **timing
    }

    with open(output_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "efficiency_metrics.json", "w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=4)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")

    save_predictions(y_true, y_prob, y_pred, output_dir)
    save_confusion_matrices(y_true, y_pred, output_dir)
    save_summary(model_name, checkpoint_path, metrics, efficiency, output_dir)

    print("Evaluation complete.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()