"""
evaluate_baseline.py

Evaluates the Week 9 custom baseline CNN for the maize leaf disease project.

This script is designed for multi-label classification, where one image can have
more than one disease/condition label.

Expected input:
- data/splits/test.csv
- data/raw/static/leaf_images/...
- models/baseline_cnn.pt

Expected output:
- results/week09_baseline/baseline_overall_metrics.json
- results/week09_baseline/baseline_classification_report.csv
- results/week09_baseline/baseline_predictions.csv
- results/week09_baseline/baseline_best_examples.csv
- results/week09_baseline/baseline_worst_examples.csv
- one-vs-rest confusion matrix image for each label
- results/week09_baseline/baseline_evaluation_summary.md
"""

from pathlib import Path
import json

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


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

TEST_CSV = Path("data/splits/test.csv")
IMAGE_ROOT = Path("data/raw/static/leaf_images/")
MODEL_PATH = Path("models/baseline_cnn.pt")

RESULTS_DIR = Path("results/week09_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 2
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------

def load_test_dataset():
    """
    Loads the test dataset using the same validation/test transforms.

    Returns:
        test_dataset: PyTorch Dataset
        test_loader: PyTorch DataLoader
    """

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image root folder not found: {IMAGE_ROOT}")

    test_dataset = MaizeMultiLabelDataset(
        csv_path=TEST_CSV,
        image_root=IMAGE_ROOT,
        transform=val_test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return test_dataset, test_loader


def load_model():
    """
    Loads the trained baseline CNN model.

    Returns:
        model: trained PyTorch model in evaluation mode
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = BaselineCNN(num_labels=len(LABELS)).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    return model


def collect_predictions(model, test_loader):
    """
    Runs the trained model on the test set and collects:
    - true labels
    - predicted probabilities
    - binary predictions after thresholding

    Args:
        model: trained baseline CNN
        test_loader: DataLoader for test data

    Returns:
        y_true: true label matrix, shape [num_images, num_labels]
        y_prob: predicted probability matrix, shape [num_images, num_labels]
        y_pred: binary prediction matrix, shape [num_images, num_labels]
    """

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            logits = model(images)

            # BCEWithLogitsLoss was used during training, so the model outputs raw logits.
            # During evaluation, sigmoid converts those logits into probabilities.
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    # Convert probabilities to 0/1 predictions using the selected threshold.
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return y_true, y_prob, y_pred


def calculate_overall_metrics(y_true, y_pred):
    """
    Calculates the main multi-label classification metrics.

    Returns:
        metrics: dictionary of overall metric values
        report_dict: detailed per-class classification report
    """

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    hamming = hamming_loss(y_true, y_pred)

    # Exact match ratio is strict.
    # It only counts an image as correct if all labels are predicted correctly.
    exact_match = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "threshold": THRESHOLD,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "hamming_loss": float(hamming),
        "exact_match_ratio": float(exact_match),
        "num_test_images": int(y_true.shape[0]),
        "num_labels": int(y_true.shape[1]),
    }

    return metrics, report_dict


def save_metrics(metrics, report_dict):
    """
    Saves overall metrics as JSON and per-class report as CSV.
    """

    metrics_path = RESULTS_DIR / "baseline_overall_metrics.json"
    report_path = RESULTS_DIR / "baseline_classification_report.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_path)

    print(f"Saved overall metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")


def save_predictions(y_true, y_prob, y_pred):
    """
    Saves true labels, predicted probabilities, and final binary predictions
    for every image in the test set.
    """

    test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)

    true_df = pd.DataFrame(
        y_true,
        columns=[f"{label}_true" for label in LABELS]
    )

    prob_df = pd.DataFrame(
        y_prob,
        columns=[f"{label}_prob" for label in LABELS]
    )

    pred_df = pd.DataFrame(
        y_pred,
        columns=[f"{label}_pred" for label in LABELS]
    )

    output_df = pd.concat(
        [
            test_df[["imgID_id", "filePath"]],
            true_df,
            prob_df,
            pred_df
        ],
        axis=1
    )

    # Count how many label decisions were wrong per image.
    label_errors = np.abs(y_true - y_pred).sum(axis=1)
    output_df["num_label_errors"] = label_errors

    predictions_path = RESULTS_DIR / "baseline_predictions.csv"
    output_df.to_csv(predictions_path, index=False)

    print(f"Saved prediction details to: {predictions_path}")

    return output_df


def save_best_and_worst_examples(output_df):
    """
    Saves the best and worst prediction examples.

    Best examples:
    - Images where all labels were predicted correctly.

    Worst examples:
    - Images with the most label-level errors.
    """

    worst_examples = output_df.sort_values(
        by="num_label_errors",
        ascending=False
    ).head(10)

    best_examples = output_df[
        output_df["num_label_errors"] == 0
    ].head(10)

    worst_path = RESULTS_DIR / "baseline_worst_examples.csv"
    best_path = RESULTS_DIR / "baseline_best_examples.csv"

    worst_examples.to_csv(worst_path, index=False)
    best_examples.to_csv(best_path, index=False)

    print(f"Saved worst examples to: {worst_path}")
    print(f"Saved best examples to: {best_path}")


def save_confusion_matrices(y_true, y_pred):
    """
    Saves one-vs-rest confusion matrices for each label.

    Multi-label classification does not use one normal confusion matrix.
    Instead, each label gets its own binary confusion matrix:
    - label absent vs label present
    """

    cms = multilabel_confusion_matrix(y_true, y_pred)

    for i, label in enumerate(LABELS):
        cm = cms[i]

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[f"No {label}", label]
        )

        display.plot(values_format="d")
        plt.title(f"One-vs-Rest Confusion Matrix: {label}")
        plt.tight_layout()

        save_path = RESULTS_DIR / f"confusion_matrix_{label}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved confusion matrix for {label}: {save_path}")


def create_evaluation_summary(metrics, report_dict):
    """
    Creates a short Markdown summary that can be used in the report,
    GitHub issue, or Week 12 presentation preparation.
    """

    report_df = pd.DataFrame(report_dict).transpose()

    # Extract only actual label rows.
    class_rows = report_df.loc[LABELS]

    best_label = class_rows["f1-score"].idxmax()
    worst_label = class_rows["f1-score"].idxmin()

    summary_path = RESULTS_DIR / "baseline_evaluation_summary.md"

    summary_text = f"""# Week 9 Baseline Evaluation Summary

## Model
Custom baseline CNN with 8 output logits.

## Task Type
Multi-label maize leaf disease and condition classification.

## Loss Function
BCEWithLogitsLoss was used during training because each image can contain more than one label.

## Threshold
A threshold of `{THRESHOLD}` was used to convert sigmoid probabilities into binary predictions.

## Overall Metrics
- Micro F1: {metrics["micro_f1"]:.4f}
- Macro F1: {metrics["macro_f1"]:.4f}
- Weighted F1: {metrics["weighted_f1"]:.4f}
- Hamming Loss: {metrics["hamming_loss"]:.4f}
- Exact Match Ratio: {metrics["exact_match_ratio"]:.4f}

## Best and Weakest Labels
- Best-performing label by F1-score: {best_label}
- Weakest-performing label by F1-score: {worst_label}

## Interpretation Notes
The baseline CNN provides a starting reference point for the project. Its results should not be treated as the final model performance. The purpose of this model is to establish whether the dataset pipeline, multi-label target format, training loop, and evaluation metrics are working correctly.

Low macro F1 compared with micro F1 may indicate that the model performs better on common labels than rare labels. This is important for this dataset because some maize disease labels may be underrepresented.

## Planned Improvements
In Week 10, stronger transfer learning and lightweight models will be trained and compared against this baseline. In Week 11 and Week 12, efficiency and robustness testing will be added to evaluate practical field-readiness.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Saved evaluation summary to: {summary_path}")


# ---------------------------------------------------------
# 3. Main evaluation pipeline
# ---------------------------------------------------------

def main():
    print("Starting baseline evaluation...")
    print(f"Using device: {DEVICE}")

    print("\nLoading test dataset...")
    _, test_loader = load_test_dataset()

    print("\nLoading baseline CNN model...")
    model = load_model()

    print("\nCollecting predictions...")
    y_true, y_prob, y_pred = collect_predictions(model, test_loader)

    print("\nCalculating metrics...")
    metrics, report_dict = calculate_overall_metrics(y_true, y_pred)

    print("\nOverall metrics:")
    print(json.dumps(metrics, indent=4))

    print("\nSaving metrics...")
    save_metrics(metrics, report_dict)

    print("\nSaving prediction details...")
    output_df = save_predictions(y_true, y_prob, y_pred)

    print("\nSaving best and worst examples...")
    save_best_and_worst_examples(output_df)

    print("\nSaving one-vs-rest confusion matrices...")
    save_confusion_matrices(y_true, y_pred)

    print("\nCreating evaluation summary...")
    create_evaluation_summary(metrics, report_dict)

    print("\nBaseline evaluation complete.")


if __name__ == "__main__":
    main()