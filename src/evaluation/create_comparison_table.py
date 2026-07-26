"""
create_comparison_table.py

Combines Week 11 model evaluation and efficiency metrics into one comparison table.
"""

import json
from pathlib import Path

import pandas as pd


EVAL_ROOT = Path("results/week11_evaluation")
OUTPUT_PATH = EVAL_ROOT / "model_comparison_table.csv"

MODELS = [
    "baseline_cnn",
    "efficientnet_b0",
    "mobilenet_v2",
    "resnet18"
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    for model_name in MODELS:
        model_dir = EVAL_ROOT / model_name

        metrics_path = model_dir / "overall_metrics.json"
        efficiency_path = model_dir / "efficiency_metrics.json"

        if not metrics_path.exists():
            print(f"Missing metrics file for {model_name}: {metrics_path}")
            continue

        if not efficiency_path.exists():
            print(f"Missing efficiency file for {model_name}: {efficiency_path}")
            continue

        metrics = load_json(metrics_path)
        efficiency = load_json(efficiency_path)

        row = {
            "model": model_name,
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "hamming_loss": metrics["hamming_loss"],
            "exact_match_ratio": metrics["exact_match_ratio"],
            "model_size_mb": efficiency["model_size_mb"],
            "total_parameters": efficiency["total_parameters"],
            "avg_inference_time_sec": efficiency["avg_inference_time_per_image_seconds"],
            "images_per_second": efficiency["images_per_second"],
        }

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    comparison_df = comparison_df.sort_values(
        by=["macro_f1", "images_per_second"],
        ascending=[False, False]
    )

    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved comparison table to:", OUTPUT_PATH)
    print(comparison_df)


if __name__ == "__main__":
    main()