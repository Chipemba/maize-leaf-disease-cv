"""
create_final_comparison.py

Combines Week 11 accuracy/efficiency results with Week 12 robustness results.
"""

from pathlib import Path
import pandas as pd


WEEK11_TABLE = Path("results/week11_evaluation/model_comparison_table.csv")
ROBUSTNESS_RANKING = Path("results/week12_robustness/robustness_ranking.csv")
OUTPUT_PATH = Path("results/week12_robustness/final_accuracy_efficiency_robustness_table.csv")


def main():
    week11 = pd.read_csv(WEEK11_TABLE)
    robust = pd.read_csv(ROBUSTNESS_RANKING)

    final_df = week11.merge(
        robust,
        on="model",
        how="inner"
    )

    final_df = final_df.sort_values(
        by=["macro_f1", "avg_macro_f1_drop", "images_per_second"],
        ascending=[False, True, False]
    )

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved final comparison table to:", OUTPUT_PATH)
    print(final_df)


if __name__ == "__main__":
    main()