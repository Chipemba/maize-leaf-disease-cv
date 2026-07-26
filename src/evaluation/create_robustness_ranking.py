"""
create_robustness_ranking.py

Creates a model-level robustness ranking using average Macro F1 drop.
"""

from pathlib import Path
import pandas as pd


DROP_TABLE = Path("results/week12_robustness/robustness_drop_table.csv")
OUTPUT_PATH = Path("results/week12_robustness/robustness_ranking.csv")


def main():
    df = pd.read_csv(DROP_TABLE)

    distorted_df = df[df["distortion"] != "clean"]

    ranking = (
        distorted_df
        .groupby("model")
        .agg(
            avg_macro_f1=("macro_f1", "mean"),
            avg_macro_f1_drop=("macro_f1_drop_from_clean", "mean"),
            worst_macro_f1_drop=("macro_f1_drop_from_clean", "max"),
            best_distorted_macro_f1=("macro_f1", "max"),
            worst_distorted_macro_f1=("macro_f1", "min")
        )
        .reset_index()
        .sort_values(by="avg_macro_f1_drop", ascending=True)
    )

    ranking.to_csv(OUTPUT_PATH, index=False)

    print("Saved robustness ranking to:", OUTPUT_PATH)
    print(ranking)


if __name__ == "__main__":
    main()