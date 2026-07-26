# This script splits the dataset into training, validation, and test sets.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = Path("data/raw/Database.csv")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

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

df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

for name, split in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df)
]:
    print(f"\n{name.upper()} split size: {len(split)}")
    print(split[LABELS].sum())
    split.to_csv(SPLIT_DIR / f"{name}.csv", index=False)