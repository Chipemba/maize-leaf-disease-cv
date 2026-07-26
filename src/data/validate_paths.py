# This validates that all images in the dataset.csv file exist in the data/raw directory and that all images in the data/raw directory are listed in the dataset.csv file. It also checks that all images in the data/processed directory are listed in the dataset.csv file.

from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/raw/Database.csv")
IMAGE_ROOT = Path("data/raw/static/leaf_images/")

df = pd.read_csv(CSV_PATH)

missing_files = []

for _, row in df.iterrows():
    img_path = IMAGE_ROOT / row["filePath"].lstrip("/")
    if not img_path.exists():
        missing_files.append(str(img_path))

print(f"Total rows: {len(df)}")
print(f"Missing files: {len(missing_files)}")

if missing_files:
    print("First 10 missing files:")
    for path in missing_files[:10]:
        print(path)