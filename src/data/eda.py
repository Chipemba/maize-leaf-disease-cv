from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("data/raw/Database.csv")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

class_counts = df[LABELS].sum().sort_values(ascending=False)
print("\nClass distribution:")
print(class_counts)

label_count_per_image = df[LABELS].sum(axis=1)
print("\nNumber of labels per image:")
print(label_count_per_image.value_counts().sort_index())

class_counts.to_csv(TABLE_DIR / "class_distribution.csv")

summary = pd.DataFrame({
    "total_images": [len(df)],
    "images_with_1_label": [(label_count_per_image == 1).sum()],
    "images_with_2_or_more_labels": [(label_count_per_image >= 2).sum()],
    "max_labels_on_one_image": [label_count_per_image.max()]
})
summary.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)

plt.figure(figsize=(10, 5))
class_counts.plot(kind="bar")
plt.title("Class Distribution in Database.csv")
plt.ylabel("Number of Positive Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
label_count_per_image.value_counts().sort_index().plot(kind="bar")
plt.title("Multi-label Distribution")
plt.xlabel("Number of Positive Labels per Image")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(FIG_DIR / "multilabel_distribution.png", dpi=200)
plt.close()

cooccurrence = df[LABELS].T.dot(df[LABELS])
cooccurrence.to_csv(TABLE_DIR / "label_cooccurrence.csv")

plt.figure(figsize=(8, 6))
plt.imshow(cooccurrence, aspect="auto")
plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
plt.yticks(range(len(LABELS)), LABELS)
plt.title("Label Co-occurrence Heatmap")
plt.colorbar()
plt.tight_layout()
plt.savefig(FIG_DIR / "label_cooccurrence_heatmap.png", dpi=200)
plt.close()