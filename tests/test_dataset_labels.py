import pandas as pd
from src.data.dataset import LABELS


def test_label_columns_exist():
    df = pd.read_csv("data/splits/train.csv")

    for label in LABELS:
        assert label in df.columns


def test_multilabel_width_is_8():
    df = pd.read_csv("data/splits/train.csv")
    y = df[LABELS].values

    assert y.shape[1] == 8