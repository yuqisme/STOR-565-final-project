"""
data_loader.py  --  shared loader for the pre-split credit card files.

Reads data/random_split/{train,validation,test}.csv, made once by
make_random_split.py (stratified random 60/20/20, seed 0). Does NOT split anything.
Features are raw (unstandardized) and include Time. Labels: 1 = fraud.

Rule: only final_test.py may call load_test(). load_train_val() never opens
the test file.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "random_split"
FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
LABEL = "Class"
DROP_COLS = ()          # Time is kept; pass drop_cols=("Time",) to remove it


def load_split(name, data_dir=None, drop_cols=DROP_COLS):
    """Load one split ('train', 'val' or 'test').

    Returns (X, y): X is a DataFrame of raw features (LABEL and drop_cols
    removed), y is an int ndarray with 1 = fraud. Time is kept by default.
    """
    if name not in FILES:
        raise ValueError(f"name must be one of {sorted(FILES)}, got {name!r}")
    df = pd.read_csv(Path(data_dir or DATA_DIR) / FILES[name])
    if LABEL not in df.columns:
        raise ValueError(f"{FILES[name]} has no '{LABEL}' column")
    if df.isnull().any().any():
        raise ValueError(f"{FILES[name]} contains missing values")
    if not df[LABEL].isin([0, 1]).all():
        raise ValueError(f"{LABEL} must be 0/1")
    X = df.drop(columns=[LABEL, *[c for c in drop_cols if c in df.columns]])
    return X, df[LABEL].to_numpy().astype(int)


def load_train_val(data_dir=None, drop_cols=DROP_COLS):
    """Return X_train, y_train, X_val, y_val. Never opens the test file."""
    X_tr, y_tr = load_split("train", data_dir, drop_cols)
    X_va, y_va = load_split("val", data_dir, drop_cols)
    return X_tr, y_tr, X_va, y_va


def load_test(data_dir=None, drop_cols=DROP_COLS):
    """Return X_test, y_test. For final_test.py ONLY."""
    return load_split("test", data_dir, drop_cols)
