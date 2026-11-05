"""Tests for src/data_loader.py using tiny synthetic CSVs (real data is not read)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_loader as dl  # noqa: E402


def _write(d, name, times, labels):
    n = len(times)
    pd.DataFrame({"Time": times, "V1": np.arange(n, dtype=float),
                  "Amount": np.ones(n), "Class": labels}).to_csv(d / name, index=False)


@pytest.fixture
def ddir(tmp_path):
    _write(tmp_path, "train.csv", [0, 1, 2, 3], [0, 0, 1, 0])
    _write(tmp_path, "validation.csv", [4, 5, 6], [0, 1, 0])
    _write(tmp_path, "test.csv", [7, 8], [0, 1])
    return tmp_path


def test_train_val_format(ddir):
    X_tr, y_tr, X_va, y_va = dl.load_train_val(ddir)
    assert list(X_tr.columns) == ["Time", "V1", "Amount"]   # Time kept, Class removed
    assert X_tr.shape == (4, 3) and X_va.shape == (3, 3)
    assert y_tr.tolist() == [0, 0, 1, 0] and y_va.tolist() == [0, 1, 0]
    assert y_tr.dtype.kind == "i"


def test_drop_time_optional(ddir):
    X, _ = dl.load_split("train", ddir, drop_cols=("Time",))
    assert "Time" not in X.columns


def test_load_train_val_never_opens_test(ddir):
    (ddir / "test.csv").unlink()                             # would raise if it were read
    dl.load_train_val(ddir)


def test_load_test(ddir):
    X, y = dl.load_test(ddir)
    assert len(X) == 2 and y.tolist() == [0, 1]


def test_bad_name_and_bad_labels(ddir):
    with pytest.raises(ValueError):
        dl.load_split("valid", ddir)
    _write(ddir, "train.csv", [0, 1], [0, 2])
    with pytest.raises(ValueError, match="0/1"):
        dl.load_split("train", ddir)


def test_missing_values_raise(ddir):
    pd.DataFrame({"Time": [0, 1], "V1": [1.0, np.nan], "Class": [0, 1]}).to_csv(
        ddir / "train.csv", index=False)
    with pytest.raises(ValueError, match="missing"):
        dl.load_split("train", ddir)
