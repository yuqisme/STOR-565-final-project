"""Tests for src/check_split.py on tiny synthetic split files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import check_split as cs  # noqa: E402


def _frame(n, n_pos, start):
    d = pd.DataFrame({c: np.arange(start, start + n, dtype=float) for c in cs.EXPECTED_COLUMNS})
    d[cs.LABEL] = 0
    d.loc[: n_pos - 1, cs.LABEL] = 1
    return d


def _write(tmp_path, sizes=((600, 6), (200, 2), (200, 2))):
    start = 0
    for (n, n_pos), fname in zip(sizes, ["train.csv", "validation.csv", "test.csv"]):
        _frame(n, n_pos, start).to_csv(tmp_path / fname, index=False)
        start += n
    return tmp_path


def _check(d):
    return cs.check(d, expected_sha=None, total_rows=1000, total_frauds=10)


def test_valid_split_passes(tmp_path):
    assert _check(_write(tmp_path)) == []


def test_overlap_is_caught(tmp_path):
    d = _write(tmp_path)
    tr = pd.read_csv(d / "train.csv")
    va = pd.read_csv(d / "validation.csv")
    va.iloc[5] = tr.iloc[100]                       # same row in train and validation
    va.to_csv(d / "validation.csv", index=False)
    assert any("duplicate" in f for f in _check(d))


def test_wrong_proportions_are_caught(tmp_path):
    d = _write(tmp_path, sizes=((800, 8), (100, 1), (100, 1)))
    assert any("expected about" in f for f in _check(d))


def test_missing_file_is_caught(tmp_path):
    d = _write(tmp_path)
    (d / "test.csv").unlink()
    assert any("missing file" in f for f in _check(d))
