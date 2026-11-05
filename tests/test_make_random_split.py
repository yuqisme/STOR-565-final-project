"""Tests for make_random_split.split() on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_random_split import split  # noqa: E402


def _df(n=2000, n_pos=40):
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"Time": np.arange(n), "V1": rng.normal(size=n), "Class": 0})
    d.loc[rng.choice(n, n_pos, replace=False), "Class"] = 1
    return d


def test_sizes_disjoint_and_complete():
    d = _df()
    tr, va, te = split(d)
    assert (len(tr), len(va), len(te)) == (1200, 400, 400)
    idx = [set(x.index) for x in (tr, va, te)]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert set().union(*idx) == set(d.index)


def test_stratified():
    tr, va, te = split(_df())
    assert (tr.Class.sum(), va.Class.sum(), te.Class.sum()) == (24, 8, 8)


def test_seeded_and_not_time_ordered():
    d = _df()
    a, b = split(d, seed=0)[0], split(d, seed=0)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not a.Time.is_monotonic_increasing
