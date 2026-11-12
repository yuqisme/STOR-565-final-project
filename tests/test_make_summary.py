"""Tests for src/make_summary.py on tiny synthetic tables."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import make_summary as ms  # noqa: E402


def _metrics(models, base):
    rows = {"model": models, "supervised": ["no", "yes"], "baseline_AUPRC_mean": [0.002] * 2}
    for i, m in enumerate(ms.METRICS):
        rows[f"{m}_mean"] = [base + 0.1 * i, base + 0.1 * i + 0.5]
        rows[f"{m}_std"] = [0.01, 0.0]
    return pd.DataFrame(rows)


def _cost(models, col, vals):
    return pd.DataFrame({"model": models * 2, "review_cost": [1, 1, 5, 5],
                         col: [0.0, 0.0] + vals})


MODELS = ["A", "B"]
VAL, TEST = _metrics(MODELS, 0.1), _metrics(MODELS[::-1], 0.2)      # test rows in another order
COST_V = _cost(MODELS, "reduction_vs_no_model_pct", [30.0, 60.0])
COST_T = _cost(MODELS, "reduction_vs_no_review_pct", [20.0, 50.0])


def test_build_summary_aligns_models_and_columns():
    s = ms.build_summary(VAL, TEST, COST_V, COST_T)
    assert list(s.model) == MODELS
    # test table was given as B, A: A's test row is the second row of TEST
    assert s.loc[0, "test_AUPRC_mean"] == pytest.approx(0.2 + 0.5)
    assert s.loc[1, "test_AUPRC_mean"] == pytest.approx(0.2)
    assert s.loc[0, "val_AUPRC_mean"] == pytest.approx(0.1)
    assert list(s.val_c5_saving_pct) == [30.0, 60.0] and list(s.test_c5_saving_pct) == [20.0, 50.0]
    assert {f"{p}_{m}_{x}" for p in ("val", "test") for m in ms.METRICS for x in ("mean", "std")} <= set(s.columns)


def test_markdown_hides_std_when_deterministic():
    same_order = _metrics(MODELS, 0.2)
    md = ms.to_markdown(ms.build_summary(VAL, same_order, COST_V, COST_T)).splitlines()
    assert md[0].startswith("| Method |") and len(md) == 4
    assert "±" in md[2] and "±" not in md[3]        # A has std 0.01, B has std 0 (deterministic)
