"""Tests for src/final_test.py on synthetic data. The real test set is never loaded."""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import final_test as ft  # noqa: E402

Y = np.array([1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.1])
AMT = np.array([100.0, 10.0, 50.0, 5.0])


def test_policy_row_hand_computed():
    r = ft.policy_row("XGBoost", Y, S, AMT, review_cost=5, threshold=0.7)
    assert r["n_flagged"] == 3 and r["frauds_flagged"] == 2
    assert r["fraud_amount_caught"] == 150 and r["missed_fraud_amount"] == 0
    assert r["total_cost"] == 15                       # 0 missed + 5 * 3 reviewed
    assert r["no_review_cost"] == 150 and r["review_all_cost"] == 20
    assert r["reduction_vs_no_review_pct"] == pytest.approx(90)
    assert r["reduction_vs_review_all_pct"] == pytest.approx(25)


def test_policy_row_infinite_threshold_flags_nothing():
    r = ft.policy_row("XGBoost", Y, S, AMT, review_cost=5, threshold=np.inf)
    assert r["n_flagged"] == 0 and r["total_cost"] == 150
    assert r["reduction_vs_no_review_pct"] == pytest.approx(0)


def test_policy_threshold_is_inclusive():
    assert ft.policy_row("XGBoost", Y, S, AMT, 1, threshold=0.9)["n_flagged"] == 1


def test_budget_rows_n_pos_uses_given_split():
    (r,) = ft.budget_rows("IF default", Y, S, AMT, budgets=("n_pos",))
    assert r["k"] == 2 and r["budget"] == "n_pos"
    assert r["count_recall"] == pytest.approx(0.5)                 # top-2 = fraud, normal
    assert r["amount_recall"] == pytest.approx(100 / 150)
    assert r["total_fraud_amount"] == 150


def test_primary_review_cost_is_five():
    assert ft.PRIMARY_REVIEW_COST == 5


def test_refuses_without_flag():
    with pytest.raises(SystemExit, match="--run-final-test"):
        ft.main([])


def _never_load(*a, **k):
    raise AssertionError("test set must not be loaded")


def test_refuses_if_already_run(tmp_path, monkeypatch):
    (tmp_path / ft.MAIN_OUTPUT).write_text("done")
    monkeypatch.setattr(ft, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ft, "load_test", _never_load)
    with pytest.raises(SystemExit, match="already"):
        ft.main(["--run-final-test"])


def test_refuses_if_train_val_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ft, "load_config", lambda name: {"data_sha256": {"train.csv": "x"}})
    monkeypatch.setattr(ft, "data_fingerprints", lambda: {"train.csv": "y"})
    monkeypatch.setattr(ft, "load_test", _never_load)
    with pytest.raises(SystemExit, match="differ"):
        ft.main(["--run-final-test"])


def test_only_final_test_calls_load_test():
    for f in SRC.glob("*.py"):
        if f.name in ("final_test.py", "data_loader.py"):
            continue
        assert "load_test(" not in f.read_text(), f"{f.name} must not load the test set"
