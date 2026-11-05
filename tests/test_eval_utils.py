"""Tests for src/eval_utils.py. Synthetic data only; no dataset files are read."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval_utils import (auprc, evaluate, plot_pr_curves, precision_at_k,  # noqa: E402
                        recall_at_k, results_table)

# Hand-computed example. Ranking by score: fraud, normal, fraud, normal.
#   threshold 0.9: P=1,   R=1/2
#   threshold 0.8: P=1/2, R=1/2
#   threshold 0.7: P=2/3, R=1
#   AP = (1/2)*1 + 0 + (1/2)*(2/3) = 5/6
Y = np.array([1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.1])


def test_auprc_hand_computed():
    assert auprc(Y, S) == pytest.approx(5 / 6)


def test_recall_at_k_hand_computed():
    assert recall_at_k(Y, S, 1) == pytest.approx(1 / 2)
    assert recall_at_k(Y, S, 2) == pytest.approx(1 / 2)
    assert recall_at_k(Y, S, 3) == pytest.approx(1.0)
    assert recall_at_k(Y, S, "n_pos") == pytest.approx(1 / 2)  # k = 2


def test_precision_at_k_hand_computed():
    assert precision_at_k(Y, S, 1) == pytest.approx(1.0)
    assert precision_at_k(Y, S, 2) == pytest.approx(1 / 2)
    assert precision_at_k(Y, S, 3) == pytest.approx(2 / 3)


def test_evaluate_hand_computed():
    res = evaluate(Y, S, ks=(1, 3, "n_pos"))
    assert res == pytest.approx({"AUPRC": 5 / 6, "recall@1": 0.5, "recall@3": 1.0,
                                 "recall@n_pos": 0.5, "baseline_AUPRC": 0.5})
    assert list(res) == ["AUPRC", "recall@1", "recall@3", "recall@n_pos", "baseline_AUPRC"]


def test_default_ks_keys():
    y = np.array([0, 1, 0, 1])
    assert list(evaluate(y, np.arange(4.0))) == [
        "AUPRC", "recall@100", "recall@200", "recall@n_pos", "baseline_AUPRC"]


def test_higher_score_means_more_anomalous():
    assert recall_at_k(Y, S, 1) == 0.5 and recall_at_k(Y, -S, 1) == 0.0
    assert auprc(Y, S) > auprc(Y, -S)


def test_k_larger_than_n_is_clipped():
    assert recall_at_k(Y, S, 1000) == 1.0
    assert precision_at_k(Y, S, 1000) == pytest.approx(0.5)  # 2 frauds / 4 points


def test_ties_all_equal_scores():
    y = np.array([1, 0, 0, 1])
    s = np.zeros(4)
    assert auprc(y, s) == pytest.approx(0.5)               # single threshold -> fraud rate
    assert recall_at_k(y, s, 1) == recall_at_k(y, s, 1)    # deterministic
    assert recall_at_k(y, s, 1) == pytest.approx(0.5)      # stable sort: row 0 first


def test_ties_depend_on_row_order():
    s = np.zeros(4)
    assert recall_at_k([1, 0, 0, 1], s, 1) == pytest.approx(0.5)
    assert recall_at_k([0, 0, 1, 1], s, 2) == pytest.approx(0.0)
    assert recall_at_k([1, 1, 0, 0], s, 2) == pytest.approx(1.0)
    assert recall_at_k([0, 0, 1, 1], s, 1) == pytest.approx(0.0)


def test_no_positives_raises():
    for f in (auprc, evaluate):
        with pytest.raises(ValueError, match="no positives"):
            f(np.zeros(5, dtype=int), np.arange(5.0))
    with pytest.raises(ValueError, match="no positives"):
        recall_at_k(np.zeros(5, dtype=int), np.arange(5.0), 2)


@pytest.mark.parametrize("y,s", [
    ([0, 1], [0.1]),               # length mismatch
    ([0, 2], [0.1, 0.2]),          # non-binary labels
    ([0, 0.5], [0.1, 0.2]),        # float labels are not silently truncated
    ([0, 1], [0.1, np.nan]),       # NaN score
    ([0, 1], [0.1, np.inf]),       # inf score
    ([[0, 1]], [[0.1, 0.2]]),      # not 1-D
])
def test_bad_inputs_raise(y, s):
    with pytest.raises(ValueError):
        evaluate(y, s)


@pytest.mark.parametrize("k", [0, -1, -3, 1.5, "all", True])
def test_bad_k_raises(k):
    # negative k used to slice from the end and return a wrong answer silently
    with pytest.raises(ValueError):
        recall_at_k(Y, S, k)
    with pytest.raises(ValueError):
        precision_at_k(Y, S, k)


def test_random_scores_auprc_near_fraud_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(int)
    res = evaluate(y, rng.random(20000))
    assert abs(res["AUPRC"] - res["baseline_AUPRC"]) < 0.01


def test_results_table():
    df = results_table(Y, {"good": S, "bad": -S}, ks=(2,))
    assert list(df.index) == ["good", "bad"]
    assert df.loc["good", "AUPRC"] == pytest.approx(round(5 / 6, 4))


def test_plot_saves_file_and_has_baseline(tmp_path):
    out = tmp_path / "pr.png"
    fig = plot_pr_curves(Y, {"m": S}, path=str(out))
    assert out.exists()
    ax = fig.axes[0]
    assert any(np.allclose(l.get_ydata(), Y.mean()) for l in ax.lines)
