"""Baseline and amount-policy checks; all data here is synthetic."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_utils import make_knn, make_logistic, tie_diagnostics
from compare_baselines import analyze_failures, build_model, feature_view, model_scores
from fraud_amount_analysis import amount_recall, optimal_cost


def test_knn_direction_shape_and_no_training_labels():
    train = np.array([[-.2, .1], [0., 0.], [.2, -.1], [.1, .1]])
    val = np.array([[.05, .02], [8., 8.]])
    model = make_knn(k=2, reference_size=4).fit(train)
    s = model.score_samples(val)
    assert s.shape == (2,) and np.isfinite(s).all()
    assert s[1] > s[0]
    model2 = make_knn(k=2, reference_size=4).fit(train, [1, 1, 1, 1])
    np.testing.assert_allclose(s, model2.score_samples(val))


@pytest.mark.parametrize("factory", [make_knn, make_logistic])
def test_scaler_fits_train_only(factory):
    train = np.array([[-2., 0.], [-1., 1.], [1., 3.], [2., 4.]])
    val = np.array([[100., 200.], [101., 201.]])
    kwargs = dict(k=1, reference_size=4) if factory is make_knn else {}
    model = factory(**kwargs).fit(train, np.array([0, 0, 1, 1]))
    before = model.named_steps["scaler"].mean_.copy()
    np.testing.assert_allclose(before, train.mean(axis=0))
    if factory is make_knn:
        model.score_samples(val)
    else:
        model.predict_proba(val)
    np.testing.assert_array_equal(before, model.named_steps["scaler"].mean_)
    assert model.named_steps["scaler"].n_samples_seen_ == len(train)


def test_logistic_positive_probability_direction():
    X = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    model = make_logistic().fit(X, [0, 0, 0, 1, 1, 1])
    scores = model.predict_proba([[-2.], [2.]])[:, 1]
    assert scores.shape == (2,) and scores[1] > scores[0]


def test_reference_sample_is_reproducible_and_train_scaler_uses_full_train():
    X = np.arange(100.).reshape(50, 2)
    a = make_knn(2, reference_size=10, seed=0).fit(X)
    b = make_knn(2, reference_size=10, seed=0).fit(X)
    c = make_knn(2, reference_size=10, seed=1).fit(X)
    ia = a.named_steps["knn"].reference_indices_
    assert len(ia) == len(np.unique(ia)) == 10
    np.testing.assert_array_equal(ia, b.named_steps["knn"].reference_indices_)
    assert not np.array_equal(ia, c.named_steps["knn"].reference_indices_)
    np.testing.assert_allclose(a.named_steps["scaler"].mean_, X.mean(axis=0))


def test_xgboost_scoring_shape_and_direction():
    X = np.r_[np.linspace(-3, -1, 30), np.linspace(1, 3, 30)][:, None]
    y = np.r_[np.zeros(30), np.ones(30)]
    model = build_model("XGBoost", dict(scale_pos_weight=1), seed=0).fit(X, y)
    scores = model_scores("XGBoost", model, np.array([[-2.], [2.]]))
    assert scores.shape == (2,) and scores[1] > scores[0]


def test_handoff_knn_feature_selection_and_scoring():
    config = dict(k=1, drop_time=True, standardized=True, reference_size=4)
    X = pd.DataFrame({"Time": [1, 2, 3, 4], "V1": [-.1, 0., .1, .2]})
    features = feature_view(X, config)
    assert list(features.columns) == ["V1"]
    model = build_model("kNN distance", config, seed=0).fit(features)
    val = pd.DataFrame({"Time": [999, 0], "V1": [.05, 8.]})
    s = model_scores("kNN distance", model, feature_view(val, config))
    assert s.shape == (2,) and s[1] > s[0]


def test_amount_recall_and_cost_hand_calculation():
    y, s, a = [1, 0, 1, 0], [.9, .8, .7, .1], [10, 99, 30, 5]
    assert amount_recall(y, s, a, 0) == 0
    assert amount_recall(y, s, a, 2) == .25
    assert amount_recall(y, s, a, 100) == 1
    result = optimal_cost(y, s, a, review_cost=5)
    # Costs for k=0..4: 40, 35, 40, 15, 20.
    assert result["chosen_k"] == result["inclusive_threshold_k"] == 3
    assert result["total_cost"] == result["inclusive_threshold_cost"] == 15
    assert result["inclusive_threshold"] == .7
    assert result["reduction_vs_no_model_pct"] == 62.5
    assert result["reduction_vs_review_all_pct"] == 25
    assert result["fraud_amount_caught"] == 40


def test_cost_search_includes_review_none_and_everything():
    assert optimal_cost([1, 0], [.9, .1], [1, 10], 20)["chosen_k"] == 0
    assert optimal_cost([0, 1], [.9, .1], [1, 10], 0)["chosen_k"] == 2


def test_cost_tie_blocks_have_distinct_deployable_threshold_policy():
    result = optimal_cost([1, 0], [.5, .5], [10, 0], 3)
    assert result["chosen_k"] == 1 and result["total_cost"] == 3
    assert result["cutoff_ties_total"] == 2 and result["cutoff_ties_flagged"] == 1
    assert result["inclusive_threshold_k"] == 2
    assert result["inclusive_threshold_cost"] == 6
    assert amount_recall([0, 1], [.5, .5], [0, 10], 1) == 0


def test_cost_matches_exhaustive_search():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 12)
        s = rng.integers(0, 4, 12)
        amounts = rng.integers(0, 50, 12)
        order = np.argsort(-s, kind="stable")
        costs = [(y[order[k:]] * amounts[order[k:]]).sum() + 5*k for k in range(13)]
        result = optimal_cost(y, s, amounts, 5)
        assert result["chosen_k"] == np.argmin(costs)
        assert result["total_cost"] == min(costs)


@pytest.mark.parametrize("y,s,a", [([1], [np.nan], [1]), ([2], [1], [1]),
                                    ([1], [1], [-1]), ([1, 0], [1], [1, 2])])
def test_amount_inputs_rejected(y, s, a):
    with pytest.raises(ValueError):
        amount_recall(y, s, a, 1)


def test_zero_fraud_amount_is_undefined():
    assert np.isnan(amount_recall([1, 0], [.9, .1], [0, 1], 1))


def test_failure_groups_partition_frauds(tmp_path, monkeypatch):
    import compare_baselines as cb
    monkeypatch.setattr(cb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cb, "FIGURES_DIR", tmp_path)
    X = pd.DataFrame({c: np.arange(205.) for c in ["Amount", "Time", "V14", "V17", "V12", "V10"]})
    y = np.zeros(205, dtype=int)
    y[[0, 201, 204]] = 1
    analyze_failures(X, y, {"a": -np.arange(205.), "b": np.arange(205.)})
    counts = pd.read_csv(tmp_path / "failure_counts_val.csv").set_index("group").fraud_count
    assert counts.sum() == 3 and counts["caught_some"] == 3 and counts["missed_all"] == 0


def test_baseline_scripts_never_import_test_loader():
    root = Path(__file__).resolve().parents[1] / "src"
    for name in ["baseline_utils", "run_logistic", "run_knn", "run_xgboost",
                 "compare_baselines", "fraud_amount_analysis"]:
        tree = ast.parse((root / f"{name}.py").read_text())
        assert not any(isinstance(n, ast.Name) and n.id in {"load_test", "load_raw", "load_split"}
                       for n in ast.walk(tree))
        assert not any(isinstance(n, ast.ImportFrom) and any(
            a.name in {"load_test", "load_raw", "load_split"} for a in n.names)
                       for n in ast.walk(tree))


def test_tie_diagnostics():
    d = tie_diagnostics([1., 1., .5, .4], budgets=(2,))
    assert d["n_probability_one"] == d["boundary_ties_at_2"] == 2
