"""Tests for src/scaling_experiment.py. Uses its synthetic load_data(); no dataset files."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scaling_experiment as se  # noqa: E402
from eval_utils import auprc  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return se.load_data()


def test_load_data_shapes_and_labels(data):
    X_train, X_val, y_val = data
    assert X_train.shape[1] == X_val.shape[1]
    assert len(X_val) == len(y_val)
    assert set(np.unique(y_val)) == {0, 1}


def test_rescale_only_touches_one_column_and_copies(data):
    X = data[0]
    before = X.copy()
    Y = se.rescale(X, 2, 10)
    np.testing.assert_array_equal(X, before)                    # input not mutated
    np.testing.assert_allclose(Y[:, 2], X[:, 2] * 10)
    np.testing.assert_array_equal(np.delete(Y, 2, 1), np.delete(X, 2, 1))


def test_isolation_forest_is_invariant_to_feature_scaling(data):
    X_train, X_val, _ = data
    base = se.if_scores(X_train, X_val)
    for f in se.FACTORS:
        s = se.if_scores(se.rescale(X_train, se.SCALE_COL, f),
                         se.rescale(X_val, se.SCALE_COL, f))
        np.testing.assert_allclose(s, base, atol=1e-8)


def test_knn_changes_under_feature_scaling(data):
    X_train, X_val, y_val = data
    base = se.knn_scores(X_train, X_val)
    big = se.knn_scores(se.rescale(X_train, se.SCALE_COL, 1000),
                        se.rescale(X_val, se.SCALE_COL, 1000))
    assert not np.allclose(base, big)
    assert auprc(y_val, base) != pytest.approx(auprc(y_val, big))


def test_knn_factor_one_is_identity(data):
    X_train, X_val, _ = data
    np.testing.assert_allclose(
        se.knn_scores(X_train, X_val),
        se.knn_scores(se.rescale(X_train, 0, 1), se.rescale(X_val, 0, 1)))


def test_knn_after_standardizing_cancels_the_constant(data):
    """Why the experiment must use RAW features (CLAUDE.md rule 4)."""
    X_train, X_val, _ = data

    def std(a, b):
        mu, sd = a.mean(0), a.std(0)
        return (a - mu) / sd, (b - mu) / sd

    a = se.knn_scores(*std(X_train, X_val))
    b = se.knn_scores(*std(se.rescale(X_train, 0, 1000), se.rescale(X_val, 0, 1000)))
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_scores_follow_higher_is_more_anomalous(data):
    X_train, X_val, y_val = data
    for s in (se.if_scores(X_train, X_val), se.knn_scores(X_train, X_val)):
        assert s[y_val == 1].mean() > s[y_val == 0].mean()
        assert auprc(y_val, s) > 5 * y_val.mean()               # far above random


def test_knn_subsampling_is_deterministic(data, monkeypatch):
    X_train, X_val, _ = data
    monkeypatch.setattr(se, "KNN_TRAIN_SUBSAMPLE", 500)
    np.testing.assert_array_equal(se.knn_scores(X_train, X_val),
                                  se.knn_scores(X_train, X_val))


def test_main_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(se, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(se, "FACTORS", [1, 100])
    se.main()
    assert (tmp_path / "results" / "scaling_experiment_auprc.csv").exists()
    assert (tmp_path / "figures" / "scaling_experiment.png").exists()
