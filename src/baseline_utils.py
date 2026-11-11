"""Shared estimators and output helpers for validation-only baselines.

kNN standardization uses the full training set; its reference sample never
uses labels. Reference sampling is inside the Pipeline, after preprocessing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEEDS = range(5)


class KNNDistance(BaseEstimator):
    """Distance to the kth training reference neighbour (larger = anomalous)."""

    def __init__(self, n_neighbors=10, reference_size=20000, random_state=0):
        self.n_neighbors = n_neighbors
        self.reference_size = reference_size
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        size = min(self.reference_size, len(X))
        if self.n_neighbors < 1 or self.n_neighbors > size:
            raise ValueError("n_neighbors must be between 1 and reference size")
        self.reference_indices_ = np.random.default_rng(self.random_state).choice(
            len(X), size=size, replace=False)
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="brute", n_jobs=-1
        ).fit(X[self.reference_indices_])
        return self

    def neighbor_distances(self, X):
        check_is_fitted(self, "neighbors_")
        # Bound native BLAS/OpenMP oversubscription; sklearn still uses n_jobs=-1.
        with threadpool_limits(limits=4):
            return self.neighbors_.kneighbors(check_array(X, dtype=float),
                                             return_distance=True)[0]

    def score_samples(self, X):
        return self.neighbor_distances(X)[:, -1]


def make_knn(k=10, standardized=True, reference_size=20000, seed=0):
    return Pipeline([
        ("scaler", StandardScaler() if standardized else "passthrough"),
        ("knn", KNNDistance(k, reference_size, seed)),
    ])


def make_logistic(C=1.0, class_weight=None, seed=0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=C, class_weight=class_weight,
                                        max_iter=3000, random_state=seed)),
    ])


def make_xgb(scale_pos_weight=1.0, seed=0):
    from xgboost import XGBClassifier
    # Fixed modest tree complexity; only the requested imbalance weight is tuned.
    return Pipeline([("xgb", XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_lambda=1, tree_method="hist", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=4,
    ))])


def prepare_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def save_config(name, config):
    prepare_dirs()
    (RESULTS_DIR / f"{name}_selected.json").write_text(
        json.dumps(config, indent=2) + "\n")


def load_config(name):
    return json.loads((RESULTS_DIR / f"{name}_selected.json").read_text())


def summarize(rows, group_cols):
    metrics = ["AUPRC", "recall@50", "recall@n_pos", "recall@200", "baseline_AUPRC"]
    grouped = pd.DataFrame(rows).groupby(group_cols, dropna=False, sort=False)
    return pd.concat([grouped[metrics].mean().add_suffix("_mean"),
                      grouped[metrics].std().fillna(0).add_suffix("_std")],
                     axis=1).reset_index()


def tie_diagnostics(scores, budgets=(50, 94, 200)):
    scores = np.asarray(scores)
    ordered = np.sort(scores)[::-1]
    out = {"n_probability_one": int((scores == 1).sum()),
           "n_at_max_score": int((scores == ordered[0]).sum())}
    for k in budgets:
        k = min(k, len(scores))
        out[f"boundary_ties_at_{k}"] = int((scores == ordered[k - 1]).sum())
    return out
