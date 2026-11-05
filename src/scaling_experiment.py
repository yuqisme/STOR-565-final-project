"""
scaling_experiment.py  --  (owner: Yuqi)

Claim to show: multiplying ONE feature by a constant
    - changes kNN-distance results (distance-based, depends on units)
    - does NOT change Isolation Forest results (splits are uniform between min and max
      of each feature, so a rescaled feature gives the same tree structure).

IMPORTANT: use RAW (unstandardized) features here. If kNN is run after StandardScaler,
the constant gets cancelled out and kNN would not change either.

Data: shared data_loader (random stratified split, Time kept).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so eval_utils imports from anywhere
from data_loader import load_train_val  # noqa: E402
from eval_utils import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]                  # repo root
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 0
K = 10
SCALE_COL_NAME = "V14"             # feature to rescale (strong fraud signal)
SCALE_COL = 0                      # its column index; load_data() sets it from the loader columns
FACTORS = [0.01, 0.1, 1, 10, 100, 1000]
KNN_TRAIN_SUBSAMPLE = 20000        # kNN is slow on big data; subsample the reference set


def load_data():
    """Raw train/val from the shared loader (never test).

    Returns X_train, X_val (float arrays), y_val; sets SCALE_COL to V14's index.
    """
    global SCALE_COL
    X_train, _, X_val, y_val = load_train_val()
    SCALE_COL = list(X_train.columns).index(SCALE_COL_NAME)
    return X_train.to_numpy(float), X_val.to_numpy(float), y_val


def synthetic_data():
    """Small synthetic stand-in with the same return format (used by the tests)."""
    rng = np.random.default_rng(SEED)
    d = 5
    X_train = rng.normal(size=(5000, d))
    X_norm = rng.normal(size=(2000, d))
    X_anom = rng.normal(loc=3.0, scale=1.0, size=(40, d))   # synthetic "frauds"
    X_val = np.vstack([X_norm, X_anom])
    y_val = np.r_[np.zeros(len(X_norm)), np.ones(len(X_anom))].astype(int)
    return X_train, X_val, y_val


def rescale(X, col, factor):
    X = np.array(X, dtype=float, copy=True)
    X[:, col] *= factor
    return X


def if_scores(X_train, X_val):
    m = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED).fit(X_train)
    return -m.score_samples(X_val)          # higher = more anomalous


def knn_scores(X_train, X_val, k=K):
    rng = np.random.default_rng(SEED)
    if len(X_train) > KNN_TRAIN_SUBSAMPLE:
        X_train = X_train[rng.choice(len(X_train), KNN_TRAIN_SUBSAMPLE, replace=False)]
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    dist, _ = nn.kneighbors(X_val)
    return dist[:, -1]                      # distance to k-th neighbour


def main():
    X_train, X_val, y_val = load_data()
    rows = []
    for f in FACTORS:
        Xtr, Xva = rescale(X_train, SCALE_COL, f), rescale(X_val, SCALE_COL, f)
        for name, fn in [("IsolationForest", if_scores), ("kNN", knn_scores)]:
            r = evaluate(y_val, fn(Xtr, Xva))
            rows.append({"factor": f, "model": name,
                         "AUPRC": r["AUPRC"], "recall@n_pos": r["recall@n_pos"]})
    df = pd.DataFrame(rows)
    table = df.pivot(index="factor", columns="model", values="AUPRC").round(4)
    print("AUPRC vs. scaling factor on feature", SCALE_COL_NAME)
    print(table)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "scaling_experiment_auprc.csv")

    # quick figure for the slides
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in table.columns:
        ax.plot(table.index, table[name], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel(f"factor multiplied onto feature {SCALE_COL_NAME}")
    ax.set_ylabel("AUPRC (validation)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "scaling_experiment.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()