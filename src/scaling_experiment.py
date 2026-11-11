"""
scaling_experiment.py  --  (owner: Yuqi)

Claim to show: multiplying ONE feature (V14) by a constant
    - changes kNN-distance results (distance-based, depends on units)
    - does NOT change Isolation Forest results (splits are uniform between min and max
      of each feature, so a rescaled feature gives the same tree structure).

IMPORTANT: use RAW (unstandardized) features here. If kNN is run after StandardScaler,
the constant gets cancelled out and kNN would not change either.

Real data: train/validation from the shared data_loader (never the test set); Time kept.
Settings match the rest of the repo: IF default (100 trees, psi=256) and IF tuned
(200 trees, psi=4096) from run_if.CONFIGS; kNN = baseline_utils.make_knn without
standardization (D's raw control: k=10, 20,000 label-free references). Seeds 0-4.

Outputs: results/scaling_experiment_auprc.csv, figures/scaling_experiment.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so sibling modules import from anywhere
from baseline_utils import SEEDS, make_knn  # noqa: E402
from data_loader import load_train_val  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from run_if import CONFIGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]                  # repo root
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 0
K = 10
KNN_REFERENCE_SIZE = 20000         # label-free reference sample, as in D's kNN
SCALE_COL_NAME = "V14"             # feature to rescale (strong fraud signal)
SCALE_COL = 0                      # its column index; load_data() sets it from the loader columns
FACTORS = [0.01, 0.1, 1, 10, 100, 1000]
IF_MODELS = {"IF default": CONFIGS["default"], "IF tuned": CONFIGS["t200_psi4096"]}


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


def if_scores(X_train, X_val, params=None, seed=SEED):
    """IsolationForest scores, higher = more anomalous (params default: IF default)."""
    params = CONFIGS["default"] if params is None else params
    m = IsolationForest(random_state=seed, **params).fit(X_train)
    return -m.score_samples(X_val)


def knn_scores(X_train, X_val, k=K, seed=SEED, reference_size=KNN_REFERENCE_SIZE):
    """Raw (unstandardized) distance to the k-th reference neighbour, higher = more anomalous."""
    model = make_knn(k, standardized=False, reference_size=reference_size, seed=seed)
    return model.fit(X_train).score_samples(X_val)


def run_experiment(X_train, X_val, y_val, col, factors=FACTORS, seeds=SEEDS, models=None):
    """Rescale column `col` by each factor and score with every model over all seeds.

    Returns a DataFrame with one row per (factor, model): mean/std over seeds of AUPRC
    and recall@n_pos, plus max_abs_score_change (largest change of any validation
    score versus factor 1, over seeds): about 0 means the scores do not depend on the scale.
    """
    if models is None:
        models = {name: (lambda a, b, sd, p=p: if_scores(a, b, p, sd)) for name, p in IF_MODELS.items()}
        models["kNN"] = lambda a, b, sd: knn_scores(a, b, seed=sd)
    seeds = list(seeds)
    scores = {}
    for f in sorted(set(factors) | {1}):                       # factor 1 is the reference
        Xtr, Xva = rescale(X_train, col, f), rescale(X_val, col, f)
        for name, fn in models.items():
            for sd in seeds:
                scores[(f, name, sd)] = fn(Xtr, Xva, sd)
    rows = []
    for f in factors:
        for name in models:
            res = [evaluate(y_val, scores[(f, name, sd)]) for sd in seeds]
            auprc = np.array([r["AUPRC"] for r in res])
            rec = np.array([r["recall@n_pos"] for r in res])
            change = max(np.abs(scores[(f, name, sd)] - scores[(1, name, sd)]).max() for sd in seeds)
            rows.append({"factor": f, "model": name,
                         "AUPRC_mean": auprc.mean(), "AUPRC_std": auprc.std(ddof=1) if len(seeds) > 1 else 0.0,
                         "recall@n_pos_mean": rec.mean(), "recall@n_pos_std": rec.std(ddof=1) if len(seeds) > 1 else 0.0,
                         "max_abs_score_change": change})
    return pd.DataFrame(rows)


def plot_results(table, path):
    """AUPRC mean +/- std over seeds vs. the factor (log x), one line per model."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, d in table.groupby("model", sort=False):
        ax.errorbar(d["factor"], d["AUPRC_mean"], yerr=d["AUPRC_std"], marker="o",
                    capsize=3, label=name)
    ax.set_xscale("log")
    ax.set(xlabel=f"factor multiplied onto feature {SCALE_COL_NAME}",
           ylabel="AUPRC (mean ± std, 5 seeds)",
           title="Rescaling one feature: IF vs kNN (validation, raw features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    X_train, X_val, y_val = load_data()
    table = run_experiment(X_train, X_val, y_val, SCALE_COL, factors=FACTORS, seeds=SEEDS)
    print("AUPRC (mean over seeds) vs. scaling factor on feature", SCALE_COL_NAME)
    print(table.pivot(index="factor", columns="model", values="AUPRC_mean").round(4))
    print("\nMax |score change| vs. factor 1:")
    print(table.pivot(index="factor", columns="model", values="max_abs_score_change").to_string())
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "scaling_experiment_auprc.csv", index=False)
    plot_results(table, FIGURES_DIR / "scaling_experiment.png")


if __name__ == "__main__":
    main()
