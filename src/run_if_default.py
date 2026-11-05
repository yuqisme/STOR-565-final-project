"""
run_if_default.py  --  Isolation Forest with sklearn defaults, validation set only.

Fits IsolationForest on the train file for 5 seeds, scores the validation file with -score_samples (higher = more
anomalous), and reports mean +/- std of AUPRC and recall@k.

NEVER reads the test file. Replace load_train_val() with the shared loader
once teammate B provides it.

Outputs: results/if_default_val.csv, figures/if_default_pr_val.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import evaluate, plot_pr_curves  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEEDS = range(5)
LABEL = "Class"
DROP_COLS = ["Time"]    # time index: not a feature under a time-based split


def load_train_val(data_dir=DATA_DIR):
    """Read the pre-split train/validation CSVs (no test).

    Returns X_train (DataFrame), X_val (DataFrame), y_val (int ndarray, 1=fraud).
    Features are raw (unstandardized); LABEL and DROP_COLS are removed.
    """
    train = pd.read_csv(Path(data_dir) / "train.csv")
    val = pd.read_csv(Path(data_dir) / "validation.csv")
    feats = [c for c in train.columns if c != LABEL and c not in DROP_COLS]
    return train[feats], val[feats], val[LABEL].to_numpy().astype(int)


def run(X_train, X_val, y_val, seeds=SEEDS):
    """Fit IF per seed; return (per-seed metrics DataFrame, {seed: scores})."""
    rows, scores = [], {}
    for sd in seeds:
        m = IsolationForest(n_estimators=100, max_samples=10000, random_state=sd)
        m.fit(X_train)
        scores[sd] = -m.score_samples(X_val)          # higher = more anomalous
        rows.append(evaluate(y_val, scores[sd]))
    return pd.DataFrame(rows, index=list(seeds)), scores


def summarize(per_seed):
    """DataFrame with columns metric, mean, std (sample std, ddof=1)."""
    out = pd.DataFrame({"mean": per_seed.mean(), "std": per_seed.std()})
    return out.rename_axis("metric").reset_index()


def main():
    X_train, X_val, y_val = load_train_val()
    print(f"train {X_train.shape}, val {X_val.shape}, val frauds {y_val.sum()} "
          f"({y_val.mean():.4%})")
    per_seed, scores = run(X_train, X_val, y_val)
    summary = summarize(per_seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "if_default_val.csv", index=False)
    plot_pr_curves(y_val, {f"IF seed {sd}": s for sd, s in scores.items()},
                   path=FIGURES_DIR / "if_default_pr_val.png",
                   title="Isolation Forest (defaults), validation")
    for _, r in summary.iterrows():
        print(f"{r['metric']:>15}: {r['mean']:.4f} +/- {r['std']:.4f}")


if __name__ == "__main__":
    main()
