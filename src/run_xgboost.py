"""Supervised XGBoost, validation-only imbalance-weight tuning, seeds 0-4."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val
from eval_utils import evaluate, plot_pr_curves
from baseline_utils import make_xgb, prepare_dirs, save_config, summarize, SEEDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def main():
    prepare_dirs()
    X_train, y_train, X_val, y_val = load_train_val()
    ratio = float((y_train == 0).sum() / y_train.sum())
    rows, seed_zero_scores = [], {}
    for weight in (1.0, ratio):
        for seed in SEEDS:
            model = make_xgb(weight, seed).fit(X_train, y_train)
            scores = model.predict_proba(X_val)[:, 1]
            rows.append(dict(scale_pos_weight=weight, seed=seed,
                             **evaluate(y_val, scores)))
            if seed == 0:
                seed_zero_scores[weight] = scores
        print(f"XGBoost scale_pos_weight={weight:.6f} complete", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "xgboost_tuning_val_raw.csv", index=False)
    summary = summarize(rows, ["scale_pos_weight"])
    summary.to_csv(RESULTS_DIR / "xgboost_tuning_val.csv", index=False)
    best = summary.loc[summary.AUPRC_mean.idxmax()]
    weight = float(best.scale_pos_weight)
    save_config("xgboost", dict(scale_pos_weight=weight))
    best.to_frame().T.to_csv(RESULTS_DIR / "xgboost_val.csv", index=False)
    plot_pr_curves(y_val, {"XGBoost (supervised, seed 0)": seed_zero_scores[weight]},
                   path=FIGURES_DIR / "xgboost_pr_val.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
