"""Tune standardized logistic regression on validation only; never open test data."""
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val
from eval_utils import evaluate, plot_pr_curves
from baseline_utils import make_logistic, prepare_dirs, save_config, tie_diagnostics

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def main():
    prepare_dirs()
    X_train, y_train, X_val, y_val = load_train_val()
    rows, configs, predictions = [], [], []
    for C in (0.001, 0.01, 0.1, 1, 10):
        for weight in (None, "balanced"):
            config = dict(C=C, class_weight=weight, seed=0)
            with threadpool_limits(limits=4):
                model = make_logistic(**config).fit(X_train, y_train)
                scores = model.predict_proba(X_val)[:, 1]
            rows.append({**config, **evaluate(y_val, scores), **tie_diagnostics(scores)})
            configs.append(config)
            predictions.append(scores)
    grid = pd.DataFrame(rows)
    best = int(grid["AUPRC"].idxmax())  # deterministic first-config tie break
    grid.to_csv(RESULTS_DIR / "logistic_tuning_val.csv", index=False)
    grid.iloc[[best]].to_csv(RESULTS_DIR / "logistic_val.csv", index=False)
    save_config("logistic", configs[best])
    plot_pr_curves(y_val, {"Logistic (supervised, seed 0)": predictions[best]},
                   path=FIGURES_DIR / "logistic_pr_val.png")
    print(grid.iloc[[best]].to_string(index=False))
    print("Probability saturation can tie scores; recall@k uses stable row-order ties.")


if __name__ == "__main__":
    main()
