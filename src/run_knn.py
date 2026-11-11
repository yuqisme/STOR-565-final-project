"""Unsupervised kNN distance: train-only scaling, 20k references, seeds 0-4.

Tune k and Time inclusion using validation mean AUPRC. A single 50-neighbour
query supplies all candidate k values. Raw k=10 with Time is a scaling control.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val
from eval_utils import evaluate, plot_pr_curves
from baseline_utils import make_knn, prepare_dirs, save_config, summarize, SEEDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
KS = (1, 5, 10, 20, 50)
REFERENCE_SIZE = 20000


def main():
    prepare_dirs()
    rows, seed_zero_scores = [], {}
    for drop_time in (False, True):
        X_train, _, X_val, y_val = load_train_val(
            drop_cols=("Time",) if drop_time else ())
        for seed in SEEDS:
            model = make_knn(max(KS), reference_size=REFERENCE_SIZE, seed=seed)
            model.fit(X_train)
            transformed = model.named_steps["scaler"].transform(X_val)
            distances = model.named_steps["knn"].neighbor_distances(transformed)
            for k in KS:
                scores = distances[:, k - 1]
                rows.append(dict(drop_time=drop_time, standardized=True, k=k,
                                 reference_size=REFERENCE_SIZE, seed=seed,
                                 **evaluate(y_val, scores)))
                if seed == 0:
                    seed_zero_scores[(drop_time, k)] = scores.copy()
            print(f"standardized kNN: drop_time={drop_time}, seed={seed}", flush=True)
    # Raw vs scaled control uses identical references, Time included, k=10.
    X_train, _, X_val, y_val = load_train_val()
    for seed in SEEDS:
        model = make_knn(10, standardized=False, reference_size=REFERENCE_SIZE, seed=seed)
        scores = model.fit(X_train).score_samples(X_val)
        rows.append(dict(drop_time=False, standardized=False, k=10,
                         reference_size=REFERENCE_SIZE, seed=seed,
                         **evaluate(y_val, scores)))
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "knn_tuning_val_raw.csv", index=False)
    summary = summarize(rows, ["drop_time", "standardized", "k", "reference_size"])
    summary.to_csv(RESULTS_DIR / "knn_tuning_val.csv", index=False)
    candidates = summary[summary.standardized]
    best = candidates.loc[candidates.AUPRC_mean.idxmax()]
    config = dict(k=int(best.k), drop_time=bool(best.drop_time), standardized=True,
                  reference_size=REFERENCE_SIZE)
    save_config("knn", config)
    best.to_frame().T.to_csv(RESULTS_DIR / "knn_val.csv", index=False)
    scores = seed_zero_scores[(config["drop_time"], config["k"])]
    plot_pr_curves(y_val, {"kNN distance (unsupervised, seed 0)": scores},
                   path=FIGURES_DIR / "knn_pr_val.png")
    print("Selected kNN:\n", best.to_string())
    print("Raw vs standardized (Time included, k=10):\n",
          summary[(~summary.drop_time) & (summary.k == 10)].to_string(index=False))


if __name__ == "__main__":
    main()
