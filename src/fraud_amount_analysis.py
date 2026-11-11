"""Validation review-budget and cost analysis using raw Amount (amount units).

Review cost is an assumed amount-unit cost per flagged transaction. Minimize
missed fraud amount + c*k over k=0..n, using stable row-order score ties.
Also report a deployable inclusive score threshold, optimized only at tie-block
ends. This threshold can flag a different number of future rows than val k.
No probabilities or calibration are assumed for IF or kNN.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val
from eval_utils import recall_at_k
from baseline_utils import prepare_dirs, load_config
from compare_baselines import CACHE, SUPERVISED, data_fingerprints

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def ranked_amounts(y, scores, amounts):
    y, scores, amounts = np.asarray(y), np.asarray(scores, dtype=float), np.asarray(amounts, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or y.shape != amounts.shape or len(y) == 0:
        raise ValueError("inputs must be nonempty 1-D arrays with identical shape")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("binary labels and finite scores required")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("amounts must be finite and nonnegative")
    order = np.argsort(-scores, kind="stable")
    cumulative = np.r_[0., np.cumsum((amounts * y)[order])]
    return scores[order], cumulative


def amount_recall(y, scores, amounts, k):
    if isinstance(k, bool) or int(k) != k or k < 0:
        raise ValueError("k must be a nonnegative integer")
    _, caught = ranked_amounts(y, scores, amounts)
    return float(caught[min(k, len(caught) - 1)] / caught[-1]) if caught[-1] else np.nan


def optimal_cost(y, scores, amounts, review_cost):
    if not np.isfinite(review_cost) or review_cost < 0:
        raise ValueError("review_cost must be finite and nonnegative")
    ordered, caught = ranked_amounts(y, scores, amounts)
    n = len(ordered)
    costs = caught[-1] - caught + review_cost * np.arange(n + 1)
    k = int(np.argmin(costs))  # smallest k wins a cost tie; includes no review
    # Inclusive threshold policies can stop only at the end of a score tie block.
    ends = np.r_[0, np.flatnonzero(ordered[:-1] != ordered[1:]) + 1, n]
    tk = int(ends[np.argmin(costs[ends])])
    threshold = float(ordered[k - 1]) if k else np.inf
    threshold_inclusive = float(ordered[tk - 1]) if tk else np.inf
    none_cost, all_cost = float(caught[-1]), float(review_cost * n)
    return dict(review_cost=review_cost, chosen_k=k, total_cost=float(costs[k]),
                missed_fraud_amount=float(caught[-1] - caught[k]),
                fraud_amount_caught=float(caught[k]), cutoff_score=threshold,
                cutoff_ties_total=int((ordered == threshold).sum()) if k else 0,
                cutoff_ties_flagged=int((ordered[:k] == threshold).sum()) if k else 0,
                inclusive_threshold=threshold_inclusive, inclusive_threshold_k=tk,
                inclusive_threshold_cost=float(costs[tk]),
                reduction_vs_no_model_pct=100 * (1 - costs[k] / none_cost) if none_cost else np.nan,
                reduction_vs_review_all_pct=100 * (1 - costs[k] / all_cost) if all_cost else np.nan,
                no_model_cost=none_cost, review_all_cost=all_cost)


def main():
    prepare_dirs()
    _, _, X_val, y_val = load_train_val()
    manifest = load_config("baselines")
    if manifest["data_sha256"] != data_fingerprints():
        raise ValueError("Train/validation changed; rerun compare_baselines.py")
    with np.load(CACHE / "comparison_val.npz") as cache:
        scores = {name: cache[name].copy() for name in cache.files}
    amounts = X_val.Amount.to_numpy()  # raw, never scaled
    budget_rows, cost_rows = [], []
    budgets = sorted({50, int(y_val.sum()), 200, 500, 1000})
    for name, s in scores.items():
        for k in budgets:
            ar = amount_recall(y_val, s, amounts, k)
            budget_rows.append(dict(model=name, supervised=SUPERVISED[name], seed=0, k=k,
                                    count_recall=recall_at_k(y_val, s, k), amount_recall=ar,
                                    fraud_amount_caught=ar * (amounts * y_val).sum(),
                                    total_fraud_amount=(amounts * y_val).sum()))
        for c in (1, 5, 10, 20):
            cost_rows.append(dict(model=name, supervised=SUPERVISED[name], seed=0,
                                  **optimal_cost(y_val, s, amounts, c)))
    budget = pd.DataFrame(budget_rows)
    costs = pd.DataFrame(cost_rows)
    budget.to_csv(RESULTS_DIR / "fraud_amount_budget_val.csv", index=False)
    costs.to_csv(RESULTS_DIR / "fraud_cost_val.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in budget.groupby("model", sort=False):
        ax.plot(data.k, data.amount_recall, marker="o",
                ls="--" if SUPERVISED[name] == "yes" else "-", label=name)
    ax.set(xscale="log", xlabel="Transactions reviewed (k)", ylabel="Fraud amount recall",
           ylim=(0, 1.02), title="Validation fraud amount recovered, fixed seed 0")
    ax.set_xticks(budgets, [str(k) for k in budgets])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_amount_budget_val.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in costs.groupby("model", sort=False):
        ax.plot(data.review_cost, data.total_cost, marker="o", label=name,
                ls="--" if SUPERVISED[name] == "yes" else "-")
    ax.axhline(costs.no_model_cost.iloc[0], c="gray", ls=":", label="No review")
    ax.set(xlabel="Assumed review cost per flagged transaction (amount units)",
           ylabel="Missed fraud amount + review cost (amount units)",
           title="Validation-selected minimum cost (optimistic)")
    ax.set_xticks([1, 5, 10, 20])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_cost_val.png", dpi=200)
    plt.close(fig)
    print(budget[budget.k == 200].to_string(index=False))
    print(costs[costs.review_cost == 5].to_string(index=False))


if __name__ == "__main__":
    main()
