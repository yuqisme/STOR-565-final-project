"""
final_test.py  --  THE single final test-set evaluation (owner: C).

Run ONCE, only when the owner says so:   python src/final_test.py --run-final-test

Refits the five frozen configurations in results/baselines_selected.json on the TRAIN
split only (IF default, IF tuned, kNN distance, logistic regression, XGBoost), scores the
test split, and reports:
    results/final_test_metrics_raw.csv   one row per (model, seed)
    results/final_test_metrics.csv       seed-0 metrics plus mean/std over seeds 0-4
    results/final_test_budget.csv        count and amount recall at fixed review budgets
    results/final_test_policies.csv      the 20 validation-frozen cost policies applied to test;
                                         column `primary` marks the pre-declared c = 5 policies
    figures/final_test_pr.png            PR curves (seed 0; dashed = supervised)
Nothing is tuned on test: configs, seed-0 policy and thresholds come from validation
(see BASELINES_HANDOFF.md). The script refuses to run if the train/validation files differ
from the ones used to freeze the configs, or if final_test_metrics.csv already exists.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_utils import SEEDS, load_config  # noqa: E402
from compare_baselines import (SUPERVISED, build_model, data_fingerprints,  # noqa: E402
                               feature_view, model_scores)
from data_loader import load_test, load_train_val  # noqa: E402
from eval_utils import evaluate, plot_pr_curves, results_table  # noqa: E402
from fraud_amount_analysis import amount_recall  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
BUDGETS = (50, "n_pos", 200, 500, 1000)
PRIMARY_REVIEW_COST = 5     # fixed before the final run (team decision); c = 1, 10, 20 are sensitivity checks
MAIN_OUTPUT = "final_test_metrics.csv"


def budget_rows(name, y, scores, amounts, budgets=BUDGETS):
    """Count recall and raw-amount recall at fixed review budgets ('n_pos' = frauds in y)."""
    total = float((amounts * y).sum())
    rows = []
    for b in budgets:
        k = int(y.sum()) if b == "n_pos" else int(b)
        ar = amount_recall(y, scores, amounts, k)
        rows.append(dict(model=name, supervised=SUPERVISED[name], seed=0, budget=str(b), k=k,
                         count_recall=evaluate(y, scores, ks=(k,))[f"recall@{k}"],
                         amount_recall=ar, fraud_amount_caught=ar * total,
                         total_fraud_amount=total))
    return rows


def policy_row(name, y, scores, amounts, review_cost, threshold):
    """Apply a frozen inclusive-threshold policy (flag scores >= threshold) to y/scores.

    Cost = missed fraud amount + review_cost * number flagged; an infinite threshold
    flags nothing. The threshold must come from validation, never from y.
    """
    flagged = scores >= threshold
    n = len(y)
    fraud_amount = float((amounts * y).sum())
    caught = float((amounts * y)[flagged].sum())
    cost = (fraud_amount - caught) + review_cost * int(flagged.sum())
    none_cost, all_cost = fraud_amount, review_cost * n
    return dict(model=name, supervised=SUPERVISED[name], seed=0, review_cost=review_cost,
                frozen_threshold=threshold, n_flagged=int(flagged.sum()),
                frauds_flagged=int(y[flagged].sum()), fraud_amount_caught=caught,
                missed_fraud_amount=fraud_amount - caught, total_cost=cost,
                no_review_cost=none_cost, review_all_cost=all_cost,
                reduction_vs_no_review_pct=100 * (1 - cost / none_cost) if none_cost else np.nan,
                reduction_vs_review_all_pct=100 * (1 - cost / all_cost) if all_cost else np.nan)


def check_frozen_inputs(manifest):
    """Abort unless train/validation match the files the frozen configs were chosen on."""
    if manifest["data_sha256"] != data_fingerprints():
        sys.exit("Train/validation files differ from the frozen configs "
                 "(results/baselines_selected.json). Not running the final test.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-final-test", action="store_true",
                    help="required: confirms this is the one-time final evaluation")
    if not ap.parse_args(argv).run_final_test:
        sys.exit("Refusing to run: pass --run-final-test (one-time final evaluation).")
    if (RESULTS_DIR / MAIN_OUTPUT).exists():
        sys.exit(f"{MAIN_OUTPUT} already exists: the final test was already run. "
                 "Do not rerun it; delete the file only if the owner says so.")
    manifest = load_config("baselines")
    check_frozen_inputs(manifest)
    specs = manifest["models"]
    policies = pd.read_csv(RESULTS_DIR / "fraud_cost_val.csv")   # frozen on validation

    X_train, y_train, _, _ = load_train_val()
    X_test, y_test = load_test()                  # the only place the test set is loaded
    amounts = X_test["Amount"].to_numpy(float)    # raw Amount, never scaled
    print(f"train {X_train.shape}; test {X_test.shape}, frauds {y_test.sum()}")

    rows, scores = [], {}
    for name, config in specs.items():
        uses_labels = SUPERVISED[name] == "yes"
        for seed in ([0] if name == "Logistic" else SEEDS):
            model = build_model(name, config, seed)
            with threadpool_limits(limits=4):
                model.fit(feature_view(X_train, config), y_train if uses_labels else None)
                s = model_scores(name, model, feature_view(X_test, config))
            rows.append(dict(model=name, supervised=SUPERVISED[name], seed=seed,
                             **evaluate(y_test, s)))
            if seed == 0:
                scores[name] = s
        print(f"done {name}", flush=True)

    raw = pd.DataFrame(rows)
    table = results_table(y_test, scores)
    table.insert(0, "supervised", pd.Series(SUPERVISED))
    table.insert(1, "representative_seed", 0)
    for metric in evaluate(y_test, next(iter(scores.values()))):
        g = raw.groupby("model")[metric]
        table[metric + "_mean"], table[metric + "_std"] = g.mean(), g.std().fillna(0)
    budget = pd.DataFrame([r for n, s in scores.items()
                           for r in budget_rows(n, y_test, s, amounts)])
    pol = pd.DataFrame([policy_row(p.model, y_test, scores[p.model], amounts,
                                   p.review_cost, p.inclusive_threshold)
                        for p in policies.itertuples()])

    pol.insert(3, "primary", pol["review_cost"] == PRIMARY_REVIEW_COST)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    raw.to_csv(RESULTS_DIR / "final_test_metrics_raw.csv", index=False)
    budget.to_csv(RESULTS_DIR / "final_test_budget.csv", index=False)
    pol.to_csv(RESULTS_DIR / "final_test_policies.csv", index=False)
    fig = plot_pr_curves(y_test, scores, title="Test PR: solid unsupervised; dashed supervised")
    fig.set_size_inches(10, 5)
    for name, line in zip(scores, fig.axes[0].lines):
        line.set_linestyle("--" if SUPERVISED[name] == "yes" else "-")
    fig.axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_pr.png", dpi=200)
    plt.close(fig)
    table.rename_axis("model").to_csv(RESULTS_DIR / MAIN_OUTPUT)   # written last = "done" marker
    print(table.round(4).to_string())
    print(f"\nPrimary cost policy (c = {PRIMARY_REVIEW_COST}):")
    print(pol[pol.primary].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
