"""
make_summary.py  --  one table with validation and test results side by side.

Reads only committed aggregate results (no dataset files, no test data):
    results/baselines_compare_val.csv, results/fraud_cost_val.csv,
    results/final_test_metrics.csv, results/final_test_policies.csv
Writes results/summary_val_test.csv and prints a markdown version for the README.
Run after final_test.py:  python src/make_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
METRICS = ["AUPRC", "recall@50", "recall@n_pos", "recall@200"]
REVIEW_COST = 5     # primary review cost (team decision)


def build_summary(val, test, cost_val, cost_test, review_cost=REVIEW_COST):
    """Wide table, one row per model: val_*/test_* mean and std of each metric.

    cost_*: cost tables with columns model, review_cost and the % saving versus no review
    (`reduction_vs_no_model_pct` on validation, `reduction_vs_no_review_pct` on test).
    """
    out = val[["model", "supervised"]].copy()
    for prefix, df in (("val", val), ("test", test)):
        d = df.set_index("model").loc[out["model"]].reset_index()
        for m in METRICS:
            out[f"{prefix}_{m}_mean"] = d[f"{m}_mean"].to_numpy()
            out[f"{prefix}_{m}_std"] = d[f"{m}_std"].to_numpy()
        out[f"{prefix}_random_AUPRC"] = d["baseline_AUPRC_mean"].to_numpy()
    for prefix, df, col in (("val", cost_val, "reduction_vs_no_model_pct"),
                            ("test", cost_test, "reduction_vs_no_review_pct")):
        s = df[df["review_cost"] == review_cost].set_index("model")[col]
        out[f"{prefix}_c{review_cost}_saving_pct"] = s.loc[out["model"]].to_numpy()
    return out


def to_markdown(s, review_cost=REVIEW_COST):
    """README table: AUPRC (mean ± std unless deterministic), recall and cost saving, val | test."""
    def auprc(r, p):
        sd = r[f"{p}_AUPRC_std"]
        return f"{r[f'{p}_AUPRC_mean']:.3f}" + (f" ± {sd:.3f}" if sd > 0 else "")
    head = ["Method", "Labels", "AUPRC val", "AUPRC test", "R@n_pos val", "R@n_pos test",
            "R@200 val", "R@200 test", f"c={review_cost} saving val", f"c={review_cost} saving test"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cells = [r["model"], r["supervised"], auprc(r, "val"), auprc(r, "test"),
                 f"{r['val_recall@n_pos_mean']:.3f}", f"{r['test_recall@n_pos_mean']:.3f}",
                 f"{r['val_recall@200_mean']:.3f}", f"{r['test_recall@200_mean']:.3f}",
                 f"{r[f'val_c{review_cost}_saving_pct']:.1f}%", f"{r[f'test_c{review_cost}_saving_pct']:.1f}%"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    summary = build_summary(pd.read_csv(RESULTS_DIR / "baselines_compare_val.csv"),
                            pd.read_csv(RESULTS_DIR / "final_test_metrics.csv"),
                            pd.read_csv(RESULTS_DIR / "fraud_cost_val.csv"),
                            pd.read_csv(RESULTS_DIR / "final_test_policies.csv"))
    summary.to_csv(RESULTS_DIR / "summary_val_test.csv", index=False)
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
