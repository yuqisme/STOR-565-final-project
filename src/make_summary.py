"""
make_summary.py  --  one table with validation and test results side by side.

Reads only committed aggregate results (no dataset files, no test data):
    results/baselines_compare_val.csv, results/fraud_cost_val.csv,
    results/final_test_metrics.csv, results/final_test_policies.csv
Writes results/summary_val_test.csv, figures/summary_val_test.png, and prints a markdown
version of the table for the README.
Run after final_test.py:  python src/make_summary.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
# Chart colors (reference palette, light mode): validation = blue, test = orange.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_VAL, C_TEST = "#2a78d6", "#eb6834"
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


def plot_summary(s, path, review_cost=REVIEW_COST):
    """Grouped horizontal bars, validation vs. test, one panel per measure; one row per model.

    Whiskers on AUPRC are the std over seeds 0-4 (none for deterministic logistic).
    Exact numbers are in summary_val_test.csv (the table view of this figure).
    """
    panels = [("AUPRC", "val_AUPRC_mean", "test_AUPRC_mean", "AUPRC_std", 1.0, 1.0),
              ("Recall at k = number of frauds", "val_recall@n_pos_mean", "test_recall@n_pos_mean", None, 1.0, 1.0),
              ("Recall at k = 200", "val_recall@200_mean", "test_recall@200_mean", None, 1.0, 1.0),
              (f"Cost saving at c = {review_cost} (% vs. no review)",
               f"val_c{review_cost}_saving_pct", f"test_c{review_cost}_saving_pct", None, 100.0, 100.0)]
    names = [f"{m}\n{'uses labels' if sup == 'yes' else 'unsupervised'}"
             for m, sup in zip(s["model"], s["supervised"])]
    n = len(s)
    h, off = 0.3, 0.16                              # thin bars, ~2px gap between the pair
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=True, facecolor=SURFACE)
    for ax, (title, vcol, tcol, std, xmax, _) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        y = range(n)
        for dy, col, color in ((-off, vcol, C_VAL), (off, tcol, C_TEST)):
            ax.barh([i + dy for i in y], s[col], height=h, color=color, zorder=3)
            if std:                                 # seed std whiskers on AUPRC only
                sd = s[col.replace("_mean", "_std")]
                keep = sd > 0                       # deterministic models have no whisker
                ax.errorbar(s[col][keep], [i + dy for i in y if keep.iloc[i]], xerr=sd[keep],
                            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(n - 0.5, -0.5)                  # first model at the top
        ax.axhline(2.5, color=AXIS, lw=1, zorder=1)  # unsupervised | supervised
        ax.set_yticks(list(y), names)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="x", color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    fig.suptitle("Validation vs. test: Isolation Forest, kNN and supervised references",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.925, "Mean over seeds 0-4 (logistic is deterministic). Whiskers on AUPRC = std over seeds. "
             "Test is the single final run; validation has 94 frauds, test 95.",
             fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_VAL), plt.Rectangle((0, 0), 1, 1, color=C_TEST)]
    fig.legend(handles, ["Validation", "Test"], loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.99, 0.975), fontsize=10, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    summary = build_summary(pd.read_csv(RESULTS_DIR / "baselines_compare_val.csv"),
                            pd.read_csv(RESULTS_DIR / "final_test_metrics.csv"),
                            pd.read_csv(RESULTS_DIR / "fraud_cost_val.csv"),
                            pd.read_csv(RESULTS_DIR / "final_test_policies.csv"))
    summary.to_csv(RESULTS_DIR / "summary_val_test.csv", index=False)
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_summary(summary, FIGURES_DIR / "summary_val_test.png")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
