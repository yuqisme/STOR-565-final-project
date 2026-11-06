"""
tune_if.py  --  Isolation Forest hyperparameter study, VALIDATION set only.

Grid: psi (max_samples) x n_estimators, 5 seeds (0-4) each; all other
IsolationForest arguments stay at sklearn defaults. Fit on X_train (no labels),
score X_val with -score_samples (higher = more anomalous). Raw features as
returned by data_loader. NEVER reads the test file.

Outputs:
    results/if_tuning_val.csv       one row per (psi, n_estimators): mean/std over seeds
    results/if_tuning_val_raw.csv   one row per (psi, n_estimators, seed)
    figures/if_tuning_auprc.png     AUPRC mean +/- std vs psi
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val  # noqa: E402
from eval_utils import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PSIS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000, 20000, 50000]
N_ESTIMATORS = [100, 200]
SEEDS = range(5)
METRICS = ["AUPRC", "recall@50", "recall@n_pos", "recall@200"]
DEFAULT = (256, 100)    # sklearn defaults: max_samples=min(256, n), n_estimators=100


def run_grid(X_train, X_val, y_val, psis=PSIS, n_estimators=N_ESTIMATORS, seeds=SEEDS):
    """Return a DataFrame with one row per (psi, n_estimators, seed)."""
    rows = []
    for psi in psis:
        if psi > len(X_train):
            continue                                   # psi larger than the training set
        for n in n_estimators:
            for sd in seeds:
                t0 = time.perf_counter()
                m = IsolationForest(n_estimators=n, max_samples=psi, random_state=sd)
                scores = -m.fit(X_train).score_samples(X_val)   # higher = more anomalous
                secs = time.perf_counter() - t0
                r = evaluate(y_val, scores)
                rows.append({"psi": psi, "n_estimators": n, "seed": sd,
                             **{k: r[k] for k in METRICS}, "time_s": secs})
    return pd.DataFrame(rows)


def summarize(raw):
    """One row per (psi, n_estimators): <metric>_mean, <metric>_std (ddof=1), time_mean_s."""
    g = raw.groupby(["psi", "n_estimators"])
    out = pd.concat([g[METRICS].mean().add_suffix("_mean"),
                     g[METRICS].std().add_suffix("_std"),
                     g["time_s"].mean().rename("time_mean_s")], axis=1)
    cols = [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["time_mean_s"]
    return out[cols].reset_index()


def plot_auprc(summary, baseline, path):
    """AUPRC mean vs psi (log x), one line per n_estimators, error bars = std over seeds."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, d in summary.groupby("n_estimators"):
        ax.errorbar(d["psi"], d["AUPRC_mean"], yerr=d["AUPRC_std"], marker="o",
                    capsize=3, label=f"n_estimators = {n}")
    ax.axhline(baseline, ls="--", c="gray", label=f"random ({baseline:.4f})")
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["psi"].unique()))
    ax.set_xticklabels([str(p) for p in sorted(summary["psi"].unique())], rotation=45)
    ax.set(xlabel="psi (max_samples)", ylabel="AUPRC (mean ± std, 5 seeds)",
           title="Isolation Forest tuning: AUPRC vs psi (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    X_train, _, X_val, y_val = load_train_val()
    print(f"train {X_train.shape}, val {X_val.shape}, val frauds {y_val.sum()} "
          f"({y_val.mean():.4%})")
    raw = run_grid(X_train, X_val, y_val)
    summary = summarize(raw)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    raw.to_csv(RESULTS_DIR / "if_tuning_val_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "if_tuning_val.csv", index=False)
    plot_auprc(summary, y_val.mean(), FIGURES_DIR / "if_tuning_auprc.png")

    pd.set_option("display.width", 200)
    print("\nSummary, sorted by mean AUPRC:")
    print(summary.sort_values("AUPRC_mean", ascending=False).round(4).to_string(index=False))
    is_default = (summary["psi"] == DEFAULT[0]) & (summary["n_estimators"] == DEFAULT[1])
    print(f"\nsklearn default (psi={DEFAULT[0]}, n_estimators={DEFAULT[1]}):")
    print(summary[is_default].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
