"""
run_if.py  --  Isolation Forest: sklearn defaults vs. larger forests, validation only.

Configs (psi = max_samples):
    default        100 trees, psi = 256     (sklearn defaults; the control)
    t200_psi4096   200 trees, psi = 4096    (the team's "IF tuned" choice)
    t200_psi10000  200 trees, psi = 10000   (comparison)
Each is fit on the train split for 5 seeds and scored on the validation split with
-score_samples (higher = more anomalous). Reports mean +/- std of AUPRC and recall@k.

NEVER reads the test file. Data comes from the shared data_loader.

Outputs (results/, figures/):
    if_<config>_val.csv, if_<config>_pr_val.png   one per config
    if_compare_val.csv, if_compare_pr_val.png     both configs together
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val  # noqa: E402
from eval_utils import evaluate, plot_pr_curves  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEEDS = range(5)
CONFIGS = {
    "default": dict(n_estimators=100, max_samples=256),
    "t200_psi4096": dict(n_estimators=200, max_samples=4096),
    "t200_psi10000": dict(n_estimators=200, max_samples=10000),
}


def run(X_train, X_val, y_val, seeds=SEEDS, params=None):
    """Fit IF per seed; return (per-seed metrics DataFrame, {seed: scores}).

    params: IsolationForest kwargs (default: CONFIGS["default"]).
    """
    params = CONFIGS["default"] if params is None else params
    rows, scores = [], {}
    for sd in seeds:
        m = IsolationForest(random_state=sd, **params).fit(X_train)
        scores[sd] = -m.score_samples(X_val)          # higher = more anomalous
        rows.append(evaluate(y_val, scores[sd]))
    return pd.DataFrame(rows, index=list(seeds)), scores


def summarize(per_seed):
    """DataFrame with columns metric, mean, std (sample std, ddof=1)."""
    out = pd.DataFrame({"mean": per_seed.mean(), "std": per_seed.std()})
    return out.rename_axis("metric").reset_index()


def plot_compare(y_val, all_scores, summaries, path):
    """One colour per config: thin line per seed, legend shows mean AUPRC. Random baseline dashed."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, (name, scores) in enumerate(all_scores.items()):
        auprc = summaries[name].set_index("metric").loc["AUPRC"]
        for j, s in enumerate(scores.values()):
            p, r, _ = precision_recall_curve(y_val, s)
            ax.plot(r, p, color=f"C{i}", alpha=0.6, lw=1,
                    label=f"{name} (AUPRC {auprc['mean']:.3f} ± {auprc['std']:.3f})" if j == 0 else None)
    ax.axhline(y_val.mean(), ls="--", c="gray", label=f"random ({y_val.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Isolation Forest configs, validation (5 seeds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    X_train, _, X_val, y_val = load_train_val()
    print(f"train {X_train.shape}, val {X_val.shape}, val frauds {y_val.sum()} "
          f"({y_val.mean():.4%})")
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    all_scores, summaries = {}, {}
    for name, params in CONFIGS.items():
        per_seed, scores = run(X_train, X_val, y_val, params=params)
        all_scores[name], summaries[name] = scores, summarize(per_seed)
        summaries[name].to_csv(RESULTS_DIR / f"if_{name}_val.csv", index=False)
        plot_pr_curves(y_val, {f"IF seed {sd}": s for sd, s in scores.items()},
                       path=FIGURES_DIR / f"if_{name}_pr_val.png",
                       title=f"Isolation Forest ({name}), validation")
        print(f"\n{name}: {params}")
        for _, r in summaries[name].iterrows():
            print(f"{r['metric']:>15}: {r['mean']:.4f} +/- {r['std']:.4f}")
    pd.concat(summaries, names=["config"]).reset_index("config").to_csv(
        RESULTS_DIR / "if_compare_val.csv", index=False)
    plot_compare(y_val, all_scores, summaries, FIGURES_DIR / "if_compare_pr_val.png")


if __name__ == "__main__":
    main()
