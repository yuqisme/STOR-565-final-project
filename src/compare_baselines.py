"""Refit selected baselines and IF controls; validation comparison and failures.

Table reports seed-0 scores through results_table plus means/sample std across
seeds 0-4 for random models. Seed 0 is fixed in advance for plots, failures and
business policies; it is not a selected best seed or a five-model ensemble.
"""
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_train_val
from eval_utils import evaluate, plot_pr_curves, results_table
from baseline_utils import (load_config, make_knn, make_logistic, make_xgb,
                            prepare_dirs, save_config, SEEDS)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE = ROOT / ".baseline_cache"
SUPERVISED = {"IF default": "no", "IF tuned": "no", "kNN distance": "no",
              "Logistic": "yes", "XGBoost": "yes"}


def selected_specs():
    grid = pd.read_csv(RESULTS_DIR / "if_tuning_val.csv")
    best = grid.loc[grid.AUPRC_mean.idxmax()]
    return {"IF default": dict(n_estimators=100, max_samples=256),
            "IF tuned": dict(n_estimators=int(best.n_estimators), max_samples=int(best.psi)),
            "kNN distance": load_config("knn"),
            "Logistic": load_config("logistic"),
            "XGBoost": load_config("xgboost")}


def build_model(name, config, seed=0):
    """Shared final-fit entry point for C; call only with the frozen train split."""
    params = dict(config)
    if name.startswith("IF"):
        return IsolationForest(random_state=seed, **params)
    if name == "kNN distance":
        params.pop("drop_time")
        return make_knn(seed=seed, **params)
    if name == "Logistic":
        params["seed"] = seed
        return make_logistic(**params)
    if name == "XGBoost":
        return make_xgb(seed=seed, **params)
    raise ValueError(f"Unknown method: {name}")


def model_scores(name, model, X):
    if name.startswith("IF"):
        return -model.score_samples(X)
    if name == "kNN distance":
        return model.score_samples(X)
    return model.predict_proba(X)[:, 1]


def feature_view(X, config):
    return X.drop(columns="Time") if config.get("drop_time", False) else X


def analyze_failures(X_val, y_val, scores):
    """Aggregate only; transaction-level records and score arrays stay untracked."""
    caught = []
    for s in scores.values():
        mask = np.zeros(len(y_val), dtype=bool)
        mask[np.argsort(-s, kind="stable")[:200]] = True
        caught.append(mask)
    caught = np.column_stack(caught)
    n_caught = caught.sum(axis=1)
    fraud = y_val == 1
    groups = {"caught_all": fraud & (n_caught == len(scores)),
              "caught_some": fraud & (n_caught > 0) & (n_caught < len(scores)),
              "missed_all": fraud & (n_caught == 0)}
    pd.DataFrame([dict(group=g, fraud_count=int(mask.sum())) for g, mask in groups.items()]
                 ).to_csv(RESULTS_DIR / "failure_counts_val.csv", index=False)
    pd.DataFrame([dict(model=name, frauds_caught_top200=int((caught[:, i] & fraud).sum()),
                       frauds_missed_top200=int((~caught[:, i] & fraud).sum()))
                  for i, name in enumerate(scores)]).to_csv(
                      RESULTS_DIR / "failure_by_model_val.csv", index=False)
    features = ["Amount", "Time", "V14", "V17", "V12", "V10"]
    compare_groups = {"missed_all": groups["missed_all"],
                      "caught_any": fraud & (n_caught > 0), "normal": ~fraud}
    rows = []
    for group, mask in compare_groups.items():
        for feature in features:
            s = X_val.loc[mask, feature]
            rows.append(dict(group=group, feature=feature, n=len(s), mean=s.mean(),
                             median=s.median(), std=s.std(), q25=s.quantile(.25),
                             q75=s.quantile(.75), min=s.min(), max=s.max()))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "failure_cases_val.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, feature in zip(axes.flat, features):
        for i, (group, mask) in enumerate(compare_groups.items()):
            vals = X_val.loc[mask, feature].to_numpy()
            if len(vals):
                ax.boxplot([vals], positions=[i], widths=.55, showfliers=False)
        ax.set_xticks(range(3), ["Missed all", "Caught any", "Normal"], fontsize=8)
        ax.set_title(feature)
        if feature == "Amount":
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Amount units (symlog)")
    fig.suptitle("Validation top-200 failures, fixed seed 0; boxes show IQR")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_cases_val.png", dpi=200)
    plt.close(fig)


def data_fingerprints():
    # Deliberately enumerate only train and validation, never glob dataset files.
    return {name: hashlib.sha256((ROOT / "data/random_split" / name).read_bytes()).hexdigest()
            for name in ("train.csv", "validation.csv")}


def main():
    prepare_dirs()
    X_train, y_train, X_val, y_val = load_train_val()
    specs, scores, rows = selected_specs(), {}, []
    for name, config in specs.items():
        for seed in ([0] if name == "Logistic" else SEEDS):
            model = build_model(name, config, seed)
            with threadpool_limits(limits=4):
                model.fit(feature_view(X_train, config), y_train if SUPERVISED[name] == "yes" else None)
                s = model_scores(name, model, feature_view(X_val, config))
            rows.append(dict(model=name, supervised=SUPERVISED[name], seed=seed,
                             **evaluate(y_val, s)))
            if seed == 0:
                scores[name] = s
        print(f"Refit {name}: {config}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "baselines_compare_val_raw.csv", index=False)
    table = results_table(y_val, scores)
    table.insert(0, "supervised", pd.Series(SUPERVISED))
    table.insert(1, "representative_seed", 0)
    for metric in evaluate(y_val, next(iter(scores.values()))):
        grouped = raw.groupby("model")[metric]
        table[metric + "_mean"] = grouped.mean()
        table[metric + "_std"] = grouped.std().fillna(0)
    table.rename_axis("model").to_csv(RESULTS_DIR / "baselines_compare_val.csv")
    # Reuse the team's PR helper; restyle its supervised lines and save again.
    fig = plot_pr_curves(y_val, scores, title="Validation PR: solid unsupervised; dashed supervised")
    fig.set_size_inches(10, 5)
    for name, line in zip(scores, fig.axes[0].lines):
        line.set_linestyle("--" if SUPERVISED[name] == "yes" else "-")
    fig.axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baselines_compare_pr_val.png", dpi=200)
    plt.close(fig)
    analyze_failures(X_val, y_val, scores)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / "comparison_val.npz", **scores)
    save_config("baselines", dict(models=specs, seed=0,
                                  train_rows=len(X_train), train_frauds=int(y_train.sum()),
                                  val_rows=len(X_val), val_frauds=int(y_val.sum()),
                                  data_sha256=data_fingerprints(),
                                  python=platform.python_version(),
                                  packages={p: version(p) for p in
                                            ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib")}))
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
