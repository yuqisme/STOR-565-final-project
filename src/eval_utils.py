"""
eval_utils.py  --  shared evaluation code (owner: C)

CONVENTION (everyone must follow this):
    scores: higher = MORE anomalous / more likely fraud.
    y_true: 1 = fraud (anomaly), 0 = normal.

Converting each model's output to this convention:
    IsolationForest : scores = -model.score_samples(X)     (sklearn's is higher = more normal!)
    kNN distance    : scores = distance to k-th neighbour   (already higher = more anomalous)
    PCA recon error : scores = squared reconstruction error (already OK)
    Logistic reg.   : scores = model.predict_proba(X)[:, 1]  (column of class 1 = fraud)
Every function below takes scores in this convention and never negates them;
only multi_seed_if negates, because it calls IsolationForest itself.

Ties: AUPRC treats tied scores as one threshold, but recall@k / precision@k
break ties by row order (stable sort), so they depend on row order if many
scores are tied (e.g. saturated logistic probabilities).
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def _check(y_true, scores):
    """Validate inputs; return (int y_true, float scores). Raises ValueError."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or y_true.ndim != 1:
        raise ValueError("y_true and scores must be 1-D and the same length")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must be 0/1")
    y_true = y_true.astype(int)
    if y_true.sum() == 0:
        raise ValueError("no positives in y_true")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN/inf")
    return y_true, scores


def _top_k(y_true, scores, k):
    """Labels of the k highest-scored points (k clipped to n; 'n_pos' allowed)."""
    if k == "n_pos":
        k = int(y_true.sum())
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer or 'n_pos', got {k!r}")
    return y_true[np.argsort(-scores, kind="stable")[: min(int(k), len(scores))]]


def auprc(y_true, scores):
    """Area under the precision-recall curve (average precision)."""
    y_true, scores = _check(y_true, scores)
    return average_precision_score(y_true, scores)


def recall_at_k(y_true, scores, k):
    """Fraction of ALL frauds found among the k highest-scored points."""
    y_true, scores = _check(y_true, scores)
    return _top_k(y_true, scores, k).sum() / y_true.sum()


def precision_at_k(y_true, scores, k):
    """Fraction of the k highest-scored points that are really fraud."""
    y_true, scores = _check(y_true, scores)
    return _top_k(y_true, scores, k).mean()


def evaluate(y_true, scores, ks=(50, "n_pos", 200)):
    """
    Main entry point. Returns a dict, e.g.
      {'AUPRC': 0.71, 'recall@50': 0.40, 'recall@n_pos': 0.65, 'recall@200': 0.78,
       'baseline_AUPRC': 0.0017}
    k = "n_pos" means k = number of frauds in this set.
    """
    y_true, scores = _check(y_true, scores)
    n_pos = int(y_true.sum())
    out = {"AUPRC": auprc(y_true, scores)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(y_true, scores, k)
    out["baseline_AUPRC"] = n_pos / len(y_true)  # a random scorer gets ~ fraud rate
    return out


def results_table(y_true, scores_by_model, ks=(50, "n_pos", 200)):
    """scores_by_model: {'IF': scores, 'kNN': scores, ...} -> pandas DataFrame."""
    import pandas as pd
    rows = {name: evaluate(y_true, s, ks) for name, s in scores_by_model.items()}
    return pd.DataFrame(rows).T.round(4)


def plot_pr_curves(y_true, scores_by_model, path=None, title="Precision-Recall curves"):
    """Plot several models (scores: higher = more anomalous) plus the random baseline."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, _ = _check(y_true, np.zeros(len(y_true)))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores_by_model.items():
        y_true, s = _check(y_true, s)
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AUPRC={auprc(y_true, s):.3f})")
    ax.axhline(y_true.mean(), ls="--", c="gray", label=f"random ({y_true.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def multi_seed_if(X_train, X_eval, y_eval, seeds=range(5), **if_kwargs):
    """Run IsolationForest with several seeds; return mean/std of the metrics."""
    from sklearn.ensemble import IsolationForest
    rows = []
    for sd in seeds:
        m = IsolationForest(random_state=sd, **if_kwargs).fit(X_train)
        rows.append(evaluate(y_eval, -m.score_samples(X_eval)))
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.mean().round(4), df.std().round(4)