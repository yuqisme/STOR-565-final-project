"""
toy_1d.py  --  the 1-2-3-4-100 toy example for the slides (owner: A).

Data: five numbers 1, 2, 3, 4, 100. One "isolation tree" keeps cutting at a
uniformly random point between the min and max of the current group until the
target point is alone. h(x) = number of cuts needed.

We compute E[h(x)] EXACTLY (recursion over contiguous groups of the sorted
points), check it by simulation, and turn it into the anomaly score
    s(x) = 2 ** (-E[h(x)] / c(n)),   c(n) = 2 H(n-1) - 2 (n-1) / n,
the normalisation from Liu, Ting & Zhou (2008), with H(i) ~ ln(i) + 0.5772.

We also report sklearn's IsolationForest score for the same data. sklearn caps
tree depth at ceil(log2(psi)) = 3 for psi = 5 and adds c(leaf size) at a capped
leaf, so its numbers differ slightly, but the ranking is the same.

Outputs: results/toy_1d.csv, figures/toy_1d.png
Run:     python src/toy_1d.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
POINTS = (1.0, 2.0, 3.0, 4.0, 100.0)
EULER = 0.5772156649

# colours: reference categorical palette, slot 1 = normal, slot 2 = anomaly
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def c(n):
    """Average path length of an unsuccessful BST search, c(n) (Liu et al. 2008)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + EULER) - 2.0 * (n - 1) / n


def expected_cuts(points=POINTS):
    """Exact E[number of cuts to isolate each point], no depth limit. Returns a list."""
    x = tuple(sorted(points))

    @lru_cache(maxsize=None)
    def e(i, a, b):                      # point i inside the group x[a..b]
        if a == b:
            return 0.0
        width = x[b] - x[a]
        total = 1.0                      # this cut
        for j in range(a, b):            # the cut lands in gap (x[j], x[j+1])
            p = (x[j + 1] - x[j]) / width
            total += p * (e(i, a, j) if i <= j else e(i, j + 1, b))
        return total

    n = len(x)
    by_sorted = [e(i, 0, n - 1) for i in range(n)]
    order = np.argsort(points, kind="stable")
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = by_sorted[rank]
    return out


def simulate_cuts(points=POINTS, n_trees=20000, seed=0):
    """Monte Carlo estimate of E[cuts] per point (same procedure, random cuts)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    totals = np.zeros(len(pts))
    for _ in range(n_trees):
        for i, xi in enumerate(pts):
            group, h = pts, 0
            while len(group) > 1:
                cut = rng.uniform(group.min(), group.max())
                group = group[group < cut] if xi < cut else group[group >= cut]
                h += 1
            totals[i] += h
    return totals / n_trees


def sklearn_scores(points=POINTS, n_trees=5000, seed=0):
    """sklearn's anomaly score s(x) (= -score_samples) with psi = all points."""
    from sklearn.ensemble import IsolationForest
    X = np.asarray(points, dtype=float).reshape(-1, 1)
    m = IsolationForest(n_estimators=n_trees, max_samples=len(X), random_state=seed).fit(X)
    return -m.score_samples(X)


def table(points=POINTS):
    n = len(points)
    exact = expected_cuts(points)
    df = pd.DataFrame({
        "x": points,
        "E_cuts_exact": exact,
        "E_cuts_simulated": simulate_cuts(points),
        "score_s": [2.0 ** (-h / c(n)) for h in exact],
        "sklearn_score": sklearn_scores(points),
    })
    return df


def first_cut_isolates_max(points=POINTS):
    """P(first cut lands between the two largest points) = gap / range."""
    x = sorted(points)
    return (x[-1] - x[-2]) / (x[-1] - x[0])


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#c3c2b7", "axes.facecolor": "#fcfcfb",
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    p = first_cut_isolates_max(df["x"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2),
                                   gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.45})

    # top: the number line and where the first cut lands
    x = df["x"].to_numpy()
    lo, hi = sorted(x)[-2], max(x)
    ax1.axvspan(lo, hi, ymin=0.2, ymax=0.62, color=ORANGE, alpha=0.12, lw=0)
    ax1.axhline(0, color="#c3c2b7", lw=1.5, zorder=1)
    is_max = x == hi
    ax1.scatter(x[~is_max], np.zeros((~is_max).sum()), s=110, color=BLUE, zorder=3,
                edgecolor="white", linewidth=2)
    ax1.scatter(x[is_max], [0], s=110, color=ORANGE, zorder=3, edgecolor="white", linewidth=2)
    others = sorted(x[~is_max])
    ax1.annotate(", ".join(f"{v:g}" for v in others), (np.mean(others), 0), xytext=(0, -26),
                 textcoords="offset points", ha="center", color=INK, fontsize=13)
    ax1.annotate(f"{hi:g}", (hi, 0), xytext=(0, -26), textcoords="offset points",
                 ha="center", color=INK, fontsize=13)
    ax1.text((lo + hi) / 2, 0.85, f"a random first cut lands here with probability "
             f"{hi - lo:g}/{hi - min(x):g} = {p:.0%}\n→ 100 is isolated after 1 cut",
             ha="center", va="bottom", color=INK, fontsize=12.5)
    ax1.set(xlim=(-3, 104), ylim=(-1.1, 1.9), yticks=[])
    ax1.set_xticks([])
    for s in ("left", "right", "top", "bottom"):
        ax1.spines[s].set_visible(False)
    ax1.set_title("A random cut almost always separates 100 first",
                  loc="left", color=INK, fontsize=14, fontweight="bold")

    # bottom: expected number of cuts per point (exact)
    labels = [f"x = {v:g}" for v in x]
    colors = [ORANGE if v == hi else BLUE for v in x]
    ypos = np.arange(len(x))[::-1]
    ax2.barh(ypos, df["E_cuts_exact"], color=colors, height=0.55)
    for y, h, s in zip(ypos, df["E_cuts_exact"], df["score_s"]):
        ax2.text(h + 0.04, y, f"{h:.2f} cuts   s = {s:.2f}", va="center", color=INK, fontsize=12.5)
    ax2.set_yticks(ypos, labels, color=INK)
    ax2.set_xlim(0, max(df["E_cuts_exact"]) * 1.55)
    ax2.set_xlabel("expected number of cuts to isolate the point, E[h(x)]  (exact)")
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for s in ("right", "top"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"Fewer cuts → higher anomaly score   s(x) = 2^(−E[h(x)] / c(n)),  c(5) = {c(5):.2f}",
                  loc="left", color=INK, fontsize=14, fontweight="bold")

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    df = table()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    df.round(4).to_csv(RESULTS_DIR / "toy_1d.csv", index=False)
    plot(df, FIGURES_DIR / "toy_1d.png")
    print(f"c(5) = {c(5):.4f};  P(first cut isolates 100) = {first_cut_isolates_max():.4f}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
