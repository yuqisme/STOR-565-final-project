"""
demo_2d.py  --  2D picture of how an isolation tree isolates points (owner: A).

Synthetic data: 63 "normal" points from a 2D standard normal plus one anomaly x_o.
x_i is the normal point closest to the centre.

One isolation tree: at each node pick a feature at random, cut at a uniform
random value between that feature's min and max in the node, recurse until
every point is alone (no depth limit, psi = all 64 points, as in the paper's
Figure 1). h(x) = number of cuts on x's path.

Figure 1 (demo_2d_isolation.png): the cuts on the paths of x_o and x_i in ONE
tree. The tree is chosen so that both h(x_o) and h(x_i) equal their average over
1,000 trees (rounded), i.e. a typical tree, not a lucky one.
Figure 2 (demo_2d_path_convergence.png): average h over the first t trees,
t = 1..1000, for x_o and x_i, with c(64) for reference.

Run: python src/demo_2d.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_1d import BLUE, GRID, INK, INK2, MUTED, ORANGE, c  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
DATA_SEED = 3
N_NORMAL = 63
ANOMALY = (3.4, 2.9)
N_TREES = 1000
PAD = 0.35                          # drawing margin around the data


def make_data(seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(size=(N_NORMAL, 2)), ANOMALY])
    i_o = len(X) - 1
    i_i = int(np.argmin(np.linalg.norm(X[:-1], axis=1)))
    return X, i_o, i_i


def grow(X, rng):
    """Grow one fully isolating tree. Returns (depth per point, list of nodes).

    Each node: dict(idx=points in node, region=(x0, x1, y0, y1), feat, cut).
    """
    lo, hi = X.min(0) - PAD, X.max(0) + PAD
    depth = np.zeros(len(X), dtype=int)
    nodes = []
    stack = [(np.arange(len(X)), (lo[0], hi[0], lo[1], hi[1]), 0)]
    while stack:
        idx, region, d = stack.pop()
        if len(idx) == 1:
            depth[idx[0]] = d
            continue
        feats = [f for f in (0, 1) if X[idx, f].min() < X[idx, f].max()]
        f = feats[rng.integers(len(feats))]
        cut = rng.uniform(X[idx, f].min(), X[idx, f].max())
        nodes.append({"idx": set(idx.tolist()), "region": region, "feat": f, "cut": cut})
        left, right = idx[X[idx, f] < cut], idx[X[idx, f] >= cut]
        x0, x1, y0, y1 = region
        if f == 0:
            stack += [(left, (x0, cut, y0, y1), d + 1), (right, (cut, x1, y0, y1), d + 1)]
        else:
            stack += [(left, (x0, x1, y0, cut), d + 1), (right, (x0, x1, cut, y1), d + 1)]
    return depth, nodes


def path_cuts(nodes, i):
    """Cuts on point i's path, in order from the root (largest node first)."""
    return sorted((n for n in nodes if i in n["idx"]), key=lambda n: -len(n["idx"]))


def final_cell(X, nodes, i):
    """Region of the leaf that holds point i alone."""
    x0, x1, y0, y1 = path_cuts(nodes, i)[0]["region"]
    for n in path_cuts(nodes, i):
        if n["feat"] == 0:
            x0, x1 = (x0, n["cut"]) if X[i, 0] < n["cut"] else (n["cut"], x1)
        else:
            y0, y1 = (y0, n["cut"]) if X[i, 1] < n["cut"] else (n["cut"], y1)
    return x0, x1, y0, y1


def depths_over_trees(X, n_trees=N_TREES):
    """Depth of every point in each of n_trees trees (tree t uses seed t)."""
    return np.array([grow(X, np.random.default_rng(t))[0] for t in range(n_trees)])


def typical_tree(D, i_o, i_i, X=None):
    """A tree whose depths for x_o and x_i both equal their rounded averages.

    Among those typical trees, return the one where x_i's final cell is largest
    (easiest to see in a figure). Without X, return the first typical tree.
    """
    target_o, target_i = round(D[:, i_o].mean()), round(D[:, i_i].mean())
    typical = [t for t in range(len(D)) if D[t, i_o] == target_o and D[t, i_i] == target_i]
    if not typical:
        raise RuntimeError("no typical tree found")
    if X is None:
        return typical[0]

    def area(t):
        x0, x1, y0, y1 = final_cell(X, grow(X, np.random.default_rng(t))[1], i_i)
        return (x1 - x0) * (y1 - y0)
    return max(typical, key=area)


def _style(plt):
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#c3c2b7", "axes.facecolor": "#fcfcfb",
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})


def _draw_tree_path(ax, X, i_o, i_i, nodes, i, col, label_cuts, inset=False):
    """Draw the cuts on point i's path, its final cell, and the points."""
    from matplotlib.patches import Rectangle
    cuts = path_cuts(nodes, i)
    x0, x1, y0, y1 = final_cell(X, nodes, i)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=col, alpha=0.18, lw=0))
    for k, n in enumerate(cuts, 1):
        rx0, rx1, ry0, ry1 = n["region"]
        if n["feat"] == 0:
            ax.plot([n["cut"]] * 2, [ry0, ry1], color=INK2, lw=1.1, alpha=0.85)
            lx, ly = n["cut"], ry0 + 0.12 * (ry1 - ry0)
        else:
            ax.plot([rx0, rx1], [n["cut"]] * 2, color=INK2, lw=1.1, alpha=0.85)
            lx, ly = rx0 + 0.12 * (rx1 - rx0), n["cut"]
        if k in label_cuts:
            ax.text(lx, ly, str(k), ha="center", va="center", fontsize=10, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=MUTED, lw=0.6))
    others = np.ones(len(X), bool)
    others[[i_o, i_i]] = False
    big = 70 if inset else 90
    ax.scatter(*X[others].T, s=40 if inset else 22, color=BLUE, alpha=0.55, lw=0)
    ax.scatter(*X[i_i], s=big, color=BLUE, edgecolor="white", linewidth=1.8, zorder=4)
    ax.scatter(*X[i_o], s=big, color=ORANGE, edgecolor="white", linewidth=1.8, zorder=4)
    return cuts


def plot_isolation(X, i_o, i_i, nodes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    lo, hi = X.min(0) - PAD, X.max(0) + PAD
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # left: the anomaly
    ax = axes[0]
    cuts = _draw_tree_path(ax, X, i_o, i_i, nodes, i_o, ORANGE, label_cuts=range(1, 99))
    ax.annotate("$x_o$", X[i_o], xytext=(-10, -16), textcoords="offset points", ha="right",
                color=INK, fontsize=13)
    ax.set_title(f"Anomaly $x_o$: isolated after {len(cuts)} cuts", loc="left",
                 color=INK, fontsize=14, fontweight="bold")

    # right: the central normal point
    ax = axes[1]
    cuts = _draw_tree_path(ax, X, i_o, i_i, nodes, i_i, BLUE, label_cuts=range(1, 99))
    ax.set_title(f"Normal point $x_i$: isolated after {len(cuts)} cuts", loc="left",
                 color=INK, fontsize=14, fontweight="bold")
    for ax in axes:
        ax.annotate("$x_i$", X[i_i], xytext=(-10, -16), textcoords="offset points", ha="right",
                    color=INK, fontsize=13)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xticks=[], yticks=[], aspect="equal")
    axes[1].annotate("$x_o$", X[i_o], xytext=(-10, -16), textcoords="offset points", ha="right",
                     color=INK, fontsize=13)
    fig.suptitle("Same random tree: the anomaly is cut off in a few steps, a central point needs many",
                 x=0.02, ha="left", color=INK2, fontsize=12.5, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def plot_convergence(D, i_o, i_i, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    t = np.arange(1, len(D) + 1)
    run_o = np.cumsum(D[:, i_o]) / t
    run_i = np.cumsum(D[:, i_i]) / t
    cn = c(D.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, run_i, color=BLUE, lw=2)
    ax.plot(t, run_o, color=ORANGE, lw=2)
    ax.axhline(cn, color=MUTED, lw=1.2, ls="--")
    ax.text(t[-1] * 1.08, run_i[-1], f"$x_i$ (normal)  {run_i[-1]:.1f}", va="center", color=INK)
    ax.text(t[-1] * 1.08, run_o[-1], f"$x_o$ (anomaly)  {run_o[-1]:.1f}", va="center", color=INK)
    ax.text(t[-1] * 1.08, cn, f"c(64) = {cn:.1f}", va="center", color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, len(D))
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_ylim(0, max(run_i.max(), cn) * 1.15)
    ax.set_xlabel("number of trees t (log scale)")
    ax.set_ylabel("average path length")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("Averaging over trees removes the luck: the two path lengths settle apart",
                 loc="left", color=INK, fontsize=13.5, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    X, i_o, i_i = make_data()
    D = depths_over_trees(X)
    t = typical_tree(D, i_o, i_i, X)
    _, nodes = grow(X, np.random.default_rng(t))
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_isolation(X, i_o, i_i, nodes, FIGURES_DIR / "demo_2d_isolation.png")
    plot_convergence(D, i_o, i_i, FIGURES_DIR / "demo_2d_path_convergence.png")
    n = len(X)
    for name, i in [("x_o (anomaly)", i_o), ("x_i (normal)", i_i)]:
        h = D[:, i].mean()
        print(f"{name}: mean path {h:.2f} over {len(D)} trees, s = {2 ** (-h / c(n)):.3f}")
    print(f"c({n}) = {c(n):.3f}; drawn tree = seed {t} "
          f"(h(x_o) = {D[t, i_o]}, h(x_i) = {D[t, i_i]})")


if __name__ == "__main__":
    main()
