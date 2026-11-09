"""Tests for the teaching demos src/toy_1d.py and src/demo_2d.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import demo_2d  # noqa: E402
import toy_1d  # noqa: E402


def test_c_matches_paper_values():
    assert toy_1d.c(1) == 0 and toy_1d.c(2) == 1
    assert toy_1d.c(5) == pytest.approx(2 * (np.log(4) + 0.5772156649) - 8 / 5)


def test_exact_cuts_small_cases():
    assert toy_1d.expected_cuts((0.0, 1.0)) == [1.0, 1.0]          # one cut splits two points
    a, b, cc = toy_1d.expected_cuts((0.0, 1.0, 2.0))               # equal gaps
    assert a == pytest.approx(1.5) and cc == pytest.approx(1.5) and b == pytest.approx(2.0)


def test_exact_matches_simulation_and_order_is_kept():
    pts = (4.0, 100.0, 1.0, 3.0, 2.0)                              # unsorted on purpose
    exact = np.array(toy_1d.expected_cuts(pts))
    sim = toy_1d.simulate_cuts(pts, n_trees=4000, seed=1)
    assert np.allclose(exact, sim, atol=0.08)
    assert np.argmin(exact) == 1                                    # 100 is easiest to isolate


def test_first_cut_probability():
    assert toy_1d.first_cut_isolates_max() == pytest.approx(96 / 99)


def test_sklearn_ranks_points_like_the_exact_score():
    exact = np.array(toy_1d.expected_cuts())
    sk = toy_1d.sklearn_scores(n_trees=2000)
    assert list(np.argsort(exact)) == list(np.argsort(-sk))        # fewer cuts = higher score


def test_tree_isolates_every_point_and_depth_equals_path():
    X, i_o, i_i = demo_2d.make_data()
    depth, nodes = demo_2d.grow(X, np.random.default_rng(0))
    assert len(nodes) == len(X) - 1                                 # a full binary tree
    for i in (i_o, i_i, 0, 5):
        assert depth[i] == len(demo_2d.path_cuts(nodes, i))
        x0, x1, y0, y1 = demo_2d.final_cell(X, nodes, i)
        inside = (X[:, 0] >= x0) & (X[:, 0] <= x1) & (X[:, 1] >= y0) & (X[:, 1] <= y1)
        assert inside.sum() == 1 and inside[i]                      # alone in its cell


def test_typical_tree_has_average_depths():
    X, i_o, i_i = demo_2d.make_data()
    D = demo_2d.depths_over_trees(X, n_trees=200)
    t = demo_2d.typical_tree(D, i_o, i_i, X)
    assert D[t, i_o] == round(D[:, i_o].mean()) and D[t, i_i] == round(D[:, i_i].mean())
    assert D[:, i_o].mean() < D[:, i_i].mean()                      # anomaly is isolated sooner
