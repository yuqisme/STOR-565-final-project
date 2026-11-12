# Frozen validation choices for Yuqi (role C)

Prepared for Zihan Zhao (D), November 11, 2026. Source base commit:
`d5db4fcf71f2514f9d8b1444afce0b6c099362fe`. No real test file was extracted or read.

> **Status update (November 12, 2026).** C ran the final test once on November 11 with the
> configurations and policies frozen below (primary review cost c = 5; "IF tuned" = 200 trees,
> psi = 4096), using `src/final_test.py` at commit `3599a1f`. Results are in
> `results/final_test_*.csv` and `results/summary_val_test.csv` and are final: do not rerun or
> retune. The rest of this document describes what was frozen before that run (plus C's c = 5 decision).

## Exact fits and score direction

Use `results/baselines_selected.json` for configs, Python/package versions and
SHA-256 of the train/validation files. Refit **only the same training split** when
applying these thresholds; do not merge validation into training, change scaling,
resample references or choose a different seed and then reuse these numeric cutoffs.
The helper functions below are used by the comparison itself.

| Method | Configuration | Score (higher = fraud) |
|---|---|---|
| IF default | 100 trees, max_samples=256, seed=0, raw features incl. Time | −score_samples |
| IF tuned | 200 trees, max_samples=4096, seed=0, raw features incl. Time | −score_samples |
| kNN distance | StandardScaler on full train; drop Time; k=50, Euclidean brute force, n_jobs=−1; 20,000 references sampled without replacement by NumPy default_rng(0) | kth-neighbour distance |
| Logistic | StandardScaler → LogisticRegression(C=10, class_weight=None, max_iter=3000, solver=lbfgs, random_state=0); keep Time | predict_proba[:,1] |
| XGBoost | n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.8, min_child_weight=1, reg_lambda=1, tree_method=hist, eval_metric=logloss, scale_pos_weight=1, random_state=0, n_jobs=4; raw features incl. Time | predict_proba[:,1] |

XGBoost and logistic use training labels. IF and kNN use no labels. Random-model
means/stds use seeds 0–4; all frozen business policies use seed 0, selected in advance.
Logistic LBFGS is deterministic, so its displayed std=0 does not quantify uncertainty.

A final-script implementation can use the following pattern with C's already-loaded
final-evaluation features; this handoff does not load test data:

```python
from baseline_utils import load_config
from compare_baselines import build_model, feature_view, model_scores

specs = load_config("baselines")["models"]
for name, config in specs.items():
    model = build_model(name, config, seed=0)
    uses_labels = name in {"Logistic", "XGBoost"}
    model.fit(feature_view(X_train, config), y_train if uses_labels else None)
    scores = model_scores(name, model, feature_view(X_final, config))
    # Feed scores to eval_utils.evaluate. Apply frozen policies below.
```

## Review budgets and cost policies

For a fixed review budget, sort descending with `np.argsort(-scores, kind="stable")`;
review the first k rows. Count recall uses the shared evaluator. Monetary recovery
is sum(raw Amount of flagged frauds) / sum(raw Amount of all frauds). Values are
**amount units**, with no currency assumed. Review-budget tables include
k=50,94,200,500,1000; for standard `recall@n_pos`, use the evaluation split's own
positive count, not validation's 94.

The assumed loss is missed fraud amount + c × number reviewed, with
c∈{1,5,10,20}. `results/fraud_cost_val.csv` contains **all 20 frozen policies**:

- `chosen_k`, `total_cost`: validation optimum over every rank k=0..n, including
  no review and reviewing everything; equal-cost solutions prefer smaller k.
- `cutoff_score` and tie counts: describes the kth ranked row. If a tie block is
  split, this score alone cannot reproduce that fixed-k policy.
- `inclusive_threshold`, `inclusive_threshold_k`, `inclusive_threshold_cost`:
  best attainable inclusive score threshold, searched at whole tie-block ends.
  Deploy by flagging **scores >= inclusive_threshold**. An infinite threshold
  flags nothing. Future flagged counts can differ from validation's selected k.
- Percentage reductions compare with no review (all fraud amount lost) and all
  review (c × n), respectively. Assume every flagged fraud is preventable; no
  partial recovery, capacity delay or extra false-positive cost is modeled.

At c=5, rank and inclusive-threshold policies coincide (all boundaries unique):

| Method | Validation k | Inclusive threshold | Validation cost | Reduction vs no review |
|---|---:|---:|---:|---:|
| IF default | 366 | .5897698418874665 | 8,412.33 | 37.18% |
| IF tuned | 424 | .5238910381647267 | 8,592.07 | 35.84% |
| kNN distance | 685 | 10.000537105051201 | 7,919.98 | 40.85% |
| Logistic | 103 | .04944505706690619 | 4,962.70 | 62.94% |
| XGBoost | 141 | .022968342527747154 | 4,373.87 | 67.34% |

**Decision (C, before the final run): the primary review cost is c = 5 amount units**
(about 3.5% of the average validation fraud amount, 142.5; break-even precision about 3.5%).
c = 1, 10, 20 are reported only as sensitivity checks. Choose the assumed c and freeze its policy before final evaluation. **Do not optimize
k or thresholds on test labels.** The validation curves are optimistic because the
same validation labels selected models and policies. The final test script and
one-time evaluation remain C's responsibility.

## Outputs and verification

`baselines_compare_val.csv`: bare metric columns are seed 0 (rounded by the team's
`results_table`); `_mean`/`_std` columns are full-precision per-seed aggregates.
`baselines_compare_val_raw.csv` preserves independent seed metrics. The three model
grids preserve all tested configs. Failure tables aggregate frauds caught by all,
some or none at top-200; no transaction-level data or scores are published.

All five new experiment scripts ran end to end. Synthetic tests cover score
direction, shapes, train-only scaling, sampling, handoff helpers, failure groups,
amount recall, exhaustive cost optima, no-review/all-review endpoints and score ties.
Local validation score arrays are ignored in `.baseline_cache/`; rerun comparison
before amount analysis. Standard dependency deprecation warnings do not affect
passing tests. No numerical-runtime warnings remain in the final experiment logs.
