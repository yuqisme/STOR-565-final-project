# Role D — validation baselines (Zihan Zhao)

**Scope/settings.** Train: 170,235 rows/284 frauds; validation: 56,745/94;
random AUPRC=.001657. No real test data were extracted/read. PCA was dropped
because V1–V28 are already PCA outputs. Metrics use `eval_utils`.
Logistic: train-only StandardScaler → LBFGS, max_iter=3000; selected C=10,
class_weight=None from 5 C values × 2 weights. kNN: train-only standardization,
20,000 label-free references; selected k=50 without Time from 5 k values × Time
inclusion. XGBoost: 300 depth-3 histogram trees, learning_rate=.05, row/column
sampling=.8; selected scale_pos_weight=1 over 598.419. IF tuning uses C's grid.

**Results.** AUPRC mean ± sample std and mean recalls over seeds 0–4; logistic
is deterministic (one fit). Seed variation is not a confidence interval.

| Method | Uses labels | AUPRC | R@50 | R@94 | R@200 |
|---|---|---|---|---|---|
| IF default (100, ψ=256) | No | .1589 ± .0176 | .1894 | .2787 | .4085 |
| IF tuned (200, ψ=4096) | No | .2346 ± .0341 | .2021 | .3021 | .4617 |
| kNN distance | No | .1733 ± .0019 | .1915 | .2660 | .4128 |
| Logistic | Yes | .7431 (deterministic) | .4787 | .7660 | .8404 |
| XGBoost | Yes | .8335 ± .0009 | .5277 | .8213 | .8426 |

**Takeaways.**

- With Time/k=10 fixed, raw kNN AUPRC=.0033±.0001 versus standardized
  .1242±.0064: scaling matters for distances.
- Tuned IF leads unsupervised methods; XGBoost leads supervised references,
  which additionally use fraud labels.
- Balancing is not automatically better (weighted XGBoost=.8034±.0062).
  Selected logistic has no saturated-one probabilities or boundary ties at
  50/94/200; balanced candidates have up to 23 saturated probabilities.

**Failures (fixed seed 0, top-200).** Among 94 frauds, 35 are caught by all five
methods, 45 by some, and 14 by none. Missed frauds' median V14/V12
(−.466/−.122) are closer to normal (.050/.141) than caught frauds
(−7.044/−5.789). Missed/caught median Amount is 85.765/9.265 and Time is
108,534.5/64,598.5. This suggests normal-looking profiles can evade detection;
these descriptive comparisons do not establish causes.

**Business (seed 0).** At 200 reviews, logistic/XGBoost both catch 84.04% of
fraud counts but recover 66.81%/72.60% of 13,390.62 fraud amount units;
kNN recovers 19.06% with 41.49% count recall. Skewed amounts make count recall
insufficient. Assuming review cost c=5 amount units, XGBoost's optimum is k=141,
threshold≥.022968342527747154, cost=4,373.87: 67.34% below no review and
98.46% below reviewing everything. Costs c={1,5,10,20} are assumptions;
validation-optimized costs are optimistic, so freeze policies before final testing.

**Limits/handoff.** Only 94 validation frauds, two historical days, validation
reuse and finite references limit generalization. Plots/failures/policies use
predeclared seed 0, not mean scores or the best seed. Calibration, bootstrap
intervals, expected-loss ranking and final testing are omitted. Exact settings,
package/data fingerprints and cutoffs: `results/*_selected.json`,
`results/fraud_cost_val.csv`, `BASELINES_HANDOFF.md`.
