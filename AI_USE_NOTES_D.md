# AI-use notes — Zihan Zhao, role D

AI assisted with baseline implementation, validation-grid execution, comparison
and failure-case analysis, fraud-amount budgeting/cost analysis, synthetic tests,
figures, reproducibility configuration and documentation. The extension requested
by the user is XGBoost; Random Forest and a PCA reconstruction baseline were not built.

Executed checks: all new scripts ran on the supplied train/validation split;
synthetic tests passed; split sizes/positive counts, feature-label separation and
absence of exact feature-row overlap between train and validation were checked.
The real test file was never extracted or read. Only aggregate output tables,
figures, configuration metadata and code are committed; data and score caches stay local.

Zihan should personally review the implementations and these interpretation points:

- Understand kNN's full-train scaler followed by the 20,000-row label-free reference
  sample, and the distinction between unsupervised scores and supervised probabilities.
- Reconcile CSV metrics with the concise notes; seed-0 plots/business policies differ
  from five-seed mean metrics, and seed std is not a dataset confidence interval.
- Verify the descriptive missed-fraud interpretation; anonymized PCA features do
  not support causal or domain-level interpretations.
- Agree with C on assumed review costs, full prevention of flagged fraud losses,
  exact train-only refits, and frozen inclusive thresholds before the final test run.
- Confirm course AI-use disclosure requirements with E (Jingran), and incorporate
  this note into the team's AI-use log. This file records AI assistance, not a claim
  that a human teammate has already reviewed or approved the work.

The agent also diagnosed local NumPy/OpenMP dependency issues, pinned the validated
package versions, and reran the experiments. The working files were prepared from
repository commit d5db4fcf71f2514f9d8b1444afce0b6c099362fe. GitHub publication is
authorized by the user's request in this chat; no teammate messages were sent.
