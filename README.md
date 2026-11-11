# STOR 565 # Isolation Forest for Credit Card Fraud Detection

STOR 565 group project: We study **Isolation Forest (IF)** as an unsupervised anomaly
detector on the credit card fraud dataset and compare it with kNN distance
(unsupervised), logistic regression and XGBoost (supervised references).
PCA reconstruction was dropped because V1–V28 are already PCA-transformed.

Because fraud is extremely rare (about 0.17% of transactions), accuracy is misleading
(predicting "all normal" already gives about 99.8%). We evaluate with **AUPRC** and **recall@k** (k = 50, k = number of frauds, and k = 200).

## Team

| Role | Member | Main responsibility |
|---|---|---|
| A: Method lead | TODO | IF math, 2D demo, sensitivity analysis |
| B: Data lead | Thu Nguyen | Data loading, split, EDA, data documentation |
| C: Evaluation & IF lead | Yuqi Peng | Evaluation function, IF runs, scaling experiment, final test run |
| D: Baselines lead | Zihan Zhao | kNN distance, logistic regression, XGBoost, failure and fraud-amount analysis |
| E: Integration lead | TODO | Repo, README, AI-use log, slides, report |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── AI_USE_LOG.md                # E: running AI-use log (team appendix)
├── AI_USE_NOTES_D.md            # D: AI-use notes for the baselines
├── BASELINES_NOTES.md           # D: baseline findings (validation)
├── BASELINES_HANDOFF.md         # D: frozen configs and policies for the final test run
├── src/
│   ├── Data565.ipynb            # B: data preparation, stratified random split, EDA
│   ├── make_random_split.py     # Kaggle download -> dedup -> random 60/20/20 split
│   ├── data_loader.py           # Shared data loader (load_test is for final_test.py only)
│   ├── check_split.py           # E: sanity check + fingerprints of the split files
│   ├── eval_utils.py            # C: evaluation metrics (AUPRC, recall@k, PR curves)
│   ├── run_if.py                # C: IF default vs. 200 trees / psi=10000
│   ├── tune_if.py               # C: IF grid over psi x n_estimators (validation only)
│   ├── scaling_experiment.py    # C: scaling experiment (IF vs kNN)
│   ├── baseline_utils.py        # D: shared kNN / logistic / XGBoost estimators, helpers
│   ├── run_knn.py               # D: kNN distance tuning (k, Time)
│   ├── run_logistic.py          # D: logistic regression tuning (C, class weight)
│   ├── run_xgboost.py           # D: XGBoost tuning (scale_pos_weight)
│   ├── compare_baselines.py     # D: refit selected models, comparison table, failure cases
│   ├── fraud_amount_analysis.py # D: review budgets and cost policies (raw Amount)
│   ├── toy_1d.py                # A: 1-2-3-4-100 toy example (exact E[h], score)
│   └── demo_2d.py               # A: 2D isolation-tree demo figures
├── results/                     # tables (csv) and selected-config JSON files
├── figures/                     # figures for slides and report
├── tests/                       # pytest unit tests (synthetic data, no real data needed)
└── test.py                      # quick row/fraud counts of the three split files
```

## Setup

```bash
git clone https://github.com/yuqisme/STOR-565-final-project.git
cd STOR-565-final-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q              # unit tests, no real data needed
```

## Data

We use the Credit Card Fraud Detection dataset provided by the Machine Learning Group at Université Libre de Bruxelles (ULB).

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Access date:** November 5, 2026
- **License:** Open Database License (ODbL) — confirm against the Kaggle dataset page.
- **Dataset size:** 284,807 transactions, including 492 fraudulent transactions (0.1727%).
- **Features:** `Time`, `V1`–`V28` (PCA-transformed and anonymized), and `Amount`.
- **Target:** `Class` (0 = legitimate, 1 = fraud).
- **Split:** Deduplicated stratified random 60/20/20 split with random seed 0. Exact duplicate rows are removed before splitting.
- **Limitations:** The dataset covers only two days in September 2013. PCA anonymization limits feature interpretability, and the small number of fraudulent transactions makes evaluation challenging.
- **Data preparation:** The dataset is downloaded automatically from Kaggle.
Run:
```bash
python src/make_random_split.py
```

The dataset is not committed to GitHub. It can be downloaded from Kaggle or accessed through the shared Google Drive folder.
If you use the Drive files, put `train.csv`, `validation.csv` and `test.csv` in `data/random_split/`.

Either way, check the split:

```bash
python src/check_split.py
```

It should report 170,235 / 56,745 / 56,746 rows (284 / 94 / 95 frauds). The Drive files
should also show "matches shared split" for all three SHA-256 fingerprints. Files
regenerated with `make_random_split.py` have the same sizes, but their train and
validation fingerprints currently differ from the Drive files' (the script flags
this), so fingerprints only identify the Drive copy.

## Conventions (everyone follows these)

1. **Score direction:** higher score = more anomalous. Labels: `1` = fraud, `0` = normal.
   For IF use `-model.score_samples(X)` since sklearn's value is higher = more normal.
2. **Split**: Deduplicated stratified random 60/20/20 split with seed 0, generated by `src/make_random_split.py`.
   All experiments use the same fixed split.
3. **Test set:** nobody touches it before the final run. Only C runs it, once, on 11/11.
   All tuning uses the validation set.
4. **Metrics:** call `eval_utils.evaluate(y_true, scores)` so all results are comparable.

## How to reproduce results

Run from the repository root. Verified end to end on 2026-11-09: the scripts below
reproduce the committed `results/*.csv` exactly.

```bash
python src/make_random_split.py   # one-time (or use the Drive files): data/random_split/
python src/check_split.py         # confirm the split
python src/run_if.py              # IF default vs. t200_psi10000, validation, ~20 s
python src/scaling_experiment.py  # scaling experiment, validation, ~20 s
python src/tune_if.py             # IF grid psi x n_estimators, validation, ~2.5 min
python src/run_knn.py             # D: kNN tuning, writes results/knn_selected.json
python src/run_logistic.py        # D: logistic tuning, writes results/logistic_selected.json
python src/run_xgboost.py         # D: XGBoost tuning, writes results/xgboost_selected.json
python src/compare_baselines.py   # D: needs the three selected-config files and tune_if.py output
python src/fraud_amount_analysis.py  # D: run after compare_baselines.py (uses its score cache)
python src/toy_1d.py              # slides: 1-2-3-4-100 toy example, a few seconds
python src/demo_2d.py             # slides: 2D isolation demo figures, a few seconds
```

Still to add (TODO):

```bash
python src/final_test.py          # C: final test-set run, once only
```

Outputs go to `results/` and `figures/`. The selected configuration JSON files are
the source for comparison refits. kNN uses a label-free, seeded 20,000-row training
reference sample; its scaler fits the entire training split, before sampling.
Random models report means and sample standard deviations over seeds 0–4;
logistic uses deterministic LBFGS with seed 0. Comparison columns without a suffix,
PR plots, failures and monetary policies use the predeclared seed 0, while
`*_mean`/`*_std` columns summarize independent seed runs, not an ensemble.

See `BASELINES_NOTES.md` for findings and `BASELINES_HANDOFF.md` for configurations,
score direction, raw-Amount accounting and frozen validation policies for C's final
test evaluation. Cost curves are validation-optimized and therefore optimistic.
The local `.baseline_cache/` is ignored; rerun comparison to regenerate it.

## Validation results so far

Mean ± sample std over seeds 0-4 (logistic is deterministic). Validation has 94 frauds,
random AUPRC = 0.0017. Source: `results/baselines_compare_val.csv`; see
`BASELINES_NOTES.md` for settings and caveats. "IF tuned" is the best mean AUPRC of
`results/if_tuning_val.csv` (200 trees, psi = 4096); AUPRC is flat within noise for
psi from about 2048 to 50000, so this choice is not sharp.

| Method | Uses labels | AUPRC | recall@50 | recall@n_pos | recall@200 |
|---|---|---|---|---|---|
| IF default (100 trees, psi=256) | no | 0.159 ± 0.018 | 0.189 | 0.279 | 0.409 |
| IF tuned (200 trees, psi=4096) | no | 0.235 ± 0.034 | 0.202 | 0.302 | 0.462 |
| kNN distance | no | 0.173 ± 0.002 | 0.192 | 0.266 | 0.413 |
| Logistic regression | yes | 0.743 | 0.479 | 0.766 | 0.840 |
| XGBoost | yes | 0.834 ± 0.001 | 0.528 | 0.821 | 0.843 |

## Timeline

| Date | Milestone |
|---|---|
| 11/6 | Fixed split and evaluation function merged |
| 11/9 | First results table (IF + 3 baselines) and 2D demo figure |
| 11/10 | Tuning and scaling experiment done |
| 11/11 | Final test run, results frozen |
| 11/14 | Slide draft submitted (due 11/16) |
| 11/23 | Presentation |
| 12/5 | Report, code, README, and AI-use appendix due (11:59 p.m.) |

## AI use

We keep a running log of what was delegated to AI, what we kept, and what we checked
(see `AI_USE_LOG.md`, maintained by E). Code written with AI assistance is read and
cross-checked by a teammate.

## References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*.
