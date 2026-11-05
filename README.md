# STOR# Isolation Forest for Credit Card Fraud Detection

STOR 565 group project. We study **Isolation Forest (IF)** as an unsupervised anomaly
detector on the credit card fraud dataset and compare it with three baselines:
kNN distance, PCA reconstruction error, and logistic regression (supervised reference).

Because fraud is extremely rare (about 0.17% of transactions), accuracy is misleading
(predicting "all normal" already gives about 99.8%). We evaluate with **AUPRC** and **recall@k** (k = 50, k = number of frauds, and k = 200).

## Team

| Role | Member | Main responsibility |
|---|---|---|
| A: Method lead | TODO | IF math, 2D demo, sensitivity analysis |
| B: Data lead | TODO | Data loading, split, EDA, data documentation |
| C: Evaluation & IF lead | Yuqi Peng | Evaluation function, IF runs, scaling experiment, final test run |
| D: Baselines lead | TODO | kNN, PCA, logistic regression, failure-case analysis |
| E: Integration lead | TODO | Repo, README, AI-use log, slides, report |

## Repository structure

```
.
├── README.md
├── CLAUDE.md                  # context for Claude Code (optional)
├── requirements.txt
├── data/                      # NOT committed (see "Data" below)
├── src/
│   ├── data_loader.py         # B: shared loader + fixed time-based split
│   ├── eval_utils.py          # C: shared evaluation (AUPRC, recall@k, PR curves)
│   └── ...
├── experiments/
│   ├── run_if_default.py      # C: IF default settings
│   ├── scaling_experiment.py  # C: rescaling one feature (kNN changes, IF does not)
│   ├── baselines.py           # D
│   └── final_test.py          # C: run ONCE on 11/11
├── results/                   # tables (csv)
├── figures/                   # plots used in slides and report
└── tests/
```

(Adjust this tree to match the actual files as they are added.)

## Setup

```bash
git clone <REPO_URL>
cd <REPO_NAME>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Suggested `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`.

## Data

We use the credit card fraud dataset (European cardholders, two days of transactions;
features `Time`, `V1`–`V28` (PCA-transformed), `Amount`, and label `Class`).

- Source: TODO (B: exact link)
- Access date: TODO
- License: TODO (B: confirm on the dataset page)
- Download: put the file at `data/creditcard.csv`. The data is **not** committed to git.
- Limitations: TODO (B: e.g., only two days, features anonymized by PCA, very few frauds)

## Conventions (everyone follows these)

1. **Score direction:** higher score = more anomalous. Labels: `1` = fraud, `0` = normal.
   For IF use `-model.score_samples(X)` since sklearn's value is higher = more normal.
2. **Split:** time-based train / validation / test, produced only by `src/data_loader.py`.
   Fixed random seed: TODO.
3. **Test set:** nobody touches it before the final run. Only C runs it, once, on 11/11.
   All tuning uses the validation set.
4. **Metrics:** call `eval_utils.evaluate(y_true, scores)` so all results are comparable.

## How to reproduce results

TODO: fill in as scripts are finished. Example:

```bash
python experiments/run_if_default.py        # IF default settings on validation set
python experiments/baselines.py             # kNN, PCA, logistic regression
python experiments/scaling_experiment.py    # scaling experiment
python experiments/final_test.py            # final test-set run (once only)
```

Outputs go to `results/` and `figures/`.

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
-565-final-project