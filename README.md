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
| B: Data lead | Thu Nguyen | Data loading, split, EDA, data documentation |
| C: Evaluation & IF lead | Yuqi Peng | Evaluation function, IF runs, scaling experiment, final test run |
| D: Baselines lead | TODO | kNN, PCA, logistic regression, failure-case analysis |
| E: Integration lead | TODO | Repo, README, AI-use log, slides, report |

## Repository structure

```
.
├── README.md
├── .gitignore
├── src/
│   ├── Data565.ipynb             # B: data preparation, chronological split, EDA
│   ├── data_loader.py           # Shared data loader
│   ├── make_random_split.py     # Existing random-split script
│   ├── eval_utils.py            # C: evaluation metrics
│   ├── run_if_default.py        # C: Isolation Forest experiments
│   └── scaling_experiment.py    # C: scaling experiment
├── results/                     # Experiment results
├── figures/                     # Figures for slides and report
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

We use the Credit Card Fraud Detection dataset provided by the Machine Learning Group at Université Libre de Bruxelles (ULB).

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Access date:** November 5, 2026
- **License:** Open Database License (ODbL) — confirm against the Kaggle dataset page.
- **Dataset size:** 284,807 transactions, including 492 fraudulent transactions (0.1727%).
- **Features:** `Time`, `V1`–`V28` (PCA-transformed and anonymized), and `Amount`.
- **Target:** `Class` (0 = legitimate, 1 = fraud).
- **Split:** Chronological 60% training, 20% validation, and 20% testing, with boundaries adjusted to avoid splitting identical timestamps.
- **Limitations:** The dataset covers only two days in September 2013. PCA anonymization limits feature interpretability, and the small number of fraudulent transactions makes evaluation challenging.

The dataset is not committed to GitHub. It can be downloaded from Kaggle or accessed through the shared Google Drive folder.

## Conventions (everyone follows these)

1. **Score direction:** higher score = more anomalous. Labels: `1` = fraud, `0` = normal.
   For IF use `-model.score_samples(X)` since sklearn's value is higher = more normal.
2. **Split**: chronological 60/20/20 train/validation/test, with no overlapping timestamps across splits. All experiments must use the same fixed split.
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
