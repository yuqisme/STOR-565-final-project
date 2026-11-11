# AI-use log

Running log for the report's one-page AI-use appendix (maintained by E).
For each entry: what we delegated to AI, what we kept or decided ourselves,
and what we checked (math or a source, code or data, a claim or figure).

| Date | Who | Tool | Delegated to AI | Kept / decided by us | How we checked it |
|---|---|---|---|---|---|
| 2026-11-09 | E | Claude | End-to-end check of the repo: ran the unit tests and `run_if.py`, `scaling_experiment.py`, `tune_if.py` on the shared Drive split; wrote `src/check_split.py` (+ tests), `requirements.txt`, README fixes | Kept the team's stratified random 60/20/20 split (seed 0) as is; no change to experiment code or results | Regenerated `results/*.csv` matched the committed files byte for byte; split files checked for row/fraud counts, no overlap or duplicates, and fingerprinted (SHA-256) |
| 2026-11-09 | A (covered by E) | Claude | Wrote `src/toy_1d.py` and `src/demo_2d.py` (+ `tests/test_demos.py`) and the figures `toy_1d.png`, `demo_2d_isolation.png`, `demo_2d_path_convergence.png` | Chose the examples (1-2-3-4-100; one 2D anomaly vs. the most central point); the drawn tree is a typical one (both path lengths equal their 1,000-tree averages) | Exact E[h] (recursion) agrees with a 20,000-tree simulation to 0.01 and gives the same ranking as sklearn's IsolationForest; c(n) checked against Liu et al. (2008); figures inspected |
