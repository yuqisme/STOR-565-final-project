# AI-use log

Running log for the report's one-page AI-use appendix (maintained by E).
For each entry: what we delegated to AI, what we kept or decided ourselves,
and what we checked (math or a source, code or data, a claim or figure).

| Date | Who | Tool | Delegated to AI | Kept / decided by us | How we checked it |
|---|---|---|---|---|---|
| 2026-11-09 | E | Claude | End-to-end check of the repo: ran the unit tests and `run_if.py`, `scaling_experiment.py`, `tune_if.py` on the shared Drive split; wrote `src/check_split.py` (+ tests), `requirements.txt`, README fixes | Kept the team's stratified random 60/20/20 split (seed 0) as is; no change to experiment code or results | Regenerated `results/*.csv` matched the committed files byte for byte; split files checked for row/fraud counts, no overlap or duplicates, and fingerprinted (SHA-256) |
