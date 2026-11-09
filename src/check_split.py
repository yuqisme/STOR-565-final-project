"""
check_split.py  --  sanity check of the shared split files (owner: E).

Run after putting train.csv / validation.csv / test.csv in data/random_split/
(from make_random_split.py or the shared Google Drive folder):

    python src/check_split.py

Checks that the three files form a valid deduplicated 60/20/20 stratified split
and prints a SHA-256 fingerprint per file. Everyone should see the SAME
fingerprints; if yours differ, you are not using the shared split.

The test file is opened only to count rows and labels, never to fit or score.
Exits with status 1 if any check fails.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import DATA_DIR, FILES, LABEL  # noqa: E402

EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", LABEL]
EXPECTED_TOTAL_ROWS = 283_726      # 284,807 raw rows minus 1,081 exact duplicates
EXPECTED_TOTAL_FRAUDS = 473        # 492 raw frauds minus 19 duplicated fraud rows
FRACS = {"train": 0.6, "val": 0.2, "test": 0.2}
# Fingerprints of the shared split (Google Drive files, checked 2026-11-09).
EXPECTED_SHA256 = {
    "train": "b4c3084e8c72ffaa2c67fa10c150238acc10c3ced96d7595c4dc25158f13ed18",
    "val": "dc0a516d6d3f99680efecc250c66b0fadb71939f65347e3b7cd37ab670b91ef8",
    "test": "27a3b706db4eb485b3e07c060231f2ade37e0ea4b72edfca9f9378848e78de86",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(data_dir=DATA_DIR, expected_sha=EXPECTED_SHA256,
          total_rows=EXPECTED_TOTAL_ROWS, total_frauds=EXPECTED_TOTAL_FRAUDS):
    """Return a list of failure messages (empty = all checks passed)."""
    data_dir = Path(data_dir)
    fails, dfs = [], {}
    for name, fname in FILES.items():
        path = data_dir / fname
        if not path.exists():
            fails.append(f"missing file {path}")
            continue
        df = pd.read_csv(path)
        dfs[name] = df
        digest = sha256(path)
        match = "" if expected_sha is None else (
            "  (matches shared split)" if digest == expected_sha.get(name) else "  (DIFFERENT from shared split)")
        print(f"{fname:15s} rows {len(df):>7,}  frauds {int(df[LABEL].sum()):>3}  "
              f"rate {df[LABEL].mean():.4%}  sha256 {digest[:12]}{match}")
        if expected_sha is not None and digest != expected_sha.get(name):
            fails.append(f"{fname}: fingerprint differs from the shared split")
        if list(df.columns) != EXPECTED_COLUMNS:
            fails.append(f"{fname}: unexpected columns")
        if df.isnull().any().any():
            fails.append(f"{fname}: missing values")
        if not df[LABEL].isin([0, 1]).all():
            fails.append(f"{fname}: {LABEL} is not 0/1")
    if len(dfs) < len(FILES):
        return fails

    allr = pd.concat(dfs.values(), ignore_index=True)
    n, n_pos = len(allr), int(allr[LABEL].sum())
    if total_rows is not None and n != total_rows:
        fails.append(f"total rows {n} != expected {total_rows}")
    if total_frauds is not None and n_pos != total_frauds:
        fails.append(f"total frauds {n_pos} != expected {total_frauds}")
    if allr.duplicated().any():
        fails.append(f"{int(allr.duplicated().sum())} duplicate rows (within or across splits)")
    for name, frac in FRACS.items():
        df = dfs[name]
        if abs(len(df) - frac * n) > 2:
            fails.append(f"{name}: {len(df)} rows, expected about {frac * n:.0f}")
        if abs(df[LABEL].sum() - frac * n_pos) > 2:
            fails.append(f"{name}: {int(df[LABEL].sum())} frauds, expected about {frac * n_pos:.0f}")
    return fails


def main():
    fails = check()
    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\nAll split checks passed.")


if __name__ == "__main__":
    main()
