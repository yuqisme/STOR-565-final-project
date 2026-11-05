"""
make_random_split.py  --  one-time stratified random 60/20/20 split.

Downloads the original Kaggle file (via data_loader.load_raw), drops exact
duplicate rows, splits randomly with a fixed seed, and writes
data/random_split/{train,validation,test}.csv. All columns (incl. Time) are kept.
Run once:  python src/make_random_split.py
"""
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "random_split"
SEED = 0
FRACS = (0.6, 0.2, 0.2)          # train, validation, test
LABEL = "Class"


def split(df, seed=SEED, fracs=FRACS):
    """Stratified random split of df into (train, val, test) DataFrames."""
    train, rest = train_test_split(df, train_size=fracs[0], stratify=df[LABEL],
                                   random_state=seed)
    val, test = train_test_split(rest, train_size=fracs[1] / (fracs[1] + fracs[2]),
                                 stratify=rest[LABEL], random_state=seed)
    return train, val, test


def main():
    pooled = load_raw()
    df = pooled.drop_duplicates().reset_index(drop=True)   # no copies across splits
    print(f"raw {len(pooled)} rows; dropped {len(pooled) - len(df)} exact duplicate rows "
          f"({int(pooled[LABEL].sum() - df[LABEL].sum())} of them fraud)")
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train), ("validation", val), ("test", test)]:
        d.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} rows")
    print(f"after dedup: {len(df)} rows, {int(df[LABEL].sum())} frauds -> {OUT_DIR}")


if __name__ == "__main__":
    main()
