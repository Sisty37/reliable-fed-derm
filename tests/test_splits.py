"""Run with: python -m pytest tests  (or python tests/test_splits.py)

Checks the integrity rules in PROTOCOL.md:
  1. lesion-grouped splits: no lesion in both train and test
  2. image-level and lesion-grouped splits of the same seed have nearly equal train size
  3. the official split and all re-splits cover the same 23,247 images
     (clean/control drop training images by design)
"""
from pathlib import Path

import pandas as pd

S = Path(__file__).resolve().parents[1] / "splits"


def _load(name):
    return pd.read_csv(S / name)


def test_lesion_grouped_disjoint():
    files = sorted(S.glob("resplit_lesion_seed*.csv")) + [S / "split_clean.csv"]
    files = [f for f in files if f.exists()]
    assert files, "no lesion-grouped split files found"
    for f in files:
        d = pd.read_csv(f)
        tr = set(d.loc[d.fold == "train", "lesion_id"])
        te = set(d.loc[d.fold == "test", "lesion_id"])
        assert not (tr & te), f"{f.name}: {len(tr & te)} lesions in both train and test"


def test_size_matched():
    for s in range(5):
        a, b = S / f"resplit_image_seed{s}.csv", S / f"resplit_lesion_seed{s}.csv"
        if a.exists() and b.exists():
            na = (pd.read_csv(a).fold == "train").sum()
            nb = (pd.read_csv(b).fold == "train").sum()
            assert abs(na - nb) <= 0.01 * na, f"seed {s}: train sizes {na} vs {nb}"


def test_same_images():
    files = [S / "split_official.csv"] + sorted(S.glob("resplit_*.csv"))
    files = [f for f in files if f.exists()]
    sets = {f.name: set(pd.read_csv(f).image) for f in files}
    if sets:
        ref = next(iter(sets.values()))
        for n, s in sets.items():
            assert s == ref, f"{n} covers a different image set"
            assert len(s) == 23247, f"{n}: {len(s)} images"


if __name__ == "__main__":
    for t in (test_lesion_grouped_disjoint, test_size_matched, test_same_images):
        t(); print("passed:", t.__name__)
