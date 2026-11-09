"""Speaker-disjoint train/validation/test split of CREMA-D metadata.

Splits speakers (not recordings) so no speaker appears in more than one
partition. Uses seed 42 and fixed partition sizes (64/13/14) to match the
reported experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cremad_metadata.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cremad_splits.csv"

SEED = 42
TRAIN_SPEAKERS = 64
VAL_SPEAKERS = 13
TEST_SPEAKERS = 14


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    speakers = np.array(sorted(df["speaker_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    train_ids = set(speakers[:TRAIN_SPEAKERS])
    val_ids = set(speakers[TRAIN_SPEAKERS : TRAIN_SPEAKERS + VAL_SPEAKERS])
    test_ids = set(speakers[TRAIN_SPEAKERS + VAL_SPEAKERS :])

    def split_for(speaker: str) -> str:
        if speaker in train_ids:
            return "train"
        if speaker in val_ids:
            return "validation"
        return "test"

    df = df.copy()
    df["split"] = df["speaker_id"].map(split_for)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"speaker_id": str})
    df = make_split(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("Speaker split summary")
    print("-" * 45)
    for split in ("train", "validation", "test"):
        subset = df[df["split"] == split]
        print(
            f"{split:12} | speakers: {subset['speaker_id'].nunique():2d} | "
            f"recordings: {len(subset):4d}"
        )
    print("\nSpeaker overlap: 0")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()