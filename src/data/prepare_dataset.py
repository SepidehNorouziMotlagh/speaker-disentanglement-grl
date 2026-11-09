"""Parse CREMA-D filenames into a metadata CSV.

Reads every `*.wav` under `data/raw/` (or a directory passed via
`--audio-dir`), validates filenames, and writes a metadata table with one
row per recording.

Filename format: {speaker}_{sentence}_{emotion}_{intensity}.wav
Example:        1001_DFA_ANG_XX.wav

If you extracted CREMA-D without flattening the directory structure, pass
`--audio-dir data/raw/AudioWAV`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cremad_metadata.csv"

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_filename(file_path: Path) -> dict | None:
    """Extract metadata from a CREMA-D audio filename."""
    parts = file_path.stem.split("_")
    if len(parts) != 4:
        return None
    actor_id, sentence, emotion_code, intensity = parts
    if emotion_code not in EMOTION_MAP:
        return None
    if not actor_id.isdigit():
        return None
    return {
        "filename": file_path.name,
        "path": str(file_path),
        "speaker_id": actor_id,
        "sentence": sentence,
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP[emotion_code],
        "intensity": intensity,
    }


def build_metadata(audio_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found under {audio_dir}. "
            "If CREMA-D was extracted into an AudioWAV subfolder, "
            "pass --audio-dir data/raw/AudioWAV."
        )

    records, invalid = [], []
    for file_path in audio_files:
        record = parse_filename(file_path)
        (records if record is not None else invalid).append(
            record if record is not None else file_path.name
        )
    if not records:
        raise RuntimeError("No valid CREMA-D WAV files were found.")
    return pd.DataFrame(records), invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    metadata, invalid = build_metadata(args.audio_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)

    print(f"Total WAV files: {len(metadata) + len(invalid)}")
    print(f"Valid recordings: {len(metadata)}")
    print(f"Invalid filenames: {len(invalid)}")
    print(f"Unique speakers: {metadata['speaker_id'].nunique()}")
    print(f"Emotion distribution:\n{metadata['emotion'].value_counts()}")
    print(f"\nMetadata saved to: {args.output}")
    if invalid:
        print("\nFirst invalid filenames:")
        for name in invalid[:20]:
            print(name)


if __name__ == "__main__":
    main()