"""Frozen WavLM embedding extraction.

Produces one 768-dim mean-pooled embedding per recording. Chunked and
resumable so Colab disconnects do not lose progress.

Note on the private API: `WavLMModel._get_feature_vector_attention_mask`
is used to convert waveform-level attention masks into frame-level masks
for mean pooling. It is a private method in `transformers`; if a future
version renames it, replace with the documented public equivalent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm

SAMPLE_RATE = 16000
BATCH_SIZE = 8
CHUNK_SIZE = 64


def load_audio(file_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio, convert to mono, resample to `sample_rate`."""
    audio, sr = sf.read(str(file_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"Invalid audio: {file_path}")
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        audio = resample_poly(
            audio, sample_rate // divisor, sr // divisor
        ).astype(np.float32)
    return audio


def extract_embedding_batch(
    audio_list: list[np.ndarray],
    wavlm,
    feature_extractor,
    device: torch.device,
) -> np.ndarray:
    """Return one 768-dim mean-pooled embedding per recording."""
    inputs = feature_extractor(
        audio_list,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = wavlm(input_values, attention_mask=attention_mask)
        else:
            outputs = wavlm(input_values, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        frame_mask = wavlm._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask
        )
        mask = frame_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1)

    return pooled.float().cpu().numpy()


def extract_split(
    split_name: str,
    df: pd.DataFrame,
    audio_dir: Path,
    output_dir: Path,
    wavlm,
    feature_extractor,
    device: torch.device,
    emotion_to_id: dict[str, int],
    speaker_to_id: dict[str, int],
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Extract embeddings for one split, resuming from completed chunks."""
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    chunk_dir = output_dir / "chunks" / split_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    final_file = output_dir / f"{split_name}.npz"

    if final_file.exists():
        print(f"{split_name}: already extracted, skipping.")
        return

    total_chunks = math.ceil(len(split_df) / chunk_size)

    for chunk_idx in tqdm(range(total_chunks), desc=f"Extracting {split_name}"):
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
        if chunk_file.exists():
            continue

        chunk_df = split_df.iloc[
            chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
        ].copy()

        chunk_embeddings = []
        for start in range(0, len(chunk_df), batch_size):
            batch_df = chunk_df.iloc[start : start + batch_size]
            audio_list = [
                load_audio(audio_dir / filename) for filename in batch_df["filename"]
            ]
            chunk_embeddings.append(
                extract_embedding_batch(audio_list, wavlm, feature_extractor, device)
            )

        embeddings = np.concatenate(chunk_embeddings, axis=0)

        emotion_labels = chunk_df["emotion"].map(emotion_to_id).to_numpy(np.int64)
        speaker_labels = (
            chunk_df["speaker_id"].map(speaker_to_id).fillna(-1).to_numpy(np.int64)
        )

        tmp = chunk_file.with_name(chunk_file.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            embeddings=embeddings.astype(np.float32),
            emotion_labels=emotion_labels,
            speaker_labels=speaker_labels,
            speaker_ids=chunk_df["speaker_id"].to_numpy(str),
            filenames=chunk_df["filename"].to_numpy(str),
            sentence_ids=chunk_df["sentence"].to_numpy(str),
            intensities=chunk_df["intensity"].to_numpy(str),
        )
        tmp.replace(chunk_file)

    keys = [
        "embeddings", "emotion_labels", "speaker_labels", "speaker_ids",
        "filenames", "sentence_ids", "intensities",
    ]
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for chunk_idx in range(total_chunks):
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
        with np.load(chunk_file, allow_pickle=False) as data:
            for key in keys:
                collected[key].append(data[key])

    merged = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    assert merged["embeddings"].shape == (len(split_df), 768)

    tmp = final_file.with_name(final_file.stem + ".tmp.npz")
    np.savez_compressed(tmp, **merged)
    tmp.replace(final_file)
    print(f"{split_name}: saved {len(merged['embeddings'])} embeddings.")