"""Torch Dataset wrapper around `.npz` embeddings, plus loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """One row = one utterance: embedding, emotion label, speaker label."""

    def __init__(self, embeddings, emotion_labels, speaker_labels=None):
        self.embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.emotion_labels = torch.as_tensor(
            np.asarray(emotion_labels), dtype=torch.long
        )
        self.speaker_labels = (
            None
            if speaker_labels is None
            else torch.as_tensor(np.asarray(speaker_labels), dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> dict:
        sample = {
            "embedding": self.embeddings[index],
            "emotion": self.emotion_labels[index],
        }
        if self.speaker_labels is not None:
            sample["speaker"] = self.speaker_labels[index]
        return sample


def load_split(embedding_dir, split: str) -> EmbeddingDataset:
    """Load one split (.npz) as an `EmbeddingDataset`."""
    path = Path(embedding_dir) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        return EmbeddingDataset(
            data["embeddings"], data["emotion_labels"], data["speaker_labels"]
        )


def load_metadata(embedding_dir, split: str) -> dict[str, np.ndarray]:
    """Speaker/sentence/emotion metadata aligned to embedding rows."""
    path = Path(embedding_dir) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {
            "speakers": data["speaker_ids"],
            "sentences": data["sentence_ids"],
            "emotions": data["emotion_labels"],
        }