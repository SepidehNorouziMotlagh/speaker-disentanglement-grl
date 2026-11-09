"""Emotion and speaker classification heads.

`EmotionClassifier` defaults to `input_dim=256` because it is used with
the 256-d shared representation. The baseline passes `input_dim=768`
explicitly to operate directly on frozen WavLM embeddings.
"""

from __future__ import annotations

import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, num_classes: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, num_speakers: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_speakers),
        )

    def forward(self, x):
        return self.network(x)