"""Shared projection network: 768-d WavLM embedding -> 256-d representation."""

from __future__ import annotations

import torch.nn as nn


class ProjectionNetwork(nn.Module):
    def __init__(
        self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)