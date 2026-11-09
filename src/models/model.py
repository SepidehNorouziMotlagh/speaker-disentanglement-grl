"""Combined multitask / adversarial speaker-disentanglement model.

`adversarial=True` (the default, matching the original notebook) inserts a
Gradient Reversal Layer between the shared representation and the speaker
head, so the speaker loss pushes the projection away from speaker identity.
Pass `adversarial=False` explicitly for the multitask control.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.projection import ProjectionNetwork
from src.models.classifiers import EmotionClassifier, SpeakerClassifier
from src.models.grl import GradientReversalLayer


class SpeakerDisentanglementModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        num_emotions: int = 6,
        num_speakers: int = 64,
        adversarial: bool = True,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.projection = ProjectionNetwork(
            input_dim=input_dim, output_dim=projection_dim
        )
        self.emotion_classifier = EmotionClassifier(
            input_dim=projection_dim, num_classes=num_emotions
        )
        self.speaker_classifier = SpeakerClassifier(
            input_dim=projection_dim, num_speakers=num_speakers
        )
        self.adversarial = adversarial
        self.grl = GradientReversalLayer(grl_lambda)

    def forward(self, x):
        representation = self.projection(x)
        emotion_logits = self.emotion_classifier(representation)
        speaker_input = (
            self.grl(representation) if self.adversarial else representation
        )
        speaker_logits = self.speaker_classifier(speaker_input)
        return emotion_logits, speaker_logits, representation