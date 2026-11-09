"""Train and evaluate emotion classification heads.

Consumes `EmbeddingDataset` batches, which are dicts with keys
"embedding", "emotion", and (optionally) "speaker".

The baseline uses a bare `EmotionClassifier` on frozen WavLM embeddings.
The multitask and adversarial models wrap a shared projection with
emotion and speaker heads (see `src.models.model`).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_emotion(
    model: nn.Module,
    loader: DataLoader,
    baseline: bool = False,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = _resolve_device(device)
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            logits = model(embeddings) if baseline else model(embeddings)[0]
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch["emotion"].numpy())

    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str,
    output_dir: Path,
    baseline: bool = False,
    max_epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    device = _resolve_device(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    emotion_criterion = nn.CrossEntropyLoss()
    speaker_criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            embeddings = batch["embedding"].to(device)
            emotions = batch["emotion"].to(device)

            optimizer.zero_grad()
            if baseline:
                logits = model(embeddings)
                loss = emotion_criterion(logits, emotions)
            else:
                speakers = batch["speaker"].to(device)
                emotion_logits, speaker_logits, _ = model(embeddings)
                loss = emotion_criterion(emotion_logits, emotions) + speaker_criterion(
                    speaker_logits, speakers
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_emotion(
            model, val_loader, baseline=baseline, device=device
        )
        avg_loss = total_loss / len(train_loader)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{max_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            if verbose:
                print("Early stopping.")
            break

    model.load_state_dict(best_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{experiment_name}_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    if verbose:
        print(f"\nBest validation Macro-F1: {best_f1:.4f}")
        print(f"Saved checkpoint: {checkpoint_path}")

    return model, history


def extract_representations(
    model: nn.Module,
    loader: DataLoader,
    baseline: bool = False,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract per-utterance representations aligned to `loader` order.

    CRITICAL: pass a loader with `shuffle=False`. The pair generator
    downstream assumes these representations line up row-by-row with the
    metadata returned by `dataset.load_metadata`. Using a shuffling loader
    here silently produces misaligned pairs and invalid attack results.

    For baseline (`baseline=True`), returns the frozen embeddings directly.
    For multitask/adversarial, returns the shared projection output.
    """
    device = _resolve_device(device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            representation = embeddings if baseline else model.projection(embeddings)
            chunks.append(representation.cpu().numpy())
    return np.concatenate(chunks, axis=0)