"""Independent speaker-verification attacker.

A small MLP trained on `|z_i - z_j|` to predict whether two utterances
belong to the same speaker. Trained on train speakers, selected on
validation AUC, evaluated on held-out test speakers.

The `StandardScaler` is fit on training pairs ONLY. Never refit it on
validation or test — that would leak and invalidate the attack.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_eer


class VerificationAttack(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attack_auc(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> float:
    device = _resolve_device(device)
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            logits = model(features)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(targets.numpy())
    return float(roc_auc_score(labels, probabilities))


def train_verification_attack(
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    name: str,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # Fit scaler on TRAIN pairs only — validation/test are transformed, not fit.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ),
            batch_size=256,
            shuffle=shuffle,
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    model = VerificationAttack(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_auc = get_attack_auc(model, val_loader, device=device)
        if verbose:
            print(
                f"{name} | Epoch {epoch + 1:02d}/{max_epochs} | "
                f"Loss: {total_loss / len(train_loader):.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if verbose:
                print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()

    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for features, targets in test_loader:
            logits = model(features.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(targets.numpy())

    probabilities_arr = np.array(probabilities)
    labels_arr = np.array(labels)
    test_auc = float(roc_auc_score(labels_arr, probabilities_arr))
    test_eer = float(compute_eer(labels_arr, probabilities_arr))

    if verbose:
        print(f"\n{name} FINAL RESULTS")
        print(f"Validation AUC: {best_auc:.4f}")
        print(f"Test ROC-AUC:   {test_auc:.4f}")
        print(f"Test EER:       {test_eer:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "validation_auc": float(best_auc),
        "test_auc": test_auc,
        "test_eer": test_eer,
    }