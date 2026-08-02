"""Auxiliary direction objective for Kronos predictor fine-tuning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirectionAuxiliaryHead(nn.Module):
    """Map the causal context-boundary representation to an up logit."""

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(int(d_model), 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_state).squeeze(-1)


def make_direction_targets(
    batch_x: torch.Tensor,
    context_length: int,
    horizon: int,
    close_index: int = 3,
) -> torch.Tensor:
    """Return binary labels for close[t+horizon] > close[t]."""
    if batch_x.ndim != 3:
        raise ValueError("batch_x must have shape [batch, time, feature]")
    context_index = int(context_length) - 1
    target_index = context_index + int(horizon)
    if context_index < 0 or target_index >= batch_x.shape[1]:
        raise ValueError("context_length and horizon are outside batch_x")
    if not 0 <= int(close_index) < batch_x.shape[2]:
        raise ValueError("close_index is outside the feature dimension")

    current_close = batch_x[:, context_index, int(close_index)]
    future_close = batch_x[:, target_index, int(close_index)]
    return (future_close > current_close).to(dtype=batch_x.dtype)


def compute_direction_objective(
    head: DirectionAuxiliaryHead,
    hidden_states: torch.Tensor,
    batch_x: torch.Tensor,
    context_length: int,
    horizon: int,
    close_index: int = 3,
) -> dict:
    """Compute leak-free direction loss from the context boundary only."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, time, d_model]")
    context_index = int(context_length) - 1
    if context_index < 0 or context_index >= hidden_states.shape[1]:
        raise ValueError("context_length is outside hidden_states")
    targets = make_direction_targets(
        batch_x,
        context_length=context_length,
        horizon=horizon,
        close_index=close_index,
    )
    logits = head(hidden_states[:, context_index, :])
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    accuracy = ((logits >= 0.0) == targets.bool()).float().mean()
    return {
        "loss": loss,
        "accuracy": accuracy,
        "logits": logits,
        "targets": targets,
    }
