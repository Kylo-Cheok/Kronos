"""Leakage-safe auxiliary objectives for short A-share forecasting horizons.

The predictor still learns the original next-token task.  This module adds
direct supervision for the actionable 1/3/5/10-trading-day endpoints without
turning the raw future window into a model input.  Raw close prices are used
only to construct labels; model inputs stay normalized as before.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_HORIZONS = (1, 3, 5, 10)
DOWN_CLASS = 0
FLAT_CLASS = 1
UP_CLASS = 2


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(horizon) for horizon in horizons)
    if not normalized or any(horizon <= 0 for horizon in normalized):
        raise ValueError("horizons must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must be unique")
    return normalized


def make_direction_deadzones(
    raw_close: torch.Tensor,
    *,
    context_length: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_deadzone: float = 0.003,
    volatility_multiplier: float = 0.5,
) -> torch.Tensor:
    """Return a past-only log-return dead zone for every requested horizon."""
    horizons = _validate_horizons(horizons)
    if raw_close.ndim != 2:
        raise ValueError("raw_close must have shape [batch, time]")
    if context_length < 2 or context_length > raw_close.shape[1]:
        raise ValueError("context_length must include at least two close prices")
    if min_deadzone < 0.0 or volatility_multiplier < 0.0:
        raise ValueError("dead-zone parameters must be non-negative")
    if not torch.isfinite(raw_close).all() or (raw_close <= 0.0).any():
        raise ValueError("raw_close must contain finite positive prices")

    past_log_returns = torch.log(raw_close[:, 1:context_length] / raw_close[:, : context_length - 1])
    daily_volatility = past_log_returns.std(dim=1, unbiased=False)
    horizon_scale = torch.sqrt(
        torch.tensor(horizons, dtype=raw_close.dtype, device=raw_close.device)
    )
    dynamic_deadzone = daily_volatility.unsqueeze(1) * horizon_scale.unsqueeze(0)
    return torch.maximum(
        dynamic_deadzone * float(volatility_multiplier),
        torch.full_like(dynamic_deadzone, float(min_deadzone)),
    )


def make_multihorizon_targets(
    raw_close: torch.Tensor,
    *,
    context_length: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_deadzone: float = 0.003,
    volatility_multiplier: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Build log-return and down/flat/up targets from raw closes only."""
    horizons = _validate_horizons(horizons)
    if raw_close.ndim != 2:
        raise ValueError("raw_close must have shape [batch, time]")
    context_index = int(context_length) - 1
    target_indices = [context_index + horizon for horizon in horizons]
    if context_index < 0 or max(target_indices) >= raw_close.shape[1]:
        raise ValueError("context_length and horizons are outside raw_close")
    if not torch.isfinite(raw_close).all() or (raw_close <= 0.0).any():
        raise ValueError("raw_close must contain finite positive prices")

    current_close = raw_close[:, context_index].unsqueeze(1)
    future_close = raw_close[:, target_indices]
    future_returns = torch.log(future_close / current_close)
    deadzones = make_direction_deadzones(
        raw_close,
        context_length=context_length,
        horizons=horizons,
        min_deadzone=min_deadzone,
        volatility_multiplier=volatility_multiplier,
    )
    direction = torch.full_like(future_returns, FLAT_CLASS, dtype=torch.long)
    direction = torch.where(future_returns > deadzones, UP_CLASS, direction)
    direction = torch.where(future_returns < -deadzones, DOWN_CLASS, direction)
    return {
        "returns": future_returns,
        "direction": direction,
        "deadzone": deadzones,
        "current_close": current_close,
        "future_close": future_close,
    }


class MultiHorizonForecastHead(nn.Module):
    """Pool only causal context states and emit direct short-horizon endpoints."""

    def __init__(
        self,
        d_model: int,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        pool_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = _validate_horizons(horizons)
        self.pool_size = int(pool_size)
        if self.pool_size < 1:
            raise ValueError("pool_size must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        width = int(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.return_head = nn.Linear(width, len(self.horizons))
        self.direction_head = nn.Linear(width, len(self.horizons) * 3)

    def forward(
        self, hidden_states: torch.Tensor, *, context_length: int
    ) -> dict[str, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, time, d_model]")
        context_index = int(context_length) - 1
        if context_index < 0 or context_index >= hidden_states.shape[1]:
            raise ValueError("context_length is outside hidden_states")
        pool_start = max(0, context_index - self.pool_size + 1)
        causal_states = hidden_states[:, pool_start : context_index + 1, :]
        pooled = torch.cat((causal_states[:, -1, :], causal_states.mean(dim=1)), dim=-1)
        representation = self.trunk(pooled)
        direction_logits = self.direction_head(representation).reshape(
            hidden_states.shape[0], len(self.horizons), 3
        )
        return {
            "return_prediction": self.return_head(representation),
            "direction_logits": direction_logits,
        }


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t).

    ``weight`` acts as per-class alpha (same semantics as F.cross_entropy).
    ``gamma`` down-weights easy/well-classified examples so the loss focuses
    on hard UP/DOWN confusions instead of the FLAT majority.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    p_t = torch.exp(-ce)
    focal = ((1.0 - p_t) ** gamma) * ce
    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


def compute_multihorizon_objective(
    head: MultiHorizonForecastHead,
    hidden_states: torch.Tensor,
    raw_close: torch.Tensor,
    *,
    context_length: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_deadzone: float = 0.003,
    volatility_multiplier: float = 0.5,
    huber_delta: float = 0.02,
    class_weights: torch.Tensor | None = None,
    consistency_loss_weight: float = 0.0,
    horizon_weights: Sequence[float] | None = None,
    direction_loss_type: str = "ce",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    swap_penalty_weight: float = 0.0,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Compute direct return and three-class direction losses at all horizons.

    When ``consistency_loss_weight`` > 0, adds a soft penalty whenever the
    direction head and the return head disagree on sign: if P(up) > P(down)
    but the predicted log-return is negative (or vice versa), the product
    ``direction_sign * return_pred`` is negative and the hinge fires.  This
    keeps the two heads consistent without forcing a hard tie.
    """
    horizons = _validate_horizons(horizons)
    if tuple(head.horizons) != horizons:
        raise ValueError("head horizons and requested horizons must match")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    outputs = head(hidden_states, context_length=context_length)
    targets = make_multihorizon_targets(
        raw_close,
        context_length=context_length,
        horizons=horizons,
        min_deadzone=min_deadzone,
        volatility_multiplier=volatility_multiplier,
    )
    return_loss = F.huber_loss(
        outputs["return_prediction"], targets["returns"], delta=float(huber_delta)
    )
    direction_logits = outputs["direction_logits"]
    weights = class_weights.to(device=direction_logits.device, dtype=direction_logits.dtype) if class_weights is not None else None
    use_focal = direction_loss_type.lower() == "focal"
    if horizon_weights is not None:
        hw = torch.as_tensor(
            [float(x) for x in horizon_weights],
            device=direction_logits.device, dtype=direction_logits.dtype,
        )
        if hw.numel() != len(horizons):
            raise ValueError("horizon_weights must match horizons")
        ce = focal_loss(
            direction_logits.reshape(-1, 3), targets["direction"].reshape(-1),
            weight=weights, gamma=focal_gamma, reduction="none",
        ) if use_focal else F.cross_entropy(
            direction_logits.reshape(-1, 3), targets["direction"].reshape(-1),
            weight=weights, reduction="none", label_smoothing=label_smoothing,
        )
        ce = ce.reshape(direction_logits.shape[0], len(horizons))
        direction_loss = (ce * hw).sum() / (hw.sum() * ce.shape[0])
    else:
        if use_focal:
            direction_loss = focal_loss(
                direction_logits.reshape(-1, 3), targets["direction"].reshape(-1),
                weight=weights, gamma=focal_gamma,
            )
        else:
            direction_loss = F.cross_entropy(
                direction_logits.reshape(-1, 3), targets["direction"].reshape(-1),
                weight=weights, label_smoothing=label_smoothing,
            )
    direction_prediction = direction_logits.argmax(dim=-1)
    direction_accuracy_by_horizon = (
        direction_prediction == targets["direction"]
    ).float().mean(dim=0)

    # Sign-consistency between direction probabilities and return prediction.
    # direction_sign in [-1, 1]: +1 = confident up, -1 = confident down.
    consistency_loss = torch.tensor(0.0, device=direction_logits.device)
    if consistency_loss_weight > 0.0:
        direction_probs = torch.softmax(direction_logits, dim=-1)  # [B, H, 3]
        direction_sign = direction_probs[..., UP_CLASS] - direction_probs[..., DOWN_CLASS]  # [B, H]
        # Penalise when sign(direction) and sign(return) disagree.
        consistency_loss = F.relu(-direction_sign * outputs["return_prediction"]).mean()

    # UP-DOWN swap penalty: hinge on P(opposite) - P(correct).  Only fires
    # when the model assigns MORE probability to the opposite class than the
    # correct one (direction reversal).  FLAT predictions (P(opp)≈P(corr))
    # incur ~0 penalty, so this does NOT incentivise FLAT collapse unlike
    # raw P(opposite).  Targets the worst error type diagnosed in the
    # confusion matrix (UP↔DOWN systematic confusion).
    swap_penalty = torch.tensor(0.0, device=direction_logits.device)
    if swap_penalty_weight > 0.0:
        direction_probs = torch.softmax(direction_logits, dim=-1)  # [B, H, 3]
        target_dir = targets["direction"]  # [B, H]
        up_mask = target_dir == UP_CLASS
        down_mask = target_dir == DOWN_CLASS
        penalties = []
        if up_mask.any():
            # true UP: penalise max(0, P(DOWN) - P(UP))
            opp_minus_corr = (
                direction_probs[..., DOWN_CLASS][up_mask]
                - direction_probs[..., UP_CLASS][up_mask]
            )
            penalties.append(F.relu(opp_minus_corr).mean())
        if down_mask.any():
            # true DOWN: penalise max(0, P(UP) - P(DOWN))
            opp_minus_corr = (
                direction_probs[..., UP_CLASS][down_mask]
                - direction_probs[..., DOWN_CLASS][down_mask]
            )
            penalties.append(F.relu(opp_minus_corr).mean())
        if penalties:
            swap_penalty = torch.stack(penalties).mean()

    return {
        "return_loss": return_loss,
        "direction_loss": direction_loss,
        "consistency_loss": consistency_loss,
        "swap_penalty": swap_penalty,
        "direction_accuracy": direction_accuracy_by_horizon.mean(),
        "direction_accuracy_by_horizon": direction_accuracy_by_horizon,
        "outputs": outputs,
        "targets": targets,
    }
