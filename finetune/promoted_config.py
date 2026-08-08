"""Promoted multihorizon inference config (Phase-5 dual-metric winner).

Direction (absolute down/flat/up):
  - h=1,10 → r10_joint_splitlr
  - h=3,5 → r5_frozen_pool48
  - Direction logits: TTA-averaged PER HORIZON over different lookback sets
    (Phase-4: uniform 5lb [124,126,128,130,132] for all horizons)
    (P5-2: per-horizon optimal — h=1 keeps 5lb to preserve gate precision,
     h=3/5/10 use tailored lookbacks boosting ungated nonflat +1.08pt)
    (P4-10: h=3 switched R10→R5 under TTA, nonflat +3.5pt)

Return (log-return endpoints, lower MAE):
  - per-horizon convex blend: h=1: 0.9*R10+0.1*R5, h=3: 0.7*R10+0.3*R5,
    h=5: 1.0*R10, h=10: 0.7*R10+0.3*R5
  - Uses single-lookback (lb=128) returns (NOT TTA-averaged) to preserve MAE

h=1 selective gate (trading):
  - actionable_score >= 0.45 (on TTA-averaged logits)
  - min_abs_return = 0.0 (magnitude filter removed; TTA confidence alone
    suffices to stay in 20–40% coverage band at 20.8%)
  - Gate magnitude uses single-lookback R10 raw return
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Phase-2 historical baseline mapping (for comparison).
PHASE2_BASELINE_MODEL_BY_HORIZON: dict[int, str] = {
    1: "r10_joint_splitlr",
    3: "r5_frozen_pool48",
    5: "r5_frozen_pool48",
    10: "r5_frozen_pool48",
}

# Phase-4 promoted direction mapping.
# P4-10: h=3 switched from R10 to R5 — under TTA, R5 gives 72.94% vs R10's 69.41%.
PROMOTED_MODEL_BY_HORIZON: dict[int, str] = {
    1: "r10_joint_splitlr",
    3: "r5_frozen_pool48",
    5: "r5_frozen_pool48",
    10: "r10_joint_splitlr",
}

# Return blend weights: primary model is R10, secondary R5.
# Per-horizon optimal weights (P4-9 coarse 0.05/0.1 step):
#   h=1: 0.9, h=3: 0.7, h=5: 1.0 (pure R10), h=10: 0.7
# Phase-5 (P5-5) fine sweep at 0.025 step:
#   h=1: 0.925, h=3: 0.775, h=5: 1.0 (unchanged), h=10: 0.65
#   MAE overall 0.037457 -> 0.037448 (-0.024% rel); pass via mae_win_direction_nonregress
PROMOTED_RETURN_BLEND = {
    "primary": "r10_joint_splitlr",
    "secondary": "r5_frozen_pool48",
    "primary_weight": 0.85,  # legacy default; overridden by per-horizon below
    "primary_weight_by_horizon": {
        1: 0.925,
        3: 0.775,
        5: 1.0,
        10: 0.65,
    },
}

# Phase-4: TTA lookbacks for direction inference.
# 128 is the trained lookback; ±4 covers a 9-day neighbourhood.
# Averaging logits across these reduces direction variance without smoothing
# returns (returns use single-lookback lb=128).
PROMOTED_TTA_LOOKBACKS: tuple[int, ...] = (124, 126, 128, 130, 132)

# Phase-5 (P5-2): Per-horizon TTA lookback configuration.
# h=1 keeps Phase-4 5lb_current to preserve gate precision (70.0%/91.3%).
# h=3/5/10 use P5-1 optimal lookbacks, boosting ungated nonflat +1.08pt
# without affecting h=1 gate (gate only depends on h=1 logits/returns).
#
# Phase-5 (P5-3): h=1 switched 5lb_current -> 4lb_left + mag=0.002 gate.
# 4lb_left has same ungated nonflat (65.93%) but better confidence
# distribution, enabling mag=0.002 to filter wrong calls: gate prec
# 70.0% -> 72.73% (+2.73pt), gated_nf 91.3% -> 92.31% (+1.00pt).
PROMOTED_TTA_LOOKBACKS_BY_HORIZON: dict[int, tuple[int, ...]] = {
    1: (124, 126, 128, 130),              # 4lb_left (P5-3: enables mag=0.002 gate win)
    3: (124, 126, 128, 130),              # 4lb_left (P5-1 optimal, +2.64pt)
    5: (124, 126, 128, 130, 132, 134),    # 6lb (P5-1 optimal, +1.38pt)
    10: (126, 128, 130, 132),             # 4lb_right (P5-1 optimal, +0.96pt)
}

# h=1 selective gate.
# Phase-4 changes vs Phase-3:
#   - confidence computed on TTA-averaged logits (more stable → +6.4pt precision)
#   - min_abs_return lowered 0.003 → 0.0 (TTA softens logits, dropping coverage
#     to 19.4% with mag=0.003; removing mag restores to 20.8% in-band)
#   - direction_source = "tta_averaged", return_source = "single_lookback_blend"
# Phase-5 (P5-3) changes vs Phase-4:
#   - h=1 lookback switched 5lb_current -> 4lb_left (better confidence distribution)
#   - min_abs_return restored 0.0 -> 0.002 (4lb_left's logits allow mag filter
#     to remove wrong calls: gate prec 70.0% -> 72.73%, gated_nf 91.3% -> 92.31%)
# Phase-5 (P5-8) changes vs P5-3:
#   - Added multi-horizon consistency filter: h=1 call requires h=5 to predict
#     the SAME direction (strict agreement). Filters 1 wrong h=1 call where h=5
#     disagreed. Gate prec 72.73% -> 75.00% (+2.27pt), gated_nf 92.31% -> 96.00%
#     (+3.69pt). Coverage 22.9% -> 22.2% (still in 20-40% band).
PROMOTED_H1_GATE = {
    "confidence_key": "actionable_score",
    "confidence_threshold": 0.45,
    "min_abs_return": 0.002,
    "require_sign_agree": False,
    "transaction_cost": 0.0005,
    "magnitude_return_source": "primary",  # R10 single-lookback raw |ret|
    "direction_source": "tta_averaged",
    "return_source": "single_lookback_blend",
    "consistency_filter": "strict_h5",  # P5-8: require h=5 same-direction agreement
}


def model_dir(exp_name: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "models"
        / f"a_share_multihorizon_predictor_{exp_name}"
        / "checkpoints"
        / "best_model"
    )


def resolve_promoted_model_dirs() -> dict[int, Path]:
    return {h: model_dir(name) for h, name in PROMOTED_MODEL_BY_HORIZON.items()}
