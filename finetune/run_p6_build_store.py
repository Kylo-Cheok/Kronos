"""Phase 6 — Round builder: build per-lookback + single-lookback caches.

Builds `outputs/p6_per_lb_store.npz` and `outputs/p6_sl_store.npz` for a
candidate set of fine-tuned multihorizon checkpoints (the full model zoo),
so the evaluator can score EVERY model per-horizon and pick the best one —
instead of being limited to the 2 models baked into the old caches.

Robust: skips any model whose checkpoint fails to load.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import (  # noqa: E402
    DEVICE,
    FEATURES,
    LOOKBACK,
    WINDOW,
    VAL_END,
    load_csv,
    load_model,
    make_windows,
    predict_window,
)
from run_p4_r7_expanded_tta import ALL_LOOKBACKS, predict_per_lookback  # noqa: E402
from run_p4_r8_h1_only_tta import save_per_lb_store  # noqa: E402
from run_p4_r9_blend_sweep import save_sl_store  # noqa: E402
from run_tta_eval import make_tta_windows  # noqa: E402

HORIZONS = (1, 3, 5, 10)


def model_dir(name: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "models"
        / f"a_share_multihorizon_predictor_{name}"
        / "checkpoints"
        / "best_model"
    )


def collect_all(names: list[str], symbol: str):
    """Collect per-lb + sl stores for all loadable models (one load pass)."""
    df = load_csv(symbol)
    tta_windows = make_tta_windows(df, ALL_LOOKBACKS)
    sl_windows = make_windows(df)
    print(f"TTA windows: {len(tta_windows)}, SL windows: {len(sl_windows)}")

    per_lb: dict = {}
    sl: dict = {}
    loaded_ok = []
    for name in names:
        path = model_dir(name)
        if not (path / "multihorizon_head.pt").exists():
            print(f"SKIP (no head): {name}")
            continue
        try:
            loaded = load_model(path)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP (load error): {name} -> {e}")
            traceback.print_exc()
            continue
        print(f"Loaded {name} (d_model={loaded.get('d_model')}, "
              f"pool={loaded.get('pool_size')}, horizons={loaded.get('horizons')})")
        loaded_ok.append(name)

        per_lb[name] = {h: {"per_lb_logits": [[] for _ in ALL_LOOKBACKS], "td": [], "tr": []}
                        for h in HORIZONS}
        sl[name] = {h: {"logits": [], "pret": [], "td": [], "tr": []} for h in HORIZONS}

        for i, w in enumerate(tta_windows):
            if i % 30 == 0:
                print(f"  [{name}] tta window {i}/{len(tta_windows)}")
            pred = predict_per_lookback(loaded, w, ALL_LOOKBACKS)
            for lb_idx, lb_logits in enumerate(pred["per_lb_logits"]):
                for hi, h in enumerate(HORIZONS):
                    per_lb[name][h]["per_lb_logits"][lb_idx].append(lb_logits[hi])
            for hi, h in enumerate(HORIZONS):
                per_lb[name][h]["td"].append(int(pred["target_direction"][hi]))
                per_lb[name][h]["tr"].append(float(pred["target_return"][hi]))

        for i, w in enumerate(sl_windows):
            if i % 40 == 0:
                print(f"  [{name}] sl window {i}/{len(sl_windows)}")
            pred = predict_window(loaded, w)
            for hi, h in enumerate(HORIZONS):
                sl[name][h]["logits"].append(pred["logits"][hi])
                sl[name][h]["pret"].append(float(pred["pred_return"][hi]))
                sl[name][h]["td"].append(int(pred["target_direction"][hi]))
                sl[name][h]["tr"].append(float(pred["target_return"][hi]))

        # Free the previous model's GPU memory before loading the next one.
        del loaded
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack
    n_tta = len(tta_windows)
    n_sl = len(sl_windows)
    for name in loaded_ok:
        for h in HORIZONS:
            per_lb[name][h]["per_lb_logits"] = [
                np.stack(arr) for arr in per_lb[name][h]["per_lb_logits"]
            ]
            for k in ("td", "tr"):
                per_lb[name][h][k] = np.asarray(per_lb[name][h][k])
            for k in ("logits", "pret", "td", "tr"):
                sl[name][h][k] = np.asarray(sl[name][h][k])

    return per_lb, sl, loaded_ok, n_tta, n_sl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="688169")
    ap.add_argument(
        "--candidates",
        default=None,
        help="Comma list of exp names; if empty, use the full zoo.",
    )
    ap.add_argument("--out-prefix", default=str(ROOT / "outputs" / "p6"))
    args = ap.parse_args()

    if args.candidates:
        names = [s.strip() for s in args.candidates.split(",") if s.strip()]
    else:
        # Full zoo (exclude unsuffixed base + r12 which lacks best_model).
        names = [
            "frozen_cw_consist", "frozen_cw_dz001", "frozen_d025", "frozen_d025_cw",
            "frozen_d025_cw_long", "frozen_d025_cw_v2", "joint_d005", "joint_d025_cw",
            "r1_consist_w3", "r10_joint_splitlr", "r10v2_joint_splitlr_50stocks",
            "r11_joint_splitlr_140stocks", "r13_joint_splitlr_related25",
            "r14_frozen_pool48_688169_only", "r15_warmstart_r10_688169",
            "r16_return_head_only_r10", "r2_cw330", "r3_joint_cw_consist",
            "r4_joint_lr2e6", "r5_frozen_pool48", "r5v2_frozen_pool48_50stocks",
            "r6_frozen_pool64", "r7_frozen_pool32", "r9_joint_pool48", "smoke_frozen",
        ]

    print(f"=== Building P6 store for {len(names)} candidates on {args.symbol} ===")
    print(f"DEVICE={DEVICE}")
    per_lb, sl, loaded_ok, n_tta, n_sl = collect_all(names, args.symbol)

    per_lb_out = Path(args.out_prefix + "_per_lb_store.npz")
    sl_out = Path(args.out_prefix + "_sl_store.npz")
    save_per_lb_store(per_lb, per_lb_out, ALL_LOOKBACKS)
    save_sl_store(sl, sl_out)

    meta = {
        "symbol": args.symbol,
        "n_tta_windows": n_tta,
        "n_sl_windows": n_sl,
        "lookbacks": list(ALL_LOOKBACKS),
        "loaded_models": loaded_ok,
        "skipped_models": [n for n in names if n not in loaded_ok],
        "per_lb_out": str(per_lb_out),
        "sl_out": str(sl_out),
    }
    Path(args.out_prefix + "_store_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nLoaded {len(loaded_ok)} models; skipped {len(meta['skipped_models'])}.")
    print(f"Saved meta -> {args.out_prefix}_store_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
