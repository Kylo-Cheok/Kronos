"""Phase 7 — Round R1: build WIDE per-lookback direction store.

Motivation: P5-1's per-horizon optimal TTA configs landed on the EDGES of
the [122,134] grid (h=1 -> leftmost, h=10 -> rightmost), a classic sign the
optimum may lie outside. Here we run GPU inference for the two production
models (r10_joint_splitlr, r5_frozen_pool48) over a 4x-wider lookback grid
lb = 104,106,...,152 (25 points) and cache per-lookback logits, so R2 can
sweep arbitrary TTA subsets with pure numpy.

Eval anchor is identical to all previous phases: 688169,
context_end_date > 2025-12-15 (144 windows), targets from each model's own
deadzone/vol config (same as p6 store builder).

Output: outputs/p7_wide_per_lb_store.npz (+ p7_wide_store_meta.json)
Key format matches run_p4_r8_h1_only_tta.save_per_lb_store so the standard
load_per_lb_store(path, WIDE_LOOKBACKS) can read it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import load_csv  # noqa: E402
from run_p6_build_store import model_dir  # noqa: E402
from run_p6_build_store import load_model  # noqa: E402  (re-exported import)
from run_p4_r7_expanded_tta import predict_per_lookback  # noqa: E402
from run_p4_r8_h1_only_tta import save_per_lb_store  # noqa: E402
from run_tta_eval import make_tta_windows  # noqa: E402

HORIZONS = (1, 3, 5, 10)
WIDE_LOOKBACKS = tuple(range(104, 153, 2))  # 104..152 step 2 -> 25 lookbacks
MODELS = ["r10_joint_splitlr", "r5_frozen_pool48"]
SYMBOL = "688169"
# Targets MUST be computed with the same context_length convention as the p6
# store (first lookback of ALL_LOOKBACKS=122): direction labels depend on the
# context via volatility-normalized deadzone. p6 used lb=122 -> we recompute
# targets explicitly with context_length=122 so labels match p6 exactly.
TARGET_CONTEXT_LB = 122


def main() -> int:
    out = ROOT / "outputs" / "p7_wide_per_lb_store.npz"
    meta_out = ROOT / "outputs" / "p7_wide_store_meta.json"

    df = load_csv(SYMBOL)
    windows = make_tta_windows(df, WIDE_LOOKBACKS)
    print(f"{len(windows)} TTA windows for {SYMBOL} "
          f"(lookbacks {WIDE_LOOKBACKS[0]}..{WIDE_LOOKBACKS[-1]})")
    assert len(windows) == 144, f"expected 144 eval windows, got {len(windows)}"

    store = {
        n: {h: {"per_lb_logits": [[] for _ in WIDE_LOOKBACKS], "td": [], "tr": []}
            for h in HORIZONS}
        for n in MODELS
    }
    t0 = time.time()
    for name in MODELS:
        loaded = load_model(model_dir(name))
        print(f"Loaded {name} (pool={loaded.get('pool_size')}, "
              f"dz={loaded.get('min_deadzone')})")
        for i, w in enumerate(windows):
            if i % 20 == 0:
                el = time.time() - t0
                print(f"  [{name}] window {i}/{len(windows)} ({el:.0f}s)")
            pred = predict_per_lookback(loaded, w, WIDE_LOOKBACKS)
            # Recompute targets with the p6 convention (context_length=122).
            max_lb = max(WIDE_LOOKBACKS)
            drop = max_lb - TARGET_CONTEXT_LB
            close122 = w["raw_close"][drop: drop + TARGET_CONTEXT_LB + 10 + 1]
            import torch as _torch
            from multihorizon_objective import make_multihorizon_targets
            tgt = make_multihorizon_targets(
                _torch.from_numpy(close122).unsqueeze(0),
                context_length=TARGET_CONTEXT_LB,
                horizons=loaded["horizons"],
                min_deadzone=loaded["min_deadzone"],
                volatility_multiplier=loaded["vol_mult"],
            )
            tdir = tgt["direction"][0].cpu().numpy()
            tret = tgt["returns"][0].cpu().numpy()
            for lb_idx, lb_logits in enumerate(pred["per_lb_logits"]):
                for hi, h in enumerate(HORIZONS):
                    store[name][h]["per_lb_logits"][lb_idx].append(lb_logits[hi])
            for hi, h in enumerate(HORIZONS):
                store[name][h]["td"].append(int(tdir[hi]))
                store[name][h]["tr"].append(float(tret[hi]))
        del loaded

    save_per_lb_store(store, out, WIDE_LOOKBACKS)
    meta = {
        "symbol": SYMBOL,
        "n_windows": len(windows),
        "lookbacks": list(WIDE_LOOKBACKS),
        "models": MODELS,
        "out": str(out),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved meta -> {meta_out} (elapsed {meta['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
