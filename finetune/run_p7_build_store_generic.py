"""Phase 7 — generic per-lookback logits store builder for new checkpoints.

Differences from run_p6_build_store / run_p7_r1_build_wide_store:
  * models and lookbacks are CLI args;
  * targets are computed with a FIXED reference config (context_length=122,
    deadzone=0.003, vol_mult=0.5 — the p6-store convention) regardless of the
    candidate model's own training deadzone, so all models are scored on the
    SAME labels as the P5-8 baseline (fair A/B).

Usage:
  python finetune/run_p7_build_store_generic.py \
      --models p7_bag_s101,p7_bag_s102 --out outputs/p7_bag_per_lb_store.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import load_csv  # noqa: E402
from multihorizon_objective import make_multihorizon_targets  # noqa: E402
from run_p6_build_store import load_model, model_dir  # noqa: E402
from run_p4_r7_expanded_tta import predict_per_lookback  # noqa: E402
from run_p4_r8_h1_only_tta import save_per_lb_store  # noqa: E402
from run_tta_eval import make_tta_windows  # noqa: E402

HORIZONS = (1, 3, 5, 10)
# Fixed reference target config (== p6 store convention).
REF_CTX = 122
REF_DEADZONE = 0.003
REF_VOL_MULT = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--lookbacks", default="122,124,126,128,130,132,134")
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", default="688169")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lookbacks = tuple(int(x) for x in args.lookbacks.split(","))
    out = Path(args.out)

    df = load_csv(args.symbol)
    windows = make_tta_windows(df, lookbacks)
    print(f"{len(windows)} windows, lookbacks={lookbacks}, models={models}")
    assert len(windows) == 144

    store = {n: {h: {"per_lb_logits": [[] for _ in lookbacks], "td": [], "tr": []}
                 for h in HORIZONS} for n in models}
    t0 = time.time()
    for name in models:
        loaded = load_model(model_dir(name))
        print(f"Loaded {name} (pool={loaded.get('pool_size')}, "
              f"dz={loaded.get('min_deadzone')})")
        for i, w in enumerate(windows):
            if i % 40 == 0:
                print(f"  [{name}] {i}/{len(windows)} ({time.time() - t0:.0f}s)")
            pred = predict_per_lookback(loaded, w, lookbacks)
            # fixed-reference targets (same labels for every model)
            drop = max(lookbacks) - REF_CTX
            close_ref = w["raw_close"][drop: drop + REF_CTX + 10 + 1]
            tgt = make_multihorizon_targets(
                torch.from_numpy(close_ref).unsqueeze(0),
                context_length=REF_CTX, horizons=(1, 3, 5, 10),
                min_deadzone=REF_DEADZONE, volatility_multiplier=REF_VOL_MULT,
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

    save_per_lb_store(store, out, lookbacks)
    meta = {"symbol": args.symbol, "n_windows": len(windows),
            "lookbacks": list(lookbacks), "models": models,
            "ref_target": {"ctx": REF_CTX, "dz": REF_DEADZONE, "vol": REF_VOL_MULT},
            "out": str(out), "elapsed_sec": round(time.time() - t0, 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out} ({meta['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
