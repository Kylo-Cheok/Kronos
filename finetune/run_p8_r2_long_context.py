"""Phase 8 — Round R2: long-context inference (192/256/384 vs 128).

Everything so far fed the head a 128-bar context (TTA only shifted +-6/+-24
days). The Kronos backbone supports up to 512 tokens, and the forecast head
pools over context states — so feeding MORE history at inference is an
untried, retrain-free lever. Distribution shift exists (head trained at 128),
which is exactly what this round measures.

For each context C in {128 (sanity), 192, 256, 384} x model {r10, r5}:
  * windows = same 144 eval targets (context_end_date > 2025-12-15);
  * single-pass logits at context C;
  * targets recomputed with the fixed reference (ctx=122, dz=0.003,
    vol_mult=0.5) -> labels bit-identical to the p6 cache;
  * report per-horizon nonflat vs baseline single-128 and vs promoted TTA.

Output: outputs/p8_longctx_store.npz + prints; eval summary in
outputs/eval_p8_r2_long_context.json
"""

from __future__ import annotations

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
from run_p4_r7_expanded_tta import predict_per_lookback, ALL_LOOKBACKS  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r7_expanded_tta import average_logits  # noqa: E402
from run_p6_eval_zoo import P5_TTA_BY_H, P5_BASE_MAP  # noqa: E402
from run_tta_eval import make_tta_windows  # noqa: E402
from selective_prediction import (  # noqa: E402
    direction_confidence_from_logits,
    nonflat_accuracy,
)

HORIZONS = (1, 3, 5, 10)
MODELS = ["r10_joint_splitlr", "r5_frozen_pool48"]
CONTEXTS = (128, 192, 256, 384)
SYMBOL = "688169"
REF_CTX, REF_DZ, REF_VOL = 122, 0.003, 0.5


def main() -> int:
    df = load_csv(SYMBOL)
    result = {"contexts": {}, "sanity": {}}
    t0 = time.time()

    for C in CONTEXTS:
        windows = make_tta_windows(df, (C,))
        print(f"C={C}: {len(windows)} windows")
        assert len(windows) == 144, (C, len(windows))
        for name in MODELS:
            loaded = load_model(model_dir(name))
            logits_by_h = {h: [] for h in HORIZONS}
            td_all, tr_all = [], []
            for i, w in enumerate(windows):
                pred = predict_per_lookback(loaded, w, (C,))
                lg = pred["per_lb_logits"][0]  # [H,3]
                # fixed-reference targets (slice ending at context_end):
                # context_end sits at row C-1 of this window; ref needs
                # 122+11 rows ending 10 rows after context_end.
                close_ref = w["raw_close"][(C - REF_CTX):(C + 11)]
                tgt = make_multihorizon_targets(
                    torch.from_numpy(close_ref).unsqueeze(0),
                    context_length=REF_CTX, horizons=(1, 3, 5, 10),
                    min_deadzone=REF_DZ, volatility_multiplier=REF_VOL)
                td_all.append(tgt["direction"][0].cpu().numpy())
                tr_all.append(tgt["returns"][0].cpu().numpy())
                for hi, h in enumerate(HORIZONS):
                    logits_by_h[h].append(lg[hi])
            del loaded
            # store sanity: C=128 must match p6 single_128 logits
            if C == 128:
                p6 = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz",
                                       ALL_LOOKBACKS)
                ref = p6[name][1]["per_lb_logits"][ALL_LOOKBACKS.index(128)]
                same = np.allclose(np.asarray(logits_by_h[1]), ref)
                result["sanity"][name] = bool(same)
                print(f"  sanity {name} C=128 == p6 single_128: {same}")
                assert same
            result["contexts"].setdefault(str(C), {})[name] = {
                "logits": {str(h): np.asarray(logits_by_h[h]).tolist() for h in HORIZONS},
            }
        # targets identical across models/contexts (same ref slice); keep last
        result["contexts"][str(C)]["targets"] = {
            "td": np.asarray(td_all).tolist(), "tr": np.asarray(tr_all).tolist()}

    out = ROOT / "outputs" / "p8_longctx_store.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"Saved {out} ({time.time() - t0:.0f}s)")

    # quick report
    print("\n=== long-context single-pass nonflat ===")
    for C in CONTEXTS:
        td = np.asarray(result["contexts"][str(C)]["targets"]["td"])
        for name in MODELS:
            row = []
            for h in HORIZONS:
                lg = np.asarray(result["contexts"][str(C)][name]["logits"][str(h)])
                row.append(nonflat_accuracy(
                    direction_confidence_from_logits(lg)["hard_pred"], td[:, HORIZONS.index(h)]))
            print(f"C={C:3d} {name:22s} " +
                  " ".join(f"h{h}={v:.2%}" for h, v in zip(HORIZONS, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
