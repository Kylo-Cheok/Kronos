"""One-off verification: forward_logits reproduces p6-cache logits exactly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from evaluate_gated_ensemble import FEATURES, load_csv, load_model, make_windows  # noqa: E402
from promoted_config import model_dir  # noqa: E402
from run_p4_r8_h1_only_tta import load_per_lb_store  # noqa: E402
from run_p4_r7_expanded_tta import ALL_LOOKBACKS  # noqa: E402
from predict_688169_promoted import forward_logits  # noqa: E402

df = load_csv("688169")
ws = make_windows(df)
last = ws[-1]
end_pos = int(df.index[df["timestamps"] == pd.Timestamp(last["context_end_date"])][0])
sub = df.iloc[: end_pos + 1]
mod = load_model(model_dir("r10_joint_splitlr"))
out = forward_logits(mod, sub[FEATURES].to_numpy(dtype=np.float32),
                     sub["timestamps"], 128)
p6 = load_per_lb_store(ROOT / "outputs" / "p6_per_lb_store.npz", ALL_LOOKBACKS)
ref = p6["r10_joint_splitlr"][1]["per_lb_logits"][ALL_LOOKBACKS.index(128)][-1]
print("context_end:", last["context_end_date"])
print("equal:", np.allclose(out["logits"][0], ref),
      "| max diff %.2e" % np.abs(out["logits"][0] - ref).max())
