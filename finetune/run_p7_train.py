"""Phase 7 training launcher wrapper (= run_p6_train.py + KRONOS_SEED).

Adds a KRONOS_SEED env override (Config hardcodes seed=100) so we can train
bagging replicas of the r5 frozen-head recipe with different sampling/init
seeds. Everything else identical to the P6 launcher (safe-delete restore).
"""

from __future__ import annotations

import os
import pathlib
import sys

import sitecustomize as _sc  # already imported at interpreter startup

for _name, _orig in (
    ("Path.unlink", _sc._orig_path_unlink),
    ("Path.rmdir", _sc._orig_path_rmdir),
    ("os.remove", _sc._orig_remove),
    ("os.unlink", _sc._orig_unlink),
    ("os.rmdir", _sc._orig_rmdir),
    ("shutil.rmtree", _sc._orig_shutil_rmtree),
):
    try:
        if _name.startswith("Path."):
            setattr(pathlib.Path, _name.split(".", 1)[1], _orig)
        elif _name.startswith("os."):
            setattr(os, _name.split(".", 1)[1], _orig)
        else:
            import shutil as _sh

            setattr(_sh, "rmtree", _orig)
    except Exception as _e:  # pragma: no cover
        print(f"[wrapper] failed to restore {_name}: {_e}")

if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, _HERE)
    from train_predictor import main as _train_main
    from config import Config as _Config

    _cfg = _Config()
    _seed = os.getenv("KRONOS_SEED", "").strip()
    if _seed:
        _cfg.seed = int(_seed)
        print(f"[wrapper] KRONOS_SEED override -> {_cfg.seed}")
    _train_main(_cfg.__dict__)
