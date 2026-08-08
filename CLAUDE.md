# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kronos is a decoder-only foundation model for financial K-line (OHLCV) sequences: a hierarchical tokenizer (BSQ quantization → s1/s2 discrete tokens) plus an autoregressive Transformer. The active work in this repo (branch `feat-improve-kylo`) is the **A-share multi-horizon finetune pipeline**: training a `MultiHorizonForecastHead` on top of a Kronos backbone for 1/3/5/10-trading-day direction + return forecasts, plus a large body of inference-level evaluation experiments (phases P4–P9).

## Commands

Python 3.10+; dependencies in `requirements.txt`. All weights are local under `weights/` — `model/__init__.py` forces `HF_HUB_OFFLINE=1`, so nothing ever downloads from Hugging Face.

- Run all tests: `pytest` (`pytest.ini` → `testpaths = tests`). CPU-only unit tests with synthetic fixtures; the regression test is the exception — it loads real local weights, so `weights/` must be present.
- Run a single test: `pytest tests/test_kronos_regression.py`
- Web UI: `cd webui && python run.py` → http://localhost:7070
- Finetune predictor (multi-GPU): `torchrun --standalone --nproc_per_node=N finetune/train_predictor.py`
- Finetune predictor (single GPU / Windows): `KRONOS_SINGLE_PROCESS=1 python finetune/train_predictor.py`
- Finetune tokenizer: `torchrun --standalone --nproc_per_node=N finetune/train_tokenizer.py`
- Build finetune dataset from cached CSVs: `python data/build_local_finetune_dataset.py`
- Download fresh A-share data: `python data/prepare_a_share_finetune.py` (needs network; AkShare/adata provider chain)
- Qlib demo pipeline (upstream, less active): `python finetune/qlib_data_preprocess.py`, `python finetune/qlib_test.py --device cuda:0`

## Architecture

### model/ — core Kronos
- `kronos.py`: `KronosTokenizer` (encoder → BSQ → decoder; `encode`/`decode` with `half=True` splits the codebook into s1/s2 token streams), `Kronos` (HierarchicalEmbedding + TemporalEmbedding + TransformerBlock stack + DependencyAwareLayer + DualHead; inference via `decode_s1`/`decode_s2`), `KronosPredictor` (normalize → `auto_regressive_inference` → denormalize; `predict`, `predict_batch`; `return_distribution` gives sample summaries with confidence intervals and up-probabilities).
- `module.py`: `BinarySphericalQuantizer`/`BSQuantizer` (sign quantization with entropy loss), `TransformerBlock` (pre-norm RMSNorm + RoPE + SwiGLU), `HierarchicalEmbedding`, `TemporalEmbedding`, `DependencyAwareLayer` (s2 conditioned on s1), `DualHead` (`compute_loss` = averaged s1+s2 cross-entropy).
- `__init__.py`: forces offline mode, maps model names (`Kronos-base`, `Kronos-small`, ...) to `weights/` directories via `load_model`/`load_tokenizer`.

### finetune/ — the active A-share pipeline
- `config.py` — one `Config` class; every tunable is overridable via `KRONOS_*` env vars: `KRONOS_DATASET_PATH`, `KRONOS_PREDICTOR_LR`, `KRONOS_HEAD_LR`, `KRONOS_JOINT_HEAD_LR`, `KRONOS_FREEZE_BACKBONE`, `KRONOS_EXP_NAME` (appends experiment tag to save folder), `KRONOS_INIT_PREDICTOR_FROM`/`KRONOS_INIT_HEAD_FROM` (warm-start backbone/head), `KRONOS_SYMBOL_FILTER`, `KRONOS_FORECAST_POOL_SIZE`, `KRONOS_DIRECTION_LOSS_WEIGHT`/`MIN_DEADZONE`/`VOL_MULT`/`CLASS_WEIGHTS`/`HORIZON_WEIGHTS`, `KRONOS_CONSISTENCY_LOSS_WEIGHT`, `KRONOS_FINETUNE_EPOCHS`/`BATCH_SIZE`.
- `dataset.py` — `QlibDataset` over train/val pickles; **validates `manifest.json` against config** (lookback_window=128, predict_window=10, window=139, symbol list) and raises on mismatch. Normalization stats come strictly from the past lookback window (no future leakage); raw closes are returned as a labels-only side channel.
- `train_predictor.py` — DDP training loop with two modes: `freeze_backbone=1` (only the head trains, at `head_learning_rate`; `KRONOS_TRAIN_RETURN_HEAD_ONLY=1` narrows further) and joint (backbone at low `predictor_learning_rate`, head at `KRONOS_JOINT_HEAD_LR`). Checkpoint selection is by **lowest val loss**; saves backbone to `best_model/` plus `multihorizon_head.pt` (with horizon/pool metadata).
- `multihorizon_objective.py` — `MultiHorizonForecastHead` (pool last `pool_size` causal states → trunk → `return_head` + `direction_head`) and `compute_multihorizon_objective`: Huber return loss + 3-class (down/flat/up) cross-entropy at horizons (1,3,5,10), optional sign-consistency loss. Labels are log-returns of raw close only; the direction deadzone is `max(daily_vol·√h·vol_mult, min_deadzone)` — **`vol_mult` is the effective lever; `min_deadzone` is masked by the dynamic term**.
- Eval/support modules: `evaluate_multihorizon.py`, `selective_prediction.py`, `dual_metric_compare.py`, `calibrate_returns.py`, `exogenous_direction_inference.py`, `promoted_config.py` (see Experiment workflow).

### finetune_csv/ — older single-symbol pipeline
Tokenizer-then-predictor finetune on arbitrary CSV (columns `timestamps, open, high, low, close, volume, amount`), driven by YAML configs (`config_loader.py`, `train_sequential.py`; examples in `configs/`). Superseded by `finetune/` for A-share work but still functional; outputs to `outputs/finetuned/{exp_name}`.

### webui/ — Flask app (port 7070)
`app.py` routes: `/api/data-files`, `/api/load-data`, `/api/load-model`, `/api/predict`, `/api/diagnostics/ab`, `/api/model-status`. `run.py` installs deps then launches. Helpers: `diagnostics.py` (forecast stability/accuracy), `data_quality.py` (ex-right price-factor gap adjustment), `interval_calibration.py` (empirical return-band guardrails).

### data/ — dataset build
`build_local_finetune_dataset.py` (offline, from cached CSVs in `data/direction_universe`) and `prepare_a_share_finetune.py` (live multi-board download) both produce the pickle + `manifest.json` layout that `QlibDataset` consumes (current default: `data/a_share_finetune_multiboard`, 140 symbols).

## Experiment workflow (phases P4–P9)

The eval experiments follow a strict naming/storage convention:
- Scripts: `finetune/run_p{phase}_r{round}_{topic}.py`. Training wrappers (`run_p6_train.py`, `run_p7_train.py`) call `train_predictor.main()` with env-var overrides; eval scripts are standalone loops (numpy-only or GPU) that load checkpoints from `outputs/models/` or cached logit stores (`outputs/*_store.npz`). Cross-importing utilities between `run_pX_rY_*` scripts is the established pattern.
- Results: round notes → `outputs/optimization_logs/phase{phase}_*.md`; eval metrics → `outputs/eval_p{phase}_*.json`.
- Model zoo: `outputs/models/a_share_multihorizon_predictor_*/checkpoints/best_model/` (~25 checkpoints, r1–r16 plus p6/p7 variants). Production per-horizon mapping (P5–8): h=1,10 → `r10_joint_splitlr`; h=3,5 → `r5_frozen_pool48`.
- **Reference label convention (P7 rule)**: any self-built cache must recompute direction labels with the fixed `ctx=122 / dz=0.003 / vol=0.5` — TTA lookback shifts pollute label calibration and produce fake ±0.6pt metric diffs.
- **Evaluation hygiene**: the test band is touched once; walk-forward selection bands are used for model/TTA choice instead. `run_p9_panel_eval.py` is the current honest cross-sectional panel standard, with cash-baseline benchmarks for gates.

## Gotchas

- **Windows + DDP**: training scripts require `torchrun`; for single-GPU use `KRONOS_SINGLE_PROCESS=1` to skip process-group init.
- **Offline**: never reference HF Hub names for weights — use `model.load_model`/`load_tokenizer` or local paths under `weights/`.
- **Dataset contract**: regenerating pickles requires matching `manifest.json` window params, or `QlibDataset` raises at construction time.
- **Leakage rules**: normalization stats only from the lookback window; raw closes are labels-only; anything used for model selection must come from the selection band, never the test band.
- Experiment scripts are one-off research scripts, not a framework — match their existing import and output-path conventions rather than inventing new ones.
