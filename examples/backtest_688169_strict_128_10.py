"""Strict, deterministic rolling-origin backtest for 688169.

This evaluates the locally fine-tuned Kronos model against the public local
Kronos-base weights with the same horizon used by the fine-tuning config:
``lookback=128`` and ``pred_len=10``.

Each origin forecasts the next ten observed sessions using only the preceding
128 rows.  The default evaluation uses the latest 15 non-overlapping target
blocks (150 observed sessions total), with every target strictly after the
fine-tuning train+validation boundary.  Adjacent forecasts share context
rows, so these are chronological evaluation blocks rather than statistically
independent samples.

The script is deliberately deterministic: both predictors use their matching
local tokenizer checkpoint, ``deterministic=True``, ``sample_count=1``, and
evaluation mode.  It scores close forecasts with MAE, RMSE, MAPE, and the
same end-point down/flat/up direction contract used by the 1/3/5/10-day
auxiliary training task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer, load_model, load_tokenizer
from model.kronos import calc_time_stamps
from finetune.multihorizon_objective import make_direction_deadzones
from finetune.multihorizon_objective import MultiHorizonForecastHead


SYMBOL = "688169"
LOOKBACK = 128
PRED_LEN = 10
MAX_CONTEXT = 512
DEFAULT_ORIGINS = 15
DEFAULT_ORIGIN_STRIDE = PRED_LEN
FINETUNE_TRAIN_RATIO = 0.8
FINETUNE_VAL_RATIO = 0.1
SEED = 688169
TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1
EVALUATED_HORIZONS = (1, 3, 5, 10)
DIRECTION_MIN_DEADZONE = 0.003
DIRECTION_VOLATILITY_MULTIPLIER = 0.5

DATA_PATH = Path(
    os.environ.get("KRONOS_BACKTEST_DATA", str(ROOT / "data" / f"{SYMBOL}_train.csv"))
)
FINETUNE_CONFIG_PATH = ROOT / "finetune_csv" / "configs" / "config_688169_daily.yaml"
DEFAULT_FINETUNED_TOKENIZER_PATH = (
    ROOT / "outputs" / "models" / "a_share_multi_tokenizer" / "checkpoints" / "best_model"
)
# The experimental multi-horizon checkpoint is intentionally opt-in through
# KRONOS_FINETUNED_MODEL.  It must beat the strict baseline before promotion.
DEFAULT_FINETUNED_MODEL_PATH = (
    ROOT / "outputs" / "models" / "a_share_multi_predictor" / "checkpoints" / "best_model"
)
FINETUNED_TOKENIZER_PATH = Path(
    os.environ.get("KRONOS_FINETUNED_TOKENIZER", str(DEFAULT_FINETUNED_TOKENIZER_PATH))
)
FINETUNED_MODEL_PATH = Path(
    os.environ.get("KRONOS_FINETUNED_MODEL", str(DEFAULT_FINETUNED_MODEL_PATH))
)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_SUFFIX = os.environ.get("KRONOS_BACKTEST_SUFFIX", "")
PREDICTIONS_PATH = OUTPUT_DIR / f"backtest_{SYMBOL}_strict_{LOOKBACK}_{PRED_LEN}{OUTPUT_SUFFIX}.csv"
METRICS_PATH = OUTPUT_DIR / f"backtest_{SYMBOL}_strict_{LOOKBACK}_{PRED_LEN}{OUTPUT_SUFFIX}.json"

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
REQUIRED_COLUMNS = ["timestamps", *FEATURE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origins",
        type=int,
        default=DEFAULT_ORIGINS,
        help=f"Number of latest target blocks to score (default: {DEFAULT_ORIGINS}).",
    )
    parser.add_argument(
        "--origin-stride",
        type=int,
        default=DEFAULT_ORIGIN_STRIDE,
        help=(
            "Rows between origins. The default is pred_len, so target blocks do not overlap "
            f"(default: {DEFAULT_ORIGIN_STRIDE})."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, such as auto, cpu, or cuda:0 (default: auto).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available")
    return requested


def seed_for_reproducibility() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest data not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Backtest data is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    if df["timestamps"].isna().any():
        raise ValueError("Backtest data contains invalid timestamps")
    if not df["timestamps"].is_monotonic_increasing:
        raise ValueError("Backtest data must be in chronological order")
    if df["timestamps"].duplicated().any():
        raise ValueError("Backtest data contains duplicate timestamps")

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Backtest data contains NaN values in model features")
    if (df["close"] <= 0).any():
        raise ValueError("Backtest data contains non-positive close prices")
    return df.reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_finetune_contract() -> Dict[str, object]:
    """Load the exact custom-CSV split contract used to create the checkpoint."""

    manifest_override = os.environ.get("KRONOS_FINETUNE_MANIFEST")
    if manifest_override:
        manifest_path = Path(manifest_override).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Fine-tuning dataset manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        lookback = int(manifest.get("lookback_window", -1))
        predict_window = int(manifest.get("predict_window", -1))
        if (lookback, predict_window) != (LOOKBACK, PRED_LEN):
            raise ValueError(
                "Backtest window does not match the fine-tuning manifest: "
                f"manifest={lookback}->{predict_window}, backtest={LOOKBACK}->{PRED_LEN}"
            )
        validation_end = manifest.get("splits", {}).get("val_end")
        if not validation_end:
            raise ValueError("Fine-tuning manifest does not declare splits.val_end")
        return {
            "path": str(manifest_path),
            "dataset_manifest": str(manifest_path),
            "train_ratio": None,
            "val_ratio": None,
            "validation_end_date": str(validation_end),
            "data_sha256": sha256_file(manifest_path),
        }

    if not FINETUNE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Fine-tuning config not found: {FINETUNE_CONFIG_PATH}")
    with FINETUNE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data_config = config.get("data", {})
    configured_path = Path(str(data_config.get("data_path", ""))).resolve()
    if configured_path != DATA_PATH.resolve():
        raise ValueError(
            "Backtest data does not match the fine-tuning data path: "
            f"config={configured_path}, backtest={DATA_PATH.resolve()}"
        )
    lookback = int(data_config.get("lookback_window", -1))
    predict_window = int(data_config.get("predict_window", -1))
    if (lookback, predict_window) != (LOOKBACK, PRED_LEN):
        raise ValueError(
            "Backtest window does not match the fine-tuning config: "
            f"config={lookback}->{predict_window}, backtest={LOOKBACK}->{PRED_LEN}"
        )
    train_ratio = float(data_config.get("train_ratio", -1))
    val_ratio = float(data_config.get("val_ratio", -1))
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Fine-tuning train/validation ratios are invalid")
    return {
        "path": str(FINETUNE_CONFIG_PATH),
        "dataset_manifest": None,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "validation_end_date": None,
        "data_sha256": sha256_file(DATA_PATH),
    }


def build_origins(
    row_count: int,
    origins: int,
    origin_stride: int,
    minimum_origin: int,
) -> List[int]:
    if origins < 1:
        raise ValueError("origins must be at least 1")
    if origin_stride < 1:
        raise ValueError("origin_stride must be at least 1")

    last_origin = row_count - PRED_LEN
    first_origin = last_origin - (origins - 1) * origin_stride
    if first_origin < max(LOOKBACK, minimum_origin):
        raise ValueError(
            "Not enough rows for the requested rolling origins: "
            f"the first origin must be at least {max(LOOKBACK, minimum_origin)}, "
            f"got {row_count}"
        )
    return [first_origin + i * origin_stride for i in range(origins)]


def load_predictor_pair(device: str) -> Dict[str, KronosPredictor]:
    public_tokenizer = load_tokenizer("Kronos-Tokenizer-base")
    public_model = load_model("Kronos-base")
    finetuned_tokenizer = KronosTokenizer.from_pretrained(str(FINETUNED_TOKENIZER_PATH))
    finetuned_model = Kronos.from_pretrained(str(FINETUNED_MODEL_PATH))

    components = [
        public_tokenizer,
        public_model,
        finetuned_tokenizer,
        finetuned_model,
    ]
    for component in components:
        component.eval()

    head_path = FINETUNED_MODEL_PATH / "multihorizon_head.pt"
    direct_head = None
    if head_path.exists():
        try:
            payload = torch.load(head_path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(head_path, map_location=device)
        direct_head = MultiHorizonForecastHead(
            int(payload["d_model"]),
            horizons=tuple(int(value) for value in payload["horizons"]),
            pool_size=int(payload["pool_size"]),
        )
        direct_head.load_state_dict(payload["state_dict"])
        direct_head.eval().to(device)

    return {
        "public_local": KronosPredictor(
            public_model, public_tokenizer, device=device, max_context=MAX_CONTEXT
        ),
        "finetuned": KronosPredictor(
            finetuned_model, finetuned_tokenizer, device=device, max_context=MAX_CONTEXT
        ),
        "finetuned_model": finetuned_model,
        "finetuned_tokenizer": finetuned_tokenizer,
        "finetuned_direct_head": direct_head,
    }


def forecast_close(
    predictor: KronosPredictor,
    context: pd.DataFrame,
    target_timestamps: pd.Series,
) -> np.ndarray:
    if len(context) != LOOKBACK or len(target_timestamps) != PRED_LEN:
        raise ValueError("Each forecast must receive exactly the configured context and target lengths")

    prediction = predictor.predict(
        df=context[FEATURE_COLUMNS].reset_index(drop=True),
        x_timestamp=context["timestamps"].reset_index(drop=True),
        y_timestamp=target_timestamps.reset_index(drop=True),
        pred_len=PRED_LEN,
        T=TEMPERATURE,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
        deterministic=True,
    )
    predicted_close = prediction["close"].to_numpy(dtype=np.float64)
    if predicted_close.shape != (PRED_LEN,) or not np.isfinite(predicted_close).all():
        raise ValueError("Predictor returned invalid close predictions")
    return predicted_close


def forecast_direct_endpoints(
    model: Kronos,
    tokenizer: KronosTokenizer,
    head: MultiHorizonForecastHead,
    context: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """Emit direct 1/3/5/10-day endpoints from observed context only."""

    if len(context) != LOOKBACK:
        raise ValueError("Direct endpoint forecast requires exactly the configured context")
    features = context[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    normalized = np.clip(
        (features - feature_mean) / (feature_std + 1e-5), -5.0, 5.0
    )
    timestamps = calc_time_stamps(context["timestamps"].reset_index(drop=True))
    device = next(model.parameters()).device
    x = torch.from_numpy(normalized).unsqueeze(0).to(device)
    stamp = torch.from_numpy(timestamps.to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(x, half=True)
        _, _, hidden_states = model(
            token_seq_0,
            token_seq_1,
            stamp,
            return_context=True,
        )
        outputs = head(hidden_states, context_length=LOOKBACK)
    predicted_returns = outputs["return_prediction"].squeeze(0).detach().cpu().numpy()
    predicted_directions = outputs["direction_logits"].argmax(dim=-1).squeeze(0).detach().cpu().numpy()
    return {
        int(horizon): {
            "log_return": float(predicted_return),
            "direction_class": int(predicted_direction),
        }
        for horizon, predicted_return, predicted_direction in zip(
            head.horizons, predicted_returns, predicted_directions
        )
    }


def metric_dict(frame: pd.DataFrame, prediction_column: str) -> Dict[str, float]:
    actual = frame["actual_close"].to_numpy(dtype=np.float64)
    predicted = frame[prediction_column].to_numpy(dtype=np.float64)
    previous_close = frame["previous_close"].to_numpy(dtype=np.float64)
    deadzone = frame["direction_deadzone"].to_numpy(dtype=np.float64)
    actual_return = np.log(actual / previous_close)
    predicted_return = np.log(predicted / previous_close)
    actual_direction = np.where(
        actual_return > deadzone,
        2,
        np.where(actual_return < -deadzone, 0, 1),
    )
    predicted_direction = np.where(
        predicted_return > deadzone,
        2,
        np.where(predicted_return < -deadzone, 0, 1),
    )
    errors = predicted - actual
    return {
        "mae_close": float(np.mean(np.abs(errors))),
        "rmse_close": float(np.sqrt(np.mean(errors**2))),
        "mape_close_pct": float(np.mean(np.abs(errors / actual)) * 100.0),
        "endpoint_direction_accuracy_pct": float(
            np.mean(actual_direction == predicted_direction) * 100.0
        ),
        "n_forecasts": int(actual.size),
    }


def collect_metrics(
    predictions: pd.DataFrame,
    prediction_column: str,
) -> Dict[str, object]:
    overall = metric_dict(predictions, prediction_column)

    by_horizon: Dict[str, Dict[str, float]] = {}
    for horizon in EVALUATED_HORIZONS:
        group = predictions[predictions["horizon"] == horizon]
        if group.empty:
            continue
        by_horizon[str(int(horizon))] = metric_dict(group, prediction_column)
    return {"overall": overall, "by_horizon": by_horizon}


def collect_direct_head_metrics(predictions: pd.DataFrame) -> Dict[str, object] | None:
    """Score direct return/direction head only at its declared endpoints."""

    available = predictions.dropna(subset=["finetuned_direct_close"])
    if available.empty:
        return None
    by_horizon: Dict[str, Dict[str, float]] = {}
    for horizon in EVALUATED_HORIZONS:
        group = available[available["horizon"] == horizon]
        if group.empty:
            continue
        summary = metric_dict(group, "finetuned_direct_close")
        actual_return = np.log(
            group["actual_close"].to_numpy(dtype=np.float64)
            / group["previous_close"].to_numpy(dtype=np.float64)
        )
        deadzone = group["direction_deadzone"].to_numpy(dtype=np.float64)
        actual_direction = np.where(
            actual_return > deadzone,
            2,
            np.where(actual_return < -deadzone, 0, 1),
        )
        direct_direction = group["finetuned_direct_direction"].to_numpy(dtype=np.int64)
        summary["direct_direction_head_accuracy_pct"] = float(
            np.mean(actual_direction == direct_direction) * 100.0
        )
        by_horizon[str(horizon)] = summary
    return {"by_horizon": by_horizon}


def run_backtest(
    df: pd.DataFrame,
    predictors: Dict[str, object],
    origins: Iterable[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for origin_index in origins:
        context = df.iloc[origin_index - LOOKBACK : origin_index].copy()
        target = df.iloc[origin_index : origin_index + PRED_LEN].copy()
        previous_close = float(context["close"].iloc[-1])

        # Only target timestamps (known calendar metadata) are sent to inference;
        # target OHLCV values are read after both model forecasts are made.
        target_timestamps = target["timestamps"].reset_index(drop=True)
        direction_deadzone = make_direction_deadzones(
            torch.as_tensor(
                context["close"].to_numpy(dtype=np.float64), dtype=torch.float64
            ).unsqueeze(0),
            context_length=LOOKBACK,
            horizons=tuple(range(1, PRED_LEN + 1)),
            min_deadzone=DIRECTION_MIN_DEADZONE,
            volatility_multiplier=DIRECTION_VOLATILITY_MULTIPLIER,
        ).squeeze(0).numpy()
        public_close = forecast_close(
            predictors["public_local"], context, target_timestamps
        )
        finetuned_close = forecast_close(
            predictors["finetuned"], context, target_timestamps
        )
        direct_head = predictors["finetuned_direct_head"]
        direct_endpoints = (
            forecast_direct_endpoints(
                predictors["finetuned_model"],
                predictors["finetuned_tokenizer"],
                direct_head,
                context,
            )
            if direct_head is not None
            else {}
        )
        actual_close = target["close"].to_numpy(dtype=np.float64)
        naive_close = np.full(PRED_LEN, previous_close, dtype=np.float64)

        for offset, (timestamp, actual, public, finetuned, naive) in enumerate(
            zip(
                target["timestamps"],
                actual_close,
                public_close,
                finetuned_close,
                naive_close,
            ),
            start=1,
        ):
            direct_endpoint = direct_endpoints.get(int(offset))
            rows.append(
                {
                    "origin_index": int(origin_index),
                    "origin_timestamp": context["timestamps"].iloc[-1].isoformat(),
                    "target_timestamp": timestamp.isoformat(),
                    "horizon": int(offset),
                    "previous_close": previous_close,
                    "actual_close": float(actual),
                    "public_local_close": float(public),
                    "finetuned_close": float(finetuned),
                    "naive_last_close": float(naive),
                    "direction_deadzone": float(direction_deadzone[offset - 1]),
                    "finetuned_direct_log_return": (
                        float(direct_endpoint["log_return"])
                        if direct_endpoint is not None
                        else np.nan
                    ),
                    "finetuned_direct_close": (
                        float(previous_close * np.exp(direct_endpoint["log_return"]))
                        if direct_endpoint is not None
                        else np.nan
                    ),
                    "finetuned_direct_direction": (
                        int(direct_endpoint["direction_class"])
                        if direct_endpoint is not None
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    origins: List[int],
    origin_stride: int,
    device: str,
    finetune_split_end: int,
    finetune_contract: Dict[str, object],
) -> Dict[str, object]:
    return {
        "evaluation": {
            "symbol": SYMBOL,
            "data_path": str(DATA_PATH),
            "data_rows": int(len(df)),
            "data_start": df["timestamps"].iloc[0].date().isoformat(),
            "data_end": df["timestamps"].iloc[-1].date().isoformat(),
            "lookback": LOOKBACK,
            "pred_len": PRED_LEN,
            "origins": len(origins),
            "origin_stride": origin_stride,
            "target_rows": int(len(predictions)),
            "target_start": predictions["target_timestamp"].iloc[0],
            "target_end": predictions["target_timestamp"].iloc[-1],
            "target_blocks_overlap": bool(origin_stride < PRED_LEN),
            "target_values_used_for_context": False,
            "finetune_config": finetune_contract,
            "finetune_train_ratio": finetune_contract.get("train_ratio"),
            "finetune_val_ratio": finetune_contract.get("val_ratio"),
            "finetune_validation_end_date": finetune_contract.get("validation_end_date"),
            "finetune_train_validation_end_index": finetune_split_end,
            "targets_after_finetune_validation": bool(
                all(origin >= finetune_split_end for origin in origins)
            ),
            "device": device,
            "max_context": MAX_CONTEXT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "sample_count": SAMPLE_COUNT,
            "deterministic": True,
            "public_tokenizer": "weights/Kronos-Tokenizer-base",
            "public_model": "weights/Kronos-base",
            "finetuned_tokenizer": str(FINETUNED_TOKENIZER_PATH),
            "finetuned_model": str(FINETUNED_MODEL_PATH),
            "finetuned_direct_head": str(FINETUNED_MODEL_PATH / "multihorizon_head.pt"),
            "direction_definition": (
                "For each 1/3/5/10-day endpoint, compare its log return from the "
                "last observed close against a past-only volatility-aware dead zone; "
                "classes are down, flat, and up."
            ),
        },
        "metrics": {
            "public_local": collect_metrics(predictions, "public_local_close"),
            "finetuned": collect_metrics(predictions, "finetuned_close"),
            "finetuned_direct_head": collect_direct_head_metrics(predictions),
            "naive_last_close": collect_metrics(predictions, "naive_last_close"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.origin_stride != DEFAULT_ORIGIN_STRIDE:
        print(
            "Warning: origin stride differs from pred_len; target blocks may overlap and "
            "the reported run is no longer the default non-overlapping evaluation."
        )

    device = resolve_device(args.device)
    seed_for_reproducibility()
    finetune_contract = load_finetune_contract()
    df = load_data(DATA_PATH)
    if finetune_contract.get("validation_end_date"):
        validation_end = pd.Timestamp(str(finetune_contract["validation_end_date"]))
        finetune_split_end = int(
            df["timestamps"].searchsorted(validation_end, side="right")
        )
    else:
        finetune_split_end = int(
            len(df)
            * (
                float(finetune_contract["train_ratio"])
                + float(finetune_contract["val_ratio"])
            )
        )
    origin_indices = build_origins(
        len(df), args.origins, args.origin_stride, finetune_split_end
    )

    print(
        f"Loading local checkpoints | device={device} | lookback={LOOKBACK} "
        f"pred_len={PRED_LEN} | origins={len(origin_indices)}"
    )
    predictors = load_predictor_pair(device)
    predictions = run_backtest(df, predictors, origin_indices)

    report = build_report(
        df,
        predictions,
        origin_indices,
        args.origin_stride,
        device,
        finetune_split_end,
        finetune_contract,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".strict_backtest_", dir=OUTPUT_DIR) as temp_dir:
        temp_dir_path = Path(temp_dir)
        staged_predictions = temp_dir_path / PREDICTIONS_PATH.name
        staged_metrics = temp_dir_path / METRICS_PATH.name
        predictions.to_csv(staged_predictions, index=False)
        staged_metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(staged_predictions, PREDICTIONS_PATH)
        os.replace(staged_metrics, METRICS_PATH)

    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
