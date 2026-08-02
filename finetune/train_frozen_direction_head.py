"""Train and audit a 10-day direction head on frozen Kronos representations.

The script deliberately separates representation extraction, chronological model
selection, probability calibration, and target-symbol testing. A rejected model
is still saved as an experiment artifact, but must not be used as actionable UI
output.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from finetune.direction_training import (
    build_direction_sample_index,
    fit_temperature_scaling,
    split_direction_sample_index,
    summarize_direction_probabilities,
)
from finetune.frozen_direction_head import (
    FrozenDirectionHead,
    assess_direction_candidate,
    prepare_context_arrays,
    select_non_overlapping_samples,
)
from model import load_model, load_tokenizer


DEFAULT_SYMBOLS = [
    "688001",
    "688002",
    "688003",
    "688005",
    "688006",
    "688007",
    "688008",
    "688009",
    "688010",
    "688011",
    "688012",
    "688015",
    "688016",
    "688018",
    "688019",
    "688020",
    "688021",
    "688022",
    "688028",
    "688029",
    "688033",
    "688066",
    "688088",
    "688099",
    "688122",
    "688169",
    "688333",
    "688388",
]

DEFAULT_FOLDS = [
    {
        "name": "problem_2022_h2",
        "train_end": "2021-12-31",
        "validation_end": "2022-06-30",
        "test_end": "2022-12-31",
    },
    {
        "name": "mid_2024",
        "train_end": "2022-12-31",
        "validation_end": "2023-12-31",
        "test_end": "2024-12-31",
    },
    {
        "name": "recent_2025_2026",
        "train_end": "2023-12-31",
        "validation_end": "2024-12-31",
        "test_end": "2026-12-31",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--target-symbol", default="688169")
    parser.add_argument("--start-date", default="20190701")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/direction_universe"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/direction_head"))
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--validation-stride", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--reuse-embeddings", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_market_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = raw.rename(columns={column: str(column).lower() for column in raw.columns})
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: missing Tencent fields {sorted(missing)}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    valid_ohlc = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame[["volume", "amount"]] >= 0).all(axis=1)
    )
    invalid = int((~valid_ohlc).sum())
    if invalid:
        frame = frame[valid_ohlc].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{symbol}: no valid rows after cleaning")
    return frame


def load_or_download_frames(
    symbols: list[str],
    *,
    data_dir: Path,
    start_date: str,
    end_date: str,
    refresh: bool,
    min_rows: int,
) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(symbols, start=1):
        cache_path = data_dir / f"{symbol}_qfq.csv"
        if cache_path.exists() and not refresh:
            frame = _sanitize_market_frame(pd.read_csv(cache_path), symbol)
            source = "cache"
        else:
            import akshare as ak

            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    raw = ak.stock_zh_a_hist_tx(
                        symbol=f"sh{symbol}",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )
                    frame = _sanitize_market_frame(raw, symbol)
                    frame.to_csv(cache_path, index=False)
                    last_error = None
                    break
                except Exception as exc:  # network endpoint is occasionally flaky
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2.0 * (attempt + 1))
            if last_error is not None:
                print(f"[{position}/{len(symbols)}] skip {symbol}: {last_error}")
                continue
            source = "download"
        if len(frame) < min_rows:
            print(f"[{position}/{len(symbols)}] skip {symbol}: only {len(frame)} rows")
            continue
        frames[symbol] = frame
        print(
            f"[{position}/{len(symbols)}] {symbol}: {len(frame)} rows "
            f"{frame['date'].min().date()}..{frame['date'].max().date()} ({source})"
        )
    return frames


def extract_frozen_embeddings(
    frames: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = load_tokenizer("Kronos-Tokenizer-base").to(device).eval()
    model = load_model("Kronos-base").to(device).eval()
    embedding_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_start in range(0, len(samples), batch_size):
            batch = samples.iloc[batch_start : batch_start + batch_size]
            values: list[np.ndarray] = []
            stamps: list[np.ndarray] = []
            technical: list[np.ndarray] = []
            for row in batch.itertuples(index=False):
                context_values, context_stamps, context_technical = prepare_context_arrays(
                    frames[row.symbol],
                    int(row.context_start),
                    int(row.context_end),
                )
                values.append(context_values)
                stamps.append(context_stamps)
                technical.append(context_technical)
            value_tensor = torch.from_numpy(np.stack(values)).to(device)
            stamp_tensor = torch.from_numpy(np.stack(stamps)).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                token_s1, token_s2 = tokenizer.encode(value_tensor, half=True)
                _, context = model.decode_s1(token_s1, token_s2, stamp_tensor)
                kronos_features = torch.cat(
                    [context[:, -1, :], context[:, -10:, :].mean(dim=1)], dim=1
                )
            batch_features = np.concatenate(
                [
                    kronos_features.float().cpu().numpy(),
                    np.stack(technical).astype(np.float32),
                ],
                axis=1,
            )
            embedding_batches.append(batch_features)
            completed = min(batch_start + len(batch), len(samples))
            if completed == len(samples) or completed % (batch_size * 10) == 0:
                print(f"embedding {completed}/{len(samples)}")

    del tokenizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(embedding_batches, axis=0)


def _make_head(architecture: str, input_dim: int) -> nn.Module:
    if architecture == "linear":
        return nn.Linear(input_dim, 1)
    if architecture == "mlp128":
        return FrozenDirectionHead(input_dim, hidden_dim=128, dropout=0.2)
    if architecture == "mlp256":
        return FrozenDirectionHead(input_dim, hidden_dim=256, dropout=0.3)
    raise ValueError(f"unknown architecture: {architecture}")


def _head_logits(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 2048):
            tensor = torch.from_numpy(features[start : start + 2048]).to(device)
            logits = model(tensor)
            if logits.ndim == 2:
                logits = logits.squeeze(-1)
            output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def train_candidate(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    model = _make_head(architecture, train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=False)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    validation_target = torch.from_numpy(validation_y.astype(np.float32)).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            if logits.ndim == 2:
                logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_y, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_tensor)
            if validation_logits.ndim == 2:
                validation_logits = validation_logits.squeeze(-1)
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    validation_logits, validation_target
                ).item()
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_logits = _head_logits(model, validation_x, device)
    calibration = fit_temperature_scaling(validation_logits, validation_y)
    validation_summary = summarize_direction_probabilities(
        calibration["calibrated_probabilities"], validation_y
    )
    return {
        "architecture": architecture,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "temperature": float(calibration["temperature"]),
        "validation_log_loss": float(calibration["calibrated_log_loss"]),
        "validation_summary": validation_summary,
    }


def _scaled_subset(
    embeddings: np.ndarray,
    subset: pd.DataFrame,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    features = scaler.transform(embeddings[subset["embedding_index"].to_numpy()]).astype(
        np.float32
    )
    targets = subset["target_up"].to_numpy(dtype=np.float32)
    return features, targets


def _momentum_probabilities(
    subset: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> np.ndarray:
    probabilities: list[float] = []
    for row in subset.itertuples(index=False):
        close = frames[row.symbol]["close"].to_numpy(dtype=float)
        context_end = int(row.context_end)
        reference = max(0, context_end - 20)
        probabilities.append(0.55 if close[context_end] > close[reference] else 0.45)
    return np.asarray(probabilities)


def run_fold(
    fold: dict[str, str],
    samples: pd.DataFrame,
    embeddings: np.ndarray,
    frames: dict[str, pd.DataFrame],
    *,
    target_symbol: str,
    horizon: int,
    train_stride: int,
    validation_stride: int,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    split = split_direction_sample_index(
        samples,
        train_end=fold["train_end"],
        validation_end=fold["validation_end"],
        test_end=fold["test_end"],
        test_symbol=target_symbol,
    )
    train = split["train"]
    validation = split["validation"]
    test = split["test"]
    train = train[train["context_end"] % train_stride == 0].reset_index(drop=True)
    validation = validation[
        validation["context_end"] % validation_stride == 0
    ].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"{fold['name']} has an empty chronological split")

    scaler = StandardScaler().fit(embeddings[train["embedding_index"].to_numpy()])
    train_x, train_y = _scaled_subset(embeddings, train, scaler)
    validation_x, validation_y = _scaled_subset(embeddings, validation, scaler)
    test_x, test_y = _scaled_subset(embeddings, test, scaler)

    candidates = []
    for architecture in ("linear", "mlp128", "mlp256"):
        candidate = train_candidate(
            architecture,
            train_x,
            train_y,
            validation_x,
            validation_y,
            device=device,
            epochs=epochs,
            patience=patience,
            seed=seed,
        )
        summary = candidate["validation_summary"]
        candidate["selection_score"] = float(
            summary["balanced_accuracy"]
            + 0.25 * summary["accuracy"]
            - 0.25 * summary["brier_score"]
        )
        candidates.append(candidate)
        print(
            f"{fold['name']} {architecture}: val_acc={summary['accuracy']:.3f} "
            f"val_bal={summary['balanced_accuracy']:.3f} "
            f"val_brier={summary['brier_score']:.3f}"
        )
    selected = max(candidates, key=lambda candidate: candidate["selection_score"])
    model = _make_head(selected["architecture"], train_x.shape[1]).to(device)
    model.load_state_dict(selected["state_dict"])
    logits = _head_logits(model, test_x, device)
    probabilities = expit(logits / selected["temperature"])
    test_summary = summarize_direction_probabilities(probabilities, test_y)

    independent = select_non_overlapping_samples(test, horizon=horizon)
    independent_positions = test.index.get_indexer(independent.index)
    # select_non_overlapping_samples resets its index, so match by stable embedding id.
    position_by_embedding = {
        int(embedding_index): position
        for position, embedding_index in enumerate(test["embedding_index"])
    }
    independent_positions = np.asarray(
        [position_by_embedding[int(value)] for value in independent["embedding_index"]]
    )
    independent_summary = summarize_direction_probabilities(
        probabilities[independent_positions], test_y[independent_positions]
    )
    acceptance = assess_direction_candidate(test_summary, independent_summary)
    momentum_summary = summarize_direction_probabilities(
        _momentum_probabilities(test, frames), test_y
    )

    fold_result = {
        **fold,
        "train_points": len(train),
        "validation_points": len(validation),
        "test_points": len(test),
        "selected_architecture": selected["architecture"],
        "temperature": selected["temperature"],
        "validation": selected["validation_summary"],
        "test": test_summary,
        "independent_test": independent_summary,
        "momentum20_test": momentum_summary,
        "acceptance": acceptance,
    }
    checkpoint = {
        "architecture": selected["architecture"],
        "input_dim": train_x.shape[1],
        "state_dict": selected["state_dict"],
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "temperature": selected["temperature"],
        "fold": fold_result,
    }
    return fold_result, checkpoint


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items() if key != "state_dict"}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()))
    if args.target_symbol not in symbols:
        symbols.append(args.target_symbol)
    frames = load_or_download_frames(
        symbols,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh_data,
        min_rows=args.lookback + args.horizon + 250,
    )
    if args.target_symbol not in frames:
        raise RuntimeError(f"target symbol {args.target_symbol} was not loaded")
    print(f"usable universe: {len(frames)} symbols")

    samples = build_direction_sample_index(
        frames, lookback=args.lookback, horizon=args.horizon, stride=1
    )
    samples["embedding_index"] = np.arange(len(samples), dtype=np.int64)
    embedding_path = args.output_dir / "frozen_embeddings.npy"
    sample_path = args.output_dir / "sample_index.csv"
    if args.reuse_embeddings and embedding_path.exists() and sample_path.exists():
        cached_samples = pd.read_csv(sample_path)
        embeddings = np.load(embedding_path)
        if len(cached_samples) != len(samples) or len(embeddings) != len(samples):
            raise ValueError("cached embeddings do not match the current sample index")
        print(f"reused {len(embeddings)} frozen embeddings")
    else:
        device = torch.device(args.device)
        embeddings = extract_frozen_embeddings(
            frames, samples, device=device, batch_size=args.batch_size
        )
        np.save(embedding_path, embeddings)
        samples.to_csv(sample_path, index=False)
        print(f"saved embeddings: {embeddings.shape}")

    device = torch.device(args.device)
    fold_results: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for fold in DEFAULT_FOLDS:
        fold_result, checkpoint = run_fold(
            fold,
            samples,
            embeddings,
            frames,
            target_symbol=args.target_symbol,
            horizon=args.horizon,
            train_stride=args.train_stride,
            validation_stride=args.validation_stride,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device=device,
        )
        fold_results.append(fold_result)
        checkpoints.append(checkpoint)
        print(
            f"{fold['name']} TEST acc={fold_result['test']['accuracy']:.3f} "
            f"bal={fold_result['test']['balanced_accuracy']:.3f} "
            f"brier={fold_result['test']['brier_score']:.3f} "
            f"independent={fold_result['independent_test']['accuracy']:.3f} "
            f"accepted={fold_result['acceptance']['accepted']}"
        )

    recent_accepted = bool(fold_results[-1]["acceptance"]["accepted"])
    accepted_folds = sum(bool(result["acceptance"]["accepted"]) for result in fold_results)
    stable_acceptance = recent_accepted and accepted_folds >= 2
    report = {
        "target_symbol": args.target_symbol,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "universe_symbols": sorted(frames),
        "sample_points": len(samples),
        "embedding_dim": int(embeddings.shape[1]),
        "folds": fold_results,
        "accepted_folds": accepted_folds,
        "stable_acceptance": stable_acceptance,
        "webui_actionable": stable_acceptance,
    }
    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    recent_checkpoint = checkpoints[-1]
    recent_checkpoint["stable_acceptance"] = stable_acceptance
    torch.save(recent_checkpoint, args.output_dir / "candidate_direction_head.pt")
    print(f"report: {report_path}")
    print(f"stable acceptance: {stable_acceptance} ({accepted_folds}/3 folds)")


if __name__ == "__main__":
    main()
