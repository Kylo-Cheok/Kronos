import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from webui.diagnostics import (
        build_ab_cases,
        build_direction_signal,
        evaluate_direction_signals,
        summarize_forecast_diagnostics,
    )
    from webui.interval_calibration import apply_return_band_guardrail
except ImportError:
    from diagnostics import (
        build_ab_cases,
        build_direction_signal,
        evaluate_direction_signals,
        summarize_forecast_diagnostics,
    )
    from interval_calibration import apply_return_band_guardrail

try:
    from webui.data_quality import adjust_corporate_action_gaps
except ImportError:
    from data_quality import adjust_corporate_action_gaps

try:
    from finetune.exogenous_direction_inference import predict_live_direction
except ImportError:
    predict_live_direction = None

try:
    from model import Kronos, KronosTokenizer, KronosPredictor, WEIGHTS
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    WEIGHTS = {}
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None


def get_verified_direction(origin_date, file_path):
    """Return the audited one-day target signal or an explicit abstention."""
    if not os.path.basename(str(file_path)).startswith('688169'):
        return {
            'status': 'abstain',
            'direction': None,
            'candidate_direction': None,
            'horizon': 1,
            'reason': 'verified_direction_model_is_target_specific',
            'probability_is_calibrated': False,
        }
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_path = os.path.join(
        project_root, 'weights', 'exogenous_direction_h1.joblib'
    )
    if predict_live_direction is None or not os.path.isfile(artifact_path):
        return {
            'status': 'abstain',
            'direction': None,
            'candidate_direction': None,
            'horizon': 1,
            'reason': 'verified_direction_model_unavailable',
            'probability_is_calibrated': False,
        }
    try:
        return predict_live_direction(
            artifact_path,
            stock_data_dir=os.path.join(project_root, 'data', 'direction_universe'),
            exogenous_data_dir=os.path.join(project_root, 'data', 'exogenous'),
            as_of=pd.Timestamp(origin_date),
        )
    except Exception as exc:
        print(f"Verified direction inference failed: {exc}")
        return {
            'status': 'abstain',
            'direction': None,
            'candidate_direction': None,
            'horizon': 1,
            'reason': 'verified_direction_inference_failed',
            'probability_is_calibrated': False,
        }

# Available model configurations – uses local weights from D:\workspace\Kronos\weights
AVAILABLE_MODELS = {
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': WEIGHTS.get('Kronos-base', ''),
        'tokenizer_id': WEIGHTS.get('Kronos-Tokenizer-base', ''),
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

# Mark which models are actually available on disk
for key, cfg in list(AVAILABLE_MODELS.items()):
    if cfg['model_id'] and os.path.isdir(cfg['model_id']) and cfg['tokenizer_id'] and os.path.isdir(cfg['tokenizer_id']):
        cfg['available'] = True
    else:
        cfg['available'] = False
        del AVAILABLE_MODELS[key]

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove invalid rows and make temporal order deterministic before
        # detecting corporate-action discontinuities.
        df = (
            df.dropna()
            .sort_values('timestamps')
            .drop_duplicates('timestamps', keep='last')
            .reset_index(drop=True)
        )
        df, adjustment_report = adjust_corporate_action_gaps(df)
        df.attrs['corporate_action_adjustment'] = adjustment_report
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def resolve_prediction_timestamps(target_timestamps, pred_len, fallback_start=None, fallback_freq=None):
    """Return the timestamps used by the forecast and comparison series.

    Backtests already know the target timestamps (``y_timestamp``). Reusing
    them is important for irregular trading calendars; synthesizing a fixed
    frequency can shift the forecast away from the actual observations.
    """
    if pred_len < 0:
        raise ValueError("pred_len must be non-negative")

    if target_timestamps is not None:
        resolved = pd.DatetimeIndex(pd.to_datetime(target_timestamps))
        if len(resolved) != pred_len:
            raise ValueError(
                f"target_timestamps length must equal pred_len={pred_len}, got {len(resolved)}"
            )
        return resolved

    if pred_len == 0:
        return pd.DatetimeIndex([])
    if fallback_start is not None and fallback_freq is not None:
        return pd.date_range(start=fallback_start, periods=pred_len, freq=fallback_freq)
    return pd.RangeIndex(start=0, stop=pred_len)


def run_rolling_prediction(
    predictor,
    context_df,
    context_timestamps,
    target_df,
    target_timestamps,
    rolling_horizon,
    T=1.0,
    top_p=0.9,
    sample_count=16,
    confidence_level=0.9,
    deterministic=False,
):
    """Run a walk-forward backtest using short, independently forecast chunks.

    After each chunk, only the observed target rows are appended to the
    context. This simulates receiving new market data between forecasts and
    avoids feeding the model's own long-horizon predictions back indefinitely.
    """
    if not isinstance(context_df, pd.DataFrame) or not isinstance(target_df, pd.DataFrame):
        raise ValueError("context_df and target_df must be pandas DataFrames")
    if not 1 <= int(rolling_horizon):
        raise ValueError("rolling_horizon must be at least 1")

    context = context_df.reset_index(drop=True).copy()
    target = target_df.reset_index(drop=True).copy()
    context_timestamps = pd.Series(
        pd.to_datetime(context_timestamps), name="timestamps"
    ).reset_index(drop=True)
    target_timestamps = pd.Series(
        pd.to_datetime(target_timestamps), name="timestamps"
    ).reset_index(drop=True)

    if len(context) != len(context_timestamps):
        raise ValueError("context_df and context_timestamps must have equal lengths")
    if len(target) != len(target_timestamps):
        raise ValueError("target_df and target_timestamps must have equal lengths")
    if len(target) == 0:
        raise ValueError("target_df must contain at least one row")
    if not set(context.columns).issubset(target.columns):
        missing = sorted(set(context.columns) - set(target.columns))
        raise ValueError(f"target_df is missing context columns: {missing}")

    context_size = len(context)
    frame_keys = ("mean", "lower", "median", "upper", "std")
    frame_chunks = {key: [] for key in frame_keys}
    point_chunks = []
    array_chunks = {
        "up_probability": [],
        "cumulative_up_probability": [],
        "interval_width": [],
        "relative_interval_width": [],
    }
    prediction_reference_chunks = []
    actual_reference_chunks = []
    direction_signals = []
    calibration_chunks = []
    rolling_chunks = 0

    for chunk_start in range(0, len(target), int(rolling_horizon)):
        chunk_end = min(chunk_start + int(rolling_horizon), len(target))
        y_timestamp = target_timestamps.iloc[chunk_start:chunk_end].reset_index(drop=True)
        forecast = predictor.predict(
            df=context,
            x_timestamp=context_timestamps,
            y_timestamp=y_timestamp,
            pred_len=len(y_timestamp),
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
            return_distribution=True,
            confidence_level=confidence_level,
            deterministic=deterministic,
        )
        available_return_samples = len(context) - len(y_timestamp)
        if available_return_samples >= 60:
            forecast = apply_return_band_guardrail(
                forecast,
                context["close"].to_numpy(dtype=float),
                confidence_level=confidence_level,
                lookback=min(252, len(context) - 1),
                min_samples=min(120, available_return_samples),
            )
            calibration_chunks.append(forecast["calibration"])

        point_forecast = forecast.get("median", forecast["prediction"])
        point_chunks.append(point_forecast)
        context_last_close = float(context["close"].iloc[-1])
        predicted_close = point_forecast["close"].to_numpy(dtype=float)
        direction_signal = build_direction_signal(
            context_close=context["close"].to_numpy(dtype=float),
            predicted_close=predicted_close,
            cumulative_up_probability=forecast.get("cumulative_up_probability"),
        )
        direction_signal.update(
            {
                "target_start_index": int(chunk_start),
                "target_end_index": int(chunk_end),
            }
        )
        direction_signals.append(direction_signal)
        observed_close = target.iloc[chunk_start:chunk_end]["close"].to_numpy(dtype=float)
        prediction_reference_chunks.append(
            np.concatenate([[context_last_close], predicted_close[:-1]])
        )
        actual_reference_chunks.append(
            np.concatenate([[context_last_close], observed_close[:-1]])
        )
        for key in frame_keys:
            if key in forecast:
                frame_chunks[key].append(forecast[key])
        for key in array_chunks:
            array_chunks[key].append(np.asarray(forecast[key]))
        rolling_chunks += 1

        # Teacher-force only the observed chunk into the next context. This
        # is valid for the backtest because target_df is historical data.
        observed_chunk = target.iloc[chunk_start:chunk_end][context.columns]
        context = pd.concat([context, observed_chunk], ignore_index=True).tail(context_size)
        context_timestamps = pd.concat(
            [context_timestamps, y_timestamp], ignore_index=True
        ).tail(context_size).reset_index(drop=True)
        context = context.reset_index(drop=True)

    result = {
        "prediction": pd.concat(point_chunks, ignore_index=True),
        "up_probability": np.concatenate(array_chunks["up_probability"]),
        "cumulative_up_probability": np.concatenate(
            array_chunks["cumulative_up_probability"]
        ),
        "interval_width": np.concatenate(array_chunks["interval_width"]),
        "relative_interval_width": np.concatenate(
            array_chunks["relative_interval_width"]
        ),
        "prediction_reference_close": np.concatenate(prediction_reference_chunks),
        "actual_reference_close": np.concatenate(actual_reference_chunks),
        "direction_signals": direction_signals,
        "rolling_horizon": int(rolling_horizon),
        "rolling_chunks": rolling_chunks,
    }
    for key, chunks in frame_chunks.items():
        if chunks:
            result[key] = pd.concat(chunks, ignore_index=True)
    if calibration_chunks:
        result["calibration"] = {
            "method": "historical_log_return_quantiles",
            "confidence_level": float(confidence_level),
            "guardrail_count": int(
                sum(chunk["guardrail_count"] for chunk in calibration_chunks)
            ),
            "chunks": calibration_chunks,
            "walk_forward": True,
        }
    return result


def build_single_forecast_direction_signal(context_df, forecast, max_horizon=10):
    """Build one short-cycle signal from the first independent forecast chunk."""
    point_forecast = forecast.get("median", forecast["prediction"])
    horizon = min(int(max_horizon), len(point_forecast))
    signal = build_direction_signal(
        context_close=context_df["close"].to_numpy(dtype=float),
        predicted_close=point_forecast["close"].to_numpy(dtype=float),
        cumulative_up_probability=forecast.get("cumulative_up_probability"),
        horizon=horizon,
    )
    signal.update({"target_start_index": 0, "target_end_index": horizon})
    return signal


def build_future_timestamps(timestamps, pred_len):
    """Build approximate future timestamps without reading future OHLCV data."""
    observed = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(observed) == 0:
        raise ValueError("timestamps must contain at least one observed value")
    if pred_len < 1:
        return pd.DatetimeIndex([])

    positive_diffs = observed.to_series().diff().dropna()
    positive_diffs = positive_diffs[positive_diffs > pd.Timedelta(0)]
    interval = positive_diffs.tail(30).median() if len(positive_diffs) else pd.Timedelta(days=1)
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        interval = pd.Timedelta(days=1)
    return pd.date_range(
        start=observed[-1] + interval,
        periods=pred_len,
        freq=interval,
    )


def prepare_prediction_window(df, lookback, pred_len, mode, start_date=None):
    """Prepare the exact context/target split shared by prediction and A/B runs."""
    if mode not in ("backtest", "future"):
        raise ValueError("mode must be 'backtest' or 'future'")
    if len(df) < lookback:
        raise ValueError(f"Insufficient data length, need at least {lookback} rows")

    required_cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        required_cols.append("volume")

    target_window = None
    target_df = None
    if mode == "backtest":
        if start_date:
            start_dt = pd.to_datetime(start_date)
            mask = df["timestamps"] >= start_dt
            time_range_df = df[mask]
            if len(time_range_df) == 0:
                raise ValueError(f"No data found at or after start time {start_dt}")
        else:
            if len(df) < lookback + pred_len:
                raise ValueError(
                    f"Backtest needs at least {lookback + pred_len} rows when start_date is omitted"
                )
            time_range_df = df.iloc[-(lookback + pred_len):]

        if len(time_range_df) < lookback + pred_len:
            raise ValueError(
                f"Insufficient backtest data, need {lookback + pred_len} rows, got {len(time_range_df)}"
            )

        context_window = time_range_df.iloc[:lookback]
        target_window = time_range_df.iloc[lookback:lookback + pred_len].copy()
        x_df = context_window[required_cols].reset_index(drop=True)
        x_timestamp = pd.Series(
            pd.to_datetime(context_window["timestamps"]), name="timestamps"
        ).reset_index(drop=True)
        target_df = target_window[required_cols].reset_index(drop=True)
        target_timestamps = pd.Series(
            pd.to_datetime(target_window["timestamps"]), name="timestamps"
        ).reset_index(drop=True)
        time_span = target_window["timestamps"].iloc[-1] - time_range_df["timestamps"].iloc[0]
        prediction_type = (
            f"Kronos historical backtest: first {lookback} data points for context, "
            f"last {pred_len} data points for comparison, time span: {time_span}"
        )
        historical_start_idx = int(df.index.get_loc(time_range_df.index[0]))
    else:
        context_window = df.tail(lookback)
        x_df = context_window[required_cols].reset_index(drop=True)
        x_timestamp = pd.Series(
            pd.to_datetime(context_window["timestamps"]), name="timestamps"
        ).reset_index(drop=True)
        target_timestamps = pd.Series(
            build_future_timestamps(df["timestamps"], pred_len),
            name="timestamps",
        )
        prediction_type = f"Kronos future forecast from latest {lookback} observed data points"
        historical_start_idx = max(0, len(df) - lookback)

    return {
        "required_cols": required_cols,
        "x_df": x_df,
        "x_timestamp": x_timestamp,
        "target_df": target_df,
        "target_window": target_window,
        "target_timestamps": target_timestamps,
        "prediction_type": prediction_type,
        "historical_start_idx": historical_start_idx,
    }


def seed_inference(seed):
    """Seed the model sampler for reproducible A/B comparisons."""
    seed = int(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def classify_uncertainty(up_probability, relative_interval_width, sample_count):
    """Classify empirical forecast spread without overstating tiny samples."""
    if sample_count < 2:
        return 'insufficient_samples'
    conviction = abs(float(up_probability) - 0.5) * 2.0
    if conviction >= 0.4 and float(relative_interval_width) <= 0.05:
        return 'low'
    if conviction >= 0.2 and float(relative_interval_width) <= 0.15:
        return 'medium'
    return 'high'


def create_prediction_chart(
    df,
    pred_df,
    lookback,
    pred_len,
    actual_df=None,
    historical_start_idx=0,
    prediction_timestamps=None,
):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Prefer the exact target timestamps supplied to the model. This keeps
        # prediction and actual candles aligned on irregular trading calendars.
        if prediction_timestamps is not None:
            pred_timestamps = resolve_prediction_timestamps(
                prediction_timestamps,
                len(pred_df),
            )
        elif 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals() and len(actual_df) == len(pred_timestamps):
                actual_timestamps = pred_timestamps
            elif 'timestamps' in actual_df.columns:
                actual_timestamps = pd.DatetimeIndex(
                    pd.to_datetime(actual_df['timestamps'])
                )
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        data_info['corporate_action_adjustment'] = df.attrs.get(
            'corporate_action_adjustment',
            {'event_count': 0, 'events': []},
        )
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 16))
        confidence_level = float(data.get('confidence_level', 0.9))
        rolling_horizon = int(data.get('rolling_horizon', 0))
        deterministic = bool(data.get('deterministic', False))

        if not 1 <= sample_count <= 64:
            return jsonify({'error': 'sample_count must be between 1 and 64'}), 400
        if not 0.0 < confidence_level < 1.0:
            return jsonify({'error': 'confidence_level must be between 0 and 1'}), 400
        if rolling_horizon < 0 or rolling_horizon > pred_len:
            return jsonify({'error': f'rolling_horizon must be between 0 and {pred_len}'}), 400
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Explicitly separate historical walk-forward backtests from
                # real future forecasts. If omitted, preserve the old UI
                # behavior when start_date is present and use future mode for
                # callers that do not provide a historical target window.
                start_date = data.get('start_date')
                mode = data.get('mode') or ('backtest' if start_date else 'future')
                if rolling_horizon and mode != 'backtest':
                    return jsonify({
                        'error': 'rolling_horizon requires mode=backtest because it uses observed target chunks'
                    }), 400

                window = prepare_prediction_window(
                    df=df,
                    lookback=lookback,
                    pred_len=pred_len,
                    mode=mode,
                    start_date=start_date,
                )
                required_cols = window['required_cols']
                x_df = window['x_df']
                x_timestamp = window['x_timestamp']
                target_df = window['target_df']
                target_window = window['target_window']
                target_timestamps = window['target_timestamps']
                prediction_type = window['prediction_type']
                historical_start_idx = window['historical_start_idx']

                if rolling_horizon:
                    forecast = run_rolling_prediction(
                        predictor=predictor,
                        context_df=x_df,
                        context_timestamps=x_timestamp,
                        target_df=target_df,
                        target_timestamps=target_timestamps,
                        rolling_horizon=rolling_horizon,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count,
                        confidence_level=confidence_level,
                        deterministic=deterministic,
                    )
                    prediction_type += (
                        f", walk-forward rolling horizon: {rolling_horizon} points, "
                        f"{forecast['rolling_chunks']} chunks"
                    )
                else:
                    forecast = predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=target_timestamps,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count,
                        return_distribution=True,
                        confidence_level=confidence_level,
                        deterministic=deterministic,
                    )
                    available_return_samples = len(x_df) - pred_len
                    if available_return_samples >= 60:
                        forecast = apply_return_band_guardrail(
                            forecast,
                            x_df['close'].to_numpy(dtype=float),
                            confidence_level=confidence_level,
                            lookback=min(252, len(x_df) - 1),
                            min_samples=min(120, available_return_samples),
                        )
                # The point forecast is the sampled median in both modes;
                # the full distribution remains available for intervals.
                pred_df = forecast.get("median", forecast["prediction"])
                direction_signals = forecast.get('direction_signals')
                if direction_signals is None:
                    direction_signals = [
                        build_single_forecast_direction_signal(
                            x_df,
                            forecast,
                            max_horizon=min(10, pred_len),
                        )
                    ]
                
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        if mode == 'backtest' and target_window is not None:
            actual_df = target_window.reset_index(drop=True)
            for _, row in actual_df.iterrows():
                actual_data.append({
                    'timestamp': row['timestamps'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                    'amount': float(row['amount']) if 'amount' in row else 0
                })
        direction_evaluation = (
            evaluate_direction_signals(
                direction_signals,
                actual_df['close'].to_numpy(dtype=float),
            )
            if actual_df is not None
            else None
        )
        verified_direction = get_verified_direction(
            pd.DatetimeIndex(pd.to_datetime(x_timestamp))[-1],
            file_path,
        )
        
        prediction_timestamps = resolve_prediction_timestamps(
            target_timestamps if 'timestamps' in df.columns else None,
            pred_len,
        )
        chart_json = create_prediction_chart(
            df,
            pred_df,
            lookback,
            pred_len,
            actual_df,
            historical_start_idx,
            prediction_timestamps=prediction_timestamps,
        )

        # Reuse the same target timestamps for saved results and chart data.
        future_timestamps = prediction_timestamps

        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            up_probability = float(forecast['up_probability'][i])
            relative_interval_width = float(forecast['relative_interval_width'][i])
            # This is an uncertainty heuristic, not a calibrated probability of correctness.
            uncertainty_level = classify_uncertainty(
                up_probability,
                relative_interval_width,
                sample_count,
            )

            timestamp = future_timestamps[i] if i < len(future_timestamps) else None
            prediction_results.append({
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0,
                'close_lower': float(forecast['lower']['close'].iloc[i]),
                'close_median': float(forecast['median']['close'].iloc[i]),
                'close_upper': float(forecast['upper']['close'].iloc[i]),
                'close_std': float(forecast['std']['close'].iloc[i]),
                'up_probability': up_probability,
                'cumulative_up_probability': float(
                    forecast['cumulative_up_probability'][i]
                ),
                'interval_width': float(forecast['interval_width'][i]),
                'relative_interval_width': relative_interval_width,
                'uncertainty_level': uncertainty_level,
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'deterministic': deterministic,
                    'confidence_level': confidence_level,
                    'rolling_horizon': rolling_horizon,
                    'rolling_chunks': forecast.get('rolling_chunks', 1),
                    'mode': mode,
                    'uses_observed_targets': mode == 'backtest',
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'data_quality': {
                'corporate_action_adjustment': df.attrs.get(
                    'corporate_action_adjustment',
                    {'event_count': 0, 'events': []},
                )
            },
            'direction_summary': {
                'policy': 'verified_exogenous_one_day',
                'policy_description': (
                    'Only the independently walk-forward-tested one-day exogenous '
                    'ensemble may produce a low-confidence candidate. Unsupported '
                    'dates and longer horizons explicitly abstain.'
                ),
                'probability_is_calibrated': False,
                'verified_signal': verified_direction,
                'signals': direction_signals,
                'evaluation': direction_evaluation,
            },
            'interval_calibration': forecast.get('calibration', {
                'method': 'sample_paths_uncalibrated',
                'confidence_level': confidence_level,
                'guardrail_count': 0,
            }),
            'confidence_summary': {
                'sample_count': sample_count,
                'confidence_level': confidence_level,
                'mean_up_probability': float(np.mean(forecast['up_probability'])),
                'mean_relative_interval_width': float(np.mean(forecast['relative_interval_width'])),
                'mean_interval_width': float(np.mean(forecast['interval_width'])),
                'rolling_horizon': rolling_horizon,
                'rolling_chunks': forecast.get('rolling_chunks', 1),
                'mode': mode,
                'uses_observed_targets': mode == 'backtest',
                'deterministic': deterministic,
                'interval_method': forecast.get('calibration', {}).get(
                    'method', 'sample_paths_uncalibrated'
                ),
                'guardrail_count': int(
                    forecast.get('calibration', {}).get('guardrail_count', 0)
                ),
            },
            'message': (
                f'Prediction completed, generated {pred_len} prediction points'
                + (f' using rolling horizon {rolling_horizon}' if rolling_horizon else '')
                + (' as a historical walk-forward backtest' if mode == 'backtest' else ' as a future forecast')
                + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
            )
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/diagnostics/ab', methods=['POST'])
def run_ab_diagnostics():
    """Compare forecast stability under fixed, reproducible inference cases."""
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 16))
        confidence_level = float(data.get('confidence_level', 0.9))
        rolling_horizon = int(data.get('rolling_horizon', 10))
        seed = int(data.get('seed', 20260802))
        requested_case_ids = data.get('case_ids')
        start_date = data.get('start_date')
        mode = data.get('mode') or ('backtest' if start_date else 'future')

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        if not 1 <= sample_count <= 64:
            return jsonify({'error': 'sample_count must be between 1 and 64'}), 400
        if not 0.0 < confidence_level < 1.0:
            return jsonify({'error': 'confidence_level must be between 0 and 1'}), 400
        if rolling_horizon < 0 or rolling_horizon > pred_len:
            return jsonify({'error': f'rolling_horizon must be between 0 and {pred_len}'}), 400
        if rolling_horizon and mode != 'backtest':
            return jsonify({
                'error': 'rolling_horizon requires mode=backtest because it uses observed target chunks'
            }), 400
        if not MODEL_AVAILABLE or predictor is None:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400

        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        window = prepare_prediction_window(df, lookback, pred_len, mode, start_date)
        actual_df = window['target_df'] if mode == 'backtest' else None
        cases = []
        case_configs = build_ab_cases(temperature, top_p, sample_count)
        if requested_case_ids is not None:
            if not isinstance(requested_case_ids, list) or not requested_case_ids:
                return jsonify({'error': 'case_ids must be a non-empty list'}), 400
            available_case_ids = {case['id'] for case in case_configs}
            unknown_case_ids = sorted(set(requested_case_ids) - available_case_ids)
            if unknown_case_ids:
                return jsonify({'error': f'Unknown case_ids: {unknown_case_ids}'}), 400
            case_configs = [
                case for case in case_configs if case['id'] in requested_case_ids
            ]

        for case_index, case in enumerate(case_configs):
            seed_inference(seed + case_index)
            if rolling_horizon:
                forecast = run_rolling_prediction(
                    predictor=predictor,
                    context_df=window['x_df'],
                    context_timestamps=window['x_timestamp'],
                    target_df=window['target_df'],
                    target_timestamps=window['target_timestamps'],
                    rolling_horizon=rolling_horizon,
                    T=case['temperature'],
                    top_p=case['top_p'],
                    sample_count=case['sample_count'],
                    confidence_level=confidence_level,
                    deterministic=case['deterministic'],
                )
            else:
                forecast = predictor.predict(
                    df=window['x_df'],
                    x_timestamp=window['x_timestamp'],
                    y_timestamp=window['target_timestamps'],
                    pred_len=pred_len,
                    T=case['temperature'],
                    top_p=case['top_p'],
                    sample_count=case['sample_count'],
                    verbose=False,
                    return_distribution=True,
                    confidence_level=confidence_level,
                    deterministic=case['deterministic'],
                )
                available_return_samples = len(window['x_df']) - pred_len
                if available_return_samples >= 60:
                    forecast = apply_return_band_guardrail(
                        forecast,
                        window['x_df']['close'].to_numpy(dtype=float),
                        confidence_level=confidence_level,
                        lookback=min(252, len(window['x_df']) - 1),
                        min_samples=min(120, available_return_samples),
                    )

            point_forecast = forecast.get('median', forecast['prediction'])
            direction_signals = forecast.get('direction_signals')
            if direction_signals is None:
                direction_signals = [
                    build_single_forecast_direction_signal(
                        window['x_df'],
                        forecast,
                        max_horizon=min(10, pred_len),
                    )
                ]
            prediction_reference_close = None
            actual_reference_close = None
            if actual_df is not None:
                if rolling_horizon:
                    prediction_reference_close = forecast['prediction_reference_close']
                    actual_reference_close = forecast['actual_reference_close']
                else:
                    context_last_close = float(window['x_df']['close'].iloc[-1])
                    predicted_close = point_forecast['close'].to_numpy(dtype=float)
                    observed_close = actual_df['close'].to_numpy(dtype=float)
                    prediction_reference_close = np.concatenate(
                        [[context_last_close], predicted_close[:-1]]
                    )
                    actual_reference_close = np.concatenate(
                        [[context_last_close], observed_close[:-1]]
                    )
            diagnostics = summarize_forecast_diagnostics(
                point_forecast,
                actual_df,
                prediction_reference_close=prediction_reference_close,
                actual_reference_close=actual_reference_close,
                up_probability=forecast['up_probability'],
                direction_signals=direction_signals,
            )
            cases.append({
                **case,
                'seed': seed + case_index,
                'rolling_chunks': int(forecast.get('rolling_chunks', 1)),
                'diagnostics': diagnostics,
                'interval_calibration': forecast.get('calibration', {
                    'method': 'sample_paths_uncalibrated',
                    'guardrail_count': 0,
                }),
            })

        return jsonify({
            'success': True,
            'mode': mode,
            'lookback': lookback,
            'pred_len': pred_len,
            'rolling_horizon': rolling_horizon,
            'seed': seed,
            'uses_observed_targets': mode == 'backtest',
            'cases': cases,
            'data_quality': {
                'corporate_action_adjustment': df.attrs.get(
                    'corporate_action_adjustment',
                    {'event_count': 0, 'events': []},
                )
            },
            'message': 'A/B diagnostic completed with fixed windows and per-case seeds',
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'A/B diagnostic failed: {str(e)}'}), 500


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model from local weights
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
