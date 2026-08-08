import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        manifest_path = Path(self.config.dataset_path) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing dataset manifest: {manifest_path}. "
                "Run data/build_local_finetune_dataset.py first."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_window = (manifest.get("lookback_window"), manifest.get("predict_window"))
        expected_window = (self.config.lookback_window, self.config.predict_window)
        if manifest_window != expected_window or manifest.get("window") != self.window:
            raise ValueError(
                "Dataset contract mismatch: "
                f"manifest={manifest_window}, window={manifest.get('window')}; "
                f"config={expected_window}, window={self.window}"
            )

        self.symbols = list(self.data.keys())
        if set(self.symbols) != set(manifest.get("symbols", [])):
            raise ValueError("Dataset symbols do not match manifest.json")
        # Optional training subset: comma-separated symbols via KRONOS_SYMBOL_FILTER.
        # Useful for related-peer experiments without rebuilding pickles.
        import os
        raw_filter = os.getenv("KRONOS_SYMBOL_FILTER", "").strip()
        if raw_filter:
            allowed = {s.strip() for s in raw_filter.split(",") if s.strip()}
            missing = allowed - set(self.symbols)
            if missing:
                raise ValueError(f"KRONOS_SYMBOL_FILTER symbols not in dataset: {sorted(missing)}")
            self.symbols = [s for s in self.symbols if s in allowed]
            print(f"[{data_type.upper()}] Symbol filter active: {len(self.symbols)} symbols")
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Keep raw closes as a supervision-only side channel.  The predictor
        # continues to receive the normalized feature matrix below; calculating
        # log returns from normalized closes would be mathematically invalid.
        raw_close = win_df['close'].values.astype(np.float32)

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        raw_close_tensor = torch.from_numpy(raw_close)

        return x_tensor, x_stamp_tensor, raw_close_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp, try_raw_close = train_dataset[100]  # Index is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"Raw close supervision shape: {try_raw_close.shape}")
    else:
        print("Dataset is empty.")
