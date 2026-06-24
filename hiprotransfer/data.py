import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_minmax(train_values, test_values=None):
    train_values = np.asarray(train_values, dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_values)
    train_scaled = scaler.transform(train_values)
    if test_values is None:
        return train_scaled.astype(np.float32), scaler
    test_values = np.asarray(test_values, dtype=np.float32)
    test_scaled = np.clip(scaler.transform(test_values), -3.0, 3.0)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def split_day_segments(series, minutes_per_day=1440, days=None):
    series = np.asarray(series, dtype=np.float32)
    max_days = series.shape[0] // minutes_per_day
    if days is not None:
        max_days = min(max_days, int(days))
    return [
        series[day * minutes_per_day:(day + 1) * minutes_per_day]
        for day in range(max_days)
    ]


class SlidingWindowDataset(Dataset):
    def __init__(self, values, window_size, masks=None):
        self.window_size = int(window_size)
        value_list = values if isinstance(values, list) else [values]
        mask_list = self._normalize_masks(masks, len(value_list))
        has_mask = any(mask is not None for mask in mask_list)

        windows = []
        window_masks = []
        for value, mask in zip(value_list, mask_list):
            value = np.asarray(value, dtype=np.float32)
            value_windows = self._to_windows(value)
            windows.append(value_windows)
            if has_mask:
                if mask is None:
                    mask = np.ones(value.shape[-1], dtype=np.float32)
                mask = np.asarray(mask, dtype=np.float32)
                window_masks.append(np.repeat(mask[np.newaxis, :], value_windows.shape[0], axis=0))

        self.windows = np.concatenate(windows, axis=0)
        self.masks = np.concatenate(window_masks, axis=0) if has_mask else None

    @staticmethod
    def _normalize_masks(masks, count):
        if masks is None:
            return [None] * count
        if isinstance(masks, list):
            if len(masks) != count:
                raise ValueError("masks list length must match values list length")
            return masks
        return [masks] * count

    def _to_windows(self, values):
        seq_len, feature_dim = values.shape
        if seq_len < self.window_size:
            padded = np.zeros((self.window_size, feature_dim), dtype=np.float32)
            padded[:seq_len] = values
            return padded.reshape(1, self.window_size, feature_dim)
        return np.lib.stride_tricks.as_strided(
            values,
            shape=(seq_len - self.window_size + 1, self.window_size, feature_dim),
            strides=(values.strides[0], values.strides[0], values.strides[1]),
        ).copy()

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        window = self.windows[index].astype(np.float32)
        if self.masks is None:
            return torch.from_numpy(window)
        return torch.from_numpy(window), torch.from_numpy(self.masks[index].astype(np.float32))


def make_loader(values, window_size, batch_size, masks=None, shuffle=False, drop_last=False):
    dataset = SlidingWindowDataset(values, window_size=window_size, masks=masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
