import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """Creates supervised sequences (lookback -> predict horizon) for multivariate series."""
    def __init__(self, csv_path, seq_len=48, target_col='target', split='train', splits=None):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        # normalize features by train stats if splits provided
        self.seq_len = seq_len
        self.target_col = target_col
        self.df = df
        self.splits = splits or {'train':[0,int(len(df)*0.7)], 'val':[int(len(df)*0.7), int(len(df)*0.85)], 'test':[int(len(df)*0.85), len(df)]}
        s,e = self.splits[split]
        self.sub = df.iloc[s:e].reset_index(drop=True)
        # features: all non-timestamp, non-target columns
        self.feature_cols = [c for c in df.columns if c not in ['timestamp', target_col]]
        # compute stats on train split for scaling
        train_slice = df.iloc[self.splits['train'][0]:self.splits['train'][1]]
        self.mean = train_slice[self.feature_cols].mean().values
        self.std = train_slice[self.feature_cols].std().values + 1e-6
        self.X = (self.sub[self.feature_cols].values - self.mean) / self.std
        self.y = self.sub[target_col].values

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]  # (seq_len, n_features)
        y = self.y[idx + self.seq_len]  # predict next step
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_data_loaders(csv_path, seq_len=48, batch_size=64, splits=None):
    train_ds = TimeSeriesDataset(csv_path, seq_len=seq_len, split='train', splits=splits)
    val_ds = TimeSeriesDataset(csv_path, seq_len=seq_len, split='val', splits=splits)
    test_ds = TimeSeriesDataset(csv_path, seq_len=seq_len, split='test', splits=splits)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
