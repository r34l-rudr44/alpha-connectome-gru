"""Sequence dataset over parquet row groups (one row group == one 20k-step sequence)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")


def feature_columns(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in META]


def read_sequence(pf, rg: int, feat: list[str]):
    """Returns X (T,F) float32, y (T,2) float32, need (T,) bool, scored (T,) bool."""
    t = pf.read_row_group(rg)
    X = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in feat], 1).astype(np.float32, copy=False)
    T = len(X)
    names = set(t.column_names)
    y = (np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1).astype(np.float32)
         if "t0" in names else np.zeros((T, 2), np.float32))
    need = t.column("need_prediction").to_numpy().astype(bool)
    scored = t.column("is_scored").to_numpy().astype(bool) if "is_scored" in names else need.copy()
    return X, y, need, scored


def holdout_split(path: str, frac: float, seed: int = 0):
    """Deterministic split of a parquet's row groups into (rest, holdout) index lists."""
    import pyarrow.parquet as pq

    n = pq.ParquetFile(path).num_row_groups
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(frac * n))
    hold = sorted(perm[:k].tolist())
    rest = sorted(perm[k:].tolist())
    return rest, hold


def maybe_augment(X: np.ndarray, aug: bool) -> np.ndarray:
    if not aug:
        return X
    from wnn.features import augment_sequence, n_aug

    out = np.empty((X.shape[0], n_aug(X.shape[1])), np.float32)
    augment_sequence(np.ascontiguousarray(X, dtype=np.float32), out)
    return out


class SeqDataset(Dataset):
    def __init__(self, path: str, indices: list[int] | None = None, feat: list[str] | None = None, aug: bool = False):
        import pyarrow.parquet as pq

        self.path = path
        self.feat = feat or feature_columns(path)
        self.aug = aug
        n = pq.ParquetFile(path).num_row_groups
        self.indices = list(range(n)) if indices is None else list(indices)
        self._pf = None  # opened lazily per worker

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._pf is None:
            import pyarrow.parquet as pq

            self._pf = pq.ParquetFile(self.path)
        X, y, need, scored = read_sequence(self._pf, self.indices[i], self.feat)
        X = maybe_augment(X, self.aug)
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(need), torch.from_numpy(scored)


class SyntheticDataset(Dataset):
    """Smoke-test data with a learnable signal: y depends on an EMA of a few features."""

    def __init__(self, n_seq=64, T=2000, F=112, seed=0, aug=False):
        self.n_seq, self.T, self.F, self.seed, self.aug = n_seq, T, F, seed, aug

    def __len__(self):
        return self.n_seq

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        X = (rng.normal(size=(self.T, self.F)) * (1 + np.arange(self.F) % 5)).astype(np.float32)
        ema = np.zeros(self.T, np.float32)
        for t in range(1, self.T):
            ema[t] = 0.9 * ema[t - 1] + 0.1 * X[t, 3]
        y = np.stack([ema + 0.3 * X[:, 7], -ema + 0.3 * X[:, 11]], 1).astype(np.float32)
        y += rng.normal(size=y.shape).astype(np.float32) * 0.3
        need = np.arange(self.T) >= 99
        X = maybe_augment(X, self.aug)
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(need), torch.from_numpy(need.copy())


def compute_feature_stats(ds: Dataset, n: int = 200, seed: int = 0):
    """Per-feature mean/std over the first n sequences (deterministic subset)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    s = s2 = cnt = 0
    for i in idx:
        X = ds[int(i)][0].double().numpy()
        X = np.nan_to_num(X)
        s = s + X.sum(0); s2 = s2 + (X ** 2).sum(0); cnt += len(X)
    mean = s / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)
