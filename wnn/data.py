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


def maybe_augment(X: np.ndarray, aug: bool, bank: str = "") -> np.ndarray:
    """aug: legacy 5-block augmentation; bank: EMA-bank spec (wnn.features.parse_bank), e.g. '100,300,1000,3000[:s]'."""
    if bank:
        from wnn.features import bank_sequence, n_bank, parse_bank

        alphas, scaled = parse_bank(bank)
        out = np.empty((X.shape[0], n_bank(X.shape[1], len(alphas))), np.float32)
        bank_sequence(np.ascontiguousarray(X, dtype=np.float32), alphas, scaled, out)
        return out
    if not aug:
        return X
    from wnn.features import augment_sequence, n_aug

    out = np.empty((X.shape[0], n_aug(X.shape[1])), np.float32)
    augment_sequence(np.ascontiguousarray(X, dtype=np.float32), out)
    return out


class SeqDataset(Dataset):
    """Yields (X, y, need, scored, soft, w). soft = teacher soft labels (T,2) if soft_path is given, else a copy of y;
    w = per-step loss weights (T,) from weight_path (n_row_groups, T) if given, else ones."""

    def __init__(self, path: str, indices: list[int] | None = None, feat: list[str] | None = None,
                 aug: bool = False, soft_path: str | None = None, weight_path: str | None = None,
                 seq_weight_path: str | None = None, bank: str = ""):
        import pyarrow.parquet as pq

        self.path = path
        self.feat = feat or feature_columns(path)
        self.aug = aug
        self.bank = bank  # EMA-bank spec (see maybe_augment)
        self.soft_path = soft_path
        self.weight_path = weight_path
        self.seq_weight_path = seq_weight_path  # (n_row_groups,) per-sequence multiplier on the row weights
        self._sw = None
        n = pq.ParquetFile(path).num_row_groups
        self.indices = list(range(n)) if indices is None else list(indices)
        self._pf = None  # opened lazily per worker
        self._soft = None
        self._w = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._pf is None:
            import pyarrow.parquet as pq

            self._pf = pq.ParquetFile(self.path)
        rg = self.indices[i]
        X, y, need, scored = read_sequence(self._pf, rg, self.feat)
        X = maybe_augment(X, self.aug, self.bank)
        if self.soft_path:
            if self._soft is None:
                self._soft = np.load(self.soft_path, mmap_mode="r")  # (n_row_groups, T, 2) float16, row-group order
            soft = np.asarray(self._soft[rg], dtype=np.float32)
        else:
            soft = y.copy()
        if self.weight_path:
            if self._w is None:
                self._w = np.load(self.weight_path, mmap_mode="r")  # (n_row_groups, T) float16
            w = np.asarray(self._w[rg], dtype=np.float32)
        else:
            w = np.ones(len(X), np.float32)
        if self.seq_weight_path:
            if self._sw is None:
                self._sw = np.load(self.seq_weight_path).astype(np.float32)
            w = w * float(self._sw[rg])
        return (torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(need), torch.from_numpy(scored),
                torch.from_numpy(soft), torch.from_numpy(w))


class SyntheticDataset(Dataset):
    """Smoke-test data with a learnable signal: y depends on an EMA of a few features."""

    def __init__(self, n_seq=64, T=2000, F=112, seed=0, aug=False, bank=""):
        self.n_seq, self.T, self.F, self.seed, self.aug, self.bank = n_seq, T, F, seed, aug, bank

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
        X = maybe_augment(X, self.aug, self.bank)
        return (torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(need), torch.from_numpy(need.copy()),
                torch.from_numpy(y.copy()), torch.ones(self.T))


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
