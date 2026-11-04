"""Global Weighted Pearson (WP) as described in the Data overview docs.

For each target: clip y and y_hat to [-2, 2], weights w = |y_clipped|,
weighted Pearson correlation over all selected rows pooled across sequences.
Score = mean over t0 and t1. Cross-checked against the pack's METRIC.md once available.
"""
import numpy as np

from config import CLIP


def weighted_pearson(y: np.ndarray, y_hat: np.ndarray, clip: float = CLIP) -> float:
    y = np.clip(y.astype(np.float64), -clip, clip)
    y_hat = np.clip(y_hat.astype(np.float64), -clip, clip)
    w = np.abs(y)
    sw = w.sum()
    if sw <= 0:
        return 0.0
    my = (w * y).sum() / sw
    mh = (w * y_hat).sum() / sw
    dy, dh = y - my, y_hat - mh
    cov = (w * dy * dh).sum()
    var = np.sqrt((w * dy * dy).sum() * (w * dh * dh).sum())
    return float(cov / var) if var > 0 else 0.0


def score(y: np.ndarray, y_hat: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """y, y_hat: (n, 2) arrays; mask: boolean (n,) selecting need_prediction & is_scored rows."""
    if mask is not None:
        y, y_hat = y[mask], y_hat[mask]
    r0 = weighted_pearson(y[:, 0], y_hat[:, 0])
    r1 = weighted_pearson(y[:, 1], y_hat[:, 1])
    return {"t0": r0, "t1": r1, "score": 0.5 * (r0 + r1), "n": int(len(y))}
