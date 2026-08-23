"""HT-SR style spectral metrics per weight matrix (source document §1.5, gates only).

alpha: power-law exponent of the tail of the eigenvalue spectrum of W^T W (Hill
estimator) with xmin chosen by minimum KS distance over candidate tails, as in
WeightWatcher. Also stable rank and the log-spectral-norm. Pure numpy, CPU.
"""
from __future__ import annotations

import numpy as np
import torch


def esd(W: torch.Tensor) -> np.ndarray:
    w = W.detach().float().cpu().numpy()
    if w.ndim != 2:
        w = w.reshape(w.shape[0], -1)
    s = np.linalg.svd(w, compute_uv=False)
    return np.sort(s ** 2)


def alpha_hill_ks(lam: np.ndarray, min_tail: int = 50) -> tuple[float, float, int]:
    """(alpha, ks, n_tail). Tail = eigenvalues >= xmin; alpha = 1 + n / sum log(x/xmin).
    xmin chosen to minimize the KS distance between the tail's empirical CDF and the
    fitted power law, over tails of at least min_tail points."""
    lam = lam[lam > 0]
    n = len(lam)
    best = (np.nan, np.inf, 0)
    for i in range(0, n - min_tail):
        xmin = lam[i]
        tail = lam[i:]
        a = 1 + len(tail) / np.sum(np.log(tail / xmin))
        cdf_emp = np.arange(1, len(tail) + 1) / len(tail)
        cdf_fit = 1 - (tail / xmin) ** (1 - a)
        ks = np.max(np.abs(cdf_emp - cdf_fit))
        if ks < best[1]:
            best = (a, ks, len(tail))
    return best


def stable_rank(lam: np.ndarray) -> float:
    return float(lam.sum() / lam.max())


def metrics_for(state: dict, min_dim: int = 256) -> dict:
    out = {}
    for k, v in state.items():
        if v.ndim == 2 and min(v.shape) >= min_dim:
            lam = esd(v)
            a, ks, nt = alpha_hill_ks(lam)
            out[k] = {"alpha": a, "ks": ks, "n_tail": nt, "stable_rank": stable_rank(lam),
                      "log_spectral_norm": float(np.log10(lam.max())), "shape": list(v.shape)}
    return out
