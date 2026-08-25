"""Streaming per-interface statistics (docs/01 C0.2).

All accumulators take activations of shape (n, d) (positions flattened) or (b, L, d)
where the position axis matters, accumulate in float64, and are finalized once.
"""
from __future__ import annotations

import numpy as np
import torch


class MeanCov:
    """Running mean and covariance in f64, Ledoit-Wolf shrinkage at finalize."""

    def __init__(self, d: int):
        self.d = d
        self.n = 0
        self.s1 = torch.zeros(d, dtype=torch.float64)
        self.s2 = torch.zeros(d, d, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, self.d).to(torch.float64)
        self.n += x.shape[0]
        self.s1 += x.sum(0)
        self.s2 += x.T @ x

    def finalize(self, shrink: float | None = None):
        mean = self.s1 / self.n
        cov = self.s2 / self.n - torch.outer(mean, mean)
        if shrink is None:
            shrink = ledoit_wolf_shrinkage(cov, self.n)
        mu = torch.trace(cov) / self.d
        cov_s = (1 - shrink) * cov + shrink * mu * torch.eye(self.d, dtype=torch.float64, device=cov.device)
        return {"mean": mean, "cov": cov, "cov_shrunk": cov_s, "shrink": float(shrink),
                "n": self.n}


def ledoit_wolf_shrinkage(cov: torch.Tensor, n: int) -> float:
    """Ledoit-Wolf 2004 shrinkage toward scaled identity, from the covariance alone.

    The exact LW estimator needs fourth moments of the data; at n >> d the
    oracle-approximating form below (Chen et al. 2010, OAS) is a close and cheap stand-in.
    """
    d = cov.shape[0]
    tr = torch.trace(cov)
    tr2 = torch.trace(cov @ cov)
    num = (1 - 2 / d) * tr2 + tr**2
    den = (n + 1 - 2 / d) * (tr2 - tr**2 / d)
    rho = float(num / den) if den > 0 else 1.0
    return min(max(rho, 0.0), 1.0)


def zca_from_cov(cov: torch.Tensor, eps: float = 1e-6):
    """Symmetric whitening W = U diag(1/sqrt(l+eps')) U^T and its inverse, with
    eps' = eps * max eigenvalue, so the condition number of W is at most
    sqrt(1/eps) whatever the activation scale."""
    lam, U = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    eps = eps * float(lam.max())
    W = U @ torch.diag(1 / torch.sqrt(lam + eps)) @ U.T
    Winv = U @ torch.diag(torch.sqrt(lam + eps)) @ U.T
    return W, Winv, lam, U


class CFSketch:
    """Sketched characteristic function: mean cos/sin of <x, u_m> * t_k.

    Directions u_m are fixed unit vectors from `seed`; frequencies t_k a fixed grid.
    Output (M, K, 2). With `whiten=(W, mean)` the sketch is taken in whitened coords.
    """

    def __init__(self, d: int, M: int = 256, K: int = 16, seed: int = 0,
                 t_max: float = 4.0, whiten=None):
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(M, d, generator=g, dtype=torch.float64)
        self.U = U / U.norm(dim=1, keepdim=True)
        self.t = torch.linspace(t_max / K, t_max, K, dtype=torch.float64)
        self.whiten = whiten
        self.n = 0
        self.acc = torch.zeros(M, K, 2, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)
        if self.whiten is not None:
            W, mean = self.whiten
            x = (x - mean) @ W
        p = x @ self.U.T  # (n, M)
        a = p[:, :, None] * self.t[None, None, :]  # (n, M, K)
        self.acc[..., 0] += torch.cos(a).sum(0)
        self.acc[..., 1] += torch.sin(a).sum(0)
        self.n += x.shape[0]

    def finalize(self):
        return {"cf": self.acc / self.n, "U": self.U, "t": self.t, "n": self.n}


def cf_distance(cf_a: torch.Tensor, cf_b: torch.Tensor) -> float:
    """Mean squared CF difference over directions and frequencies (Epps-Pulley flavour)."""
    return float(((cf_a - cf_b) ** 2).sum(-1).mean())


class Quantiles:
    """Per-channel quantiles by reservoir sampling; exact at modest n, unbiased beyond."""

    def __init__(self, d: int, levels: int = 1024, reservoir: int = 1 << 16, seed: int = 0):
        # 2^16 rows x d float32: 0.5 GB per interface at d=2048. 2^18 cost 15 GB RSS
        # across 7 interfaces on the first 1.4B harvest (22 Aug 2026) for no gain.
        self.d, self.levels, self.cap = d, levels, reservoir
        self.g = torch.Generator().manual_seed(seed)
        self.buf = torch.empty(0, d, dtype=torch.float32)
        self.n = 0

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, self.d).to(torch.float32)
        room = self.cap - self.buf.shape[0]
        if room > 0:
            self.buf = torch.cat([self.buf, x[:room]])
            x = x[room:]
        self.n += x.shape[0] + min(room, x.shape[0]) if room > 0 else x.shape[0]
        if x.shape[0]:
            # reservoir replacement, vectorized approximation: each new row replaces a
            # uniformly random slot with probability cap / n_seen_so_far
            idx = torch.randint(0, self.n, (x.shape[0],), generator=self.g)
            keep = idx < self.cap
            self.buf[idx[keep]] = x[keep]

    def finalize(self):
        q = torch.linspace(0, 1, self.levels, dtype=torch.float32)
        return {"q": q, "values": torch.quantile(self.buf, q, dim=0), "n": self.n}


class Lag1:
    """Per-channel lag-1 autocorrelation across positions, on (b, L, d) inputs."""

    def __init__(self, d: int):
        self.d = d
        self.sx = torch.zeros(d, dtype=torch.float64)
        self.sxx = torch.zeros(d, dtype=torch.float64)
        self.sxy = torch.zeros(d, dtype=torch.float64)
        self.n = 0

    def update(self, x: torch.Tensor):
        x = x.to(torch.float64)
        a, b = x[:, :-1, :], x[:, 1:, :]
        self.sx += a.sum((0, 1))
        self.sxx += (a * a).sum((0, 1))
        self.sxy += (a * b).sum((0, 1))
        self.n += a.shape[0] * a.shape[1]

    def finalize(self):
        m = self.sx / self.n
        var = self.sxx / self.n - m * m
        cov = self.sxy / self.n - m * m
        return {"rho": cov / var.clamp_min(1e-12), "n": self.n}


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean entropy per head of an attention matrix (b, h, L, L), in nats."""
    p = attn.to(torch.float32).clamp_min(1e-30)
    h = -(p * p.log()).sum(-1)  # (b, h, L), float32: float64 of a (8,16,2048,2048) map is 4 GB
    return h.mean((0, 2)).to(torch.float64)


def to_numpy(d: dict) -> dict:
    return {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in d.items()}
