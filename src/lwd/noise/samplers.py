"""Noise measures at an interface (docs/02 factor 1) and sequence structure (factor 2).

Every sampler returns (b, L, d) tensors in raw teacher coordinates, so the teacher
stage can be applied directly. Samples are defined per position; a sequence is L of
them with the chosen dependence across positions.
"""
from __future__ import annotations

import torch


def _base(b, L, d, structure: str, rho: torch.Tensor | None, g, device, dtype=torch.float32):
    """Standard-normal per channel with the requested across-position dependence."""
    # draw on the CPU generator (device-independent, reproducible), then move
    e = torch.randn(b, L, d, generator=g, dtype=dtype).to(device)
    if structure == "iid":
        return e
    if structure == "ar1":
        assert rho is not None
        r = rho.to(device, dtype).clamp(-0.999, 0.999)
        s = torch.sqrt(1 - r * r)
        z = torch.empty_like(e)
        z[:, 0] = e[:, 0]
        for t in range(1, L):
            z[:, t] = r * z[:, t - 1] + s * e[:, t]
        return z
    raise ValueError(f"unknown sequence structure {structure!r}; a mix passes 'iid' "
                     "to its noise component, the mixing is the structure")


class Isotropic:
    """N(0, sigma^2 I) with sigma^2 = tr(Sigma)/d."""

    def __init__(self, mean, cov, structure="iid", rho=None):
        self.mean = mean.float()
        self.sigma = float(torch.sqrt(torch.trace(cov.double()) / cov.shape[0]))
        self.structure, self.rho = structure, rho

    def sample(self, b, L, g, device="cpu"):
        d = self.mean.numel()
        return _base(b, L, d, self.structure, self.rho, g, device) * self.sigma + self.mean.to(device)


class Gaussian:
    """N(mean, cov_shrunk); with ar1, dependence is applied in whitened coordinates."""

    def __init__(self, mean, cov, structure="iid", rho=None, eps=1e-6):
        from lwd.harvest.stats import zca_from_cov
        _, Winv, _, _ = zca_from_cov(cov.double(), eps)
        self.mean, self.Winv = mean.float(), Winv.float()
        self.structure, self.rho = structure, rho

    def sample(self, b, L, g, device="cpu"):
        d = self.mean.numel()
        z = _base(b, L, d, self.structure, self.rho, g, device)
        return z @ self.Winv.to(device) + self.mean.to(device)


class ContractGaussianized:
    """z ~ N(0, I) (with structure) pushed through phi^{-1} of a Contract, so marginals
    and second moments match the teacher's."""

    def __init__(self, contract, structure="iid", rho=None):
        self.c, self.structure, self.rho = contract, structure, rho

    def sample(self, b, L, g, device="cpu"):
        d = self.c.affine.mean.numel()
        dt = self.c.dtype
        z = _base(b, L, d, self.structure, self.rho, g, device, dtype=dt)
        return self.c.inverse(z.reshape(-1, d)).reshape(b, L, d).float()


class AnchorMix:
    """Real anchor sequences mixed with a noise sampler at a fixed ratio."""

    def __init__(self, anchors: torch.Tensor, noise, real_frac: float = 1 / 11):
        self.anchors, self.noise, self.real_frac = anchors, noise, real_frac

    def sample(self, b, L, g, device="cpu"):
        n_real = int(round(b * self.real_frac))
        assert 1 <= n_real < b, f"batch {b} x real_frac {self.real_frac} gives {n_real} real sequences"
        idx = torch.randint(0, self.anchors.shape[0], (n_real,), generator=g)
        real = self.anchors[idx, :L].to(device).float()
        fake = self.noise.sample(b - n_real, L, g, device)
        return torch.cat([real, fake]), n_real


class LiveReal:
    """Real activations at interface k computed live from token rows: embedding plus
    the teacher blocks below the stage, so every sample is a distinct real position
    (the anchor-based RealSampler recycles ~1M positions). `rows` (n, >=L) token ids;
    `lower` is a callable ids -> interface-k activations on `device`."""

    def __init__(self, rows, lower, exclude=()):
        import numpy as np
        keep = np.array([i for i in range(len(rows)) if i not in set(exclude)])
        self.rows, self.lower = rows[keep], lower

    def sample(self, b, L, g, device="cpu"):
        import numpy as np
        idx = torch.randint(0, len(self.rows), (b,), generator=g).numpy()
        ids = torch.from_numpy(self.rows[idx, :L].astype(np.int64)).to(device)
        with torch.no_grad():
            return self.lower(ids).float()
