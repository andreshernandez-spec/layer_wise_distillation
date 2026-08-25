"""Interface normalization maps phi (docs/02, source doc §1.1): bijections applied to
both teacher and student so the optimum does not move.

  Affine:      phi(x) = (x - mean) @ W            (ZCA or PCA whitening)
  Gaussianize: phi(x) = Phi^{-1}(F_c(x_c)) per channel, then affine whitening of the
               result (RBIG-style, one iteration). Monotone per channel, so massive
               activations are squashed, never removed.

Every map has an exact inverse and a test that checks it.
"""
from __future__ import annotations

import math

import torch


class Affine:
    def __init__(self, mean: torch.Tensor, W: torch.Tensor, Winv: torch.Tensor):
        self.mean, self.W, self.Winv = mean, W, Winv

    @classmethod
    def zca(cls, mean, cov, eps=1e-6):
        from lwd.harvest.stats import zca_from_cov
        W, Winv, _, _ = zca_from_cov(cov.to(torch.float64), eps)
        return cls(mean.to(torch.float64), W, Winv)

    @classmethod
    def pca(cls, mean, cov, d_s: int | None = None, eps=1e-6):
        """Top-d_s principal directions, whitened. Inverse is the pseudo-inverse
        (projection back onto the subspace)."""
        lam, U = torch.linalg.eigh(cov.to(torch.float64))
        order = torch.argsort(lam, descending=True)[: d_s or cov.shape[0]]
        Ue, le = U[:, order], lam[order].clamp_min(0)
        W = Ue / torch.sqrt(le + eps)[None, :]
        Winv = (Ue * torch.sqrt(le + eps)[None, :]).T
        return cls(mean.to(torch.float64), W, Winv)

    def to(self, device, dtype=torch.float32):
        return Affine(self.mean.to(device, dtype), self.W.to(device, dtype), self.Winv.to(device, dtype))

    @property
    def dtype(self):
        return self.mean.dtype

    def forward(self, x):
        return (x - self.mean) @ self.W

    def inverse(self, z):
        return z @ self.Winv + self.mean


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2 * p - 1)


def _ndtr(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


class MarginalGaussianize:
    """Per-channel monotone map x_c -> Phi^{-1}(F_c(x_c)) from harvested quantiles.

    F_c is the piecewise-linear CDF through the quantile table; beyond the table it is
    extended by the tails of a normal fitted to the outermost quantiles, so the map is
    defined and invertible on all of R.
    """

    def __init__(self, q: torch.Tensor, values: torch.Tensor, clip: float = 6.0):
        # q (Q,), values (Q, d), both increasing in Q. Clamp q away from 0 and 1.
        self.q = q.to(torch.float64).clamp(1e-6, 1 - 1e-6)
        self.v = values.to(torch.float64)
        self.z = _ndtri(self.q)  # (Q,)
        self.clip = clip
        # make values strictly increasing per channel so interpolation is invertible
        self.v = torch.cummax(self.v, dim=0).values
        eps = 1e-7 * (self.v[-1] - self.v[0]).clamp_min(1e-12)
        self.v = self.v + torch.arange(self.v.shape[0], dtype=torch.float64)[:, None] * eps

    def to(self, device, dtype=torch.float32):
        m = MarginalGaussianize.__new__(MarginalGaussianize)
        m.q, m.v, m.z, m.clip = self.q.to(device, dtype), self.v.to(device, dtype), self.z.to(device, dtype), self.clip
        return m

    @staticmethod
    def _interp(x, xp, fp):
        """Per-channel 1D linear interpolation with linear extrapolation.
        x (n, d); xp (Q, d) increasing; fp (Q,) or (Q, d)."""
        Q = xp.shape[0]
        if fp.dim() == 1:
            fp = fp[:, None].expand(Q, xp.shape[1])
        # index of the right knot per element
        idx = torch.searchsorted(xp.T.contiguous(), x.T.contiguous()).T.clamp(1, Q - 1)  # (n, d)
        x0 = torch.gather(xp.T, 1, (idx - 1).T).T
        x1 = torch.gather(xp.T, 1, idx.T).T
        y0 = torch.gather(fp.T, 1, (idx - 1).T).T
        y1 = torch.gather(fp.T, 1, idx.T).T
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def forward(self, x):
        return self._interp(x, self.v, self.z).clamp(-self.clip, self.clip)

    def inverse(self, z):
        return self._interp(z, self.z[:, None].expand_as(self.v), self.v)


class Contract:
    """phi = Affine(ZCA of the Gaussianized data) o MarginalGaussianize, or Affine alone."""

    def __init__(self, gauss: MarginalGaussianize | None, affine: Affine):
        self.gauss, self.affine = gauss, affine

    @property
    def dtype(self):
        return self.affine.dtype

    def to(self, device, dtype=torch.float32):
        return Contract(self.gauss.to(device, dtype) if self.gauss else None, self.affine.to(device, dtype))

    def forward(self, x):
        return self.affine.forward(self.gauss.forward(x) if self.gauss else x)

    def inverse(self, z):
        x = self.affine.inverse(z)
        return self.gauss.inverse(x) if self.gauss else x
