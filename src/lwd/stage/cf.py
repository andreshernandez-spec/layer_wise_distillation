"""Sketched characteristic-function distance as a training loss (docs/08).

The source document's "soft shape" term: M fixed unit directions, K frequencies, and the
squared difference between two empirical characteristic functions on that grid. Here it is
a two-sample statistic between the student's and the teacher's outputs on the SAME batch,
so its optimum is the mimic optimum whatever the inputs were (real or noise). That is the
condition the source document puts on any regularizer used as a loss.

It lives in the whitened output coordinates. The sketch the harvest accumulates is in raw
coordinates and is not usable as a uniform target: at interface 0 every |cf| is 1.000 and
at interface 6 it is below 0.07 past the first few frequencies, because the raw scale
changes by orders of magnitude with depth. In whitened coordinates a projection has unit
variance at every interface and t in [0.5, 2] probes the bulk of the distribution.

What it can add to a per-position MSE: an MSE-trained student predicts conditional means,
so its outputs are under-dispersed, and the next stage then reads activations whose
marginal distribution is off. Matching the marginal along random directions penalizes
exactly that, and costs M*K cosines per position.
"""
from __future__ import annotations

import torch


class CFDistance(torch.nn.Module):
    def __init__(self, d: int, M: int = 64, freqs=(0.5, 1.0, 1.5, 2.0), seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(M, d, generator=g)
        self.register_buffer("U", U / U.norm(dim=1, keepdim=True))
        self.register_buffer("t", torch.tensor(freqs, dtype=torch.float32))

    def sketch(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d) -> (M, K, 2): mean cos and sin of t_k <x, u_m> over all positions."""
        p = x.reshape(-1, x.shape[-1]).float() @ self.U.T             # (n, M)
        a = p[:, :, None] * self.t[None, None, :]                      # (n, M, K)
        return torch.stack([torch.cos(a).mean(0), torch.sin(a).mean(0)], dim=-1)

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """Mean over directions and frequencies of |phi_s - phi_t|^2. Zero iff the two
        sketches agree; differentiable in student_out."""
        with torch.no_grad():
            ref = self.sketch(teacher_out)
        return ((self.sketch(student_out) - ref) ** 2).sum(-1).mean()
