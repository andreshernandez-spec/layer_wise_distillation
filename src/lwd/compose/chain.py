"""Composition (docs/03 C2.1): chain stage modules through the interface maps.

Stage k maps interface k to interface k+1 in raw teacher coordinates. A student stage
S_k works in contract coordinates: raw -> phi_k -> S_k -> phi_{k+1}^{-1} -> raw. Chaining
student stages therefore inserts phi_{k+1}^{-1} o phi_{k+1} at every interior interface,
which is the identity: the maps telescope. `Chain` makes that explicit so the teacher's
own stages, wrapped the same way, reproduce the teacher to float precision, and a
deployed student can fold each affine phi into its neighbouring linear layers later.
"""
from __future__ import annotations

import torch
from torch import nn


def _phi(phi, x, inverse=False):
    if phi is None:
        return x
    f = phi.inverse if inverse else phi.forward
    s = x.shape
    return f(x.reshape(-1, s[-1]).to(phi.dtype)).reshape(*s[:-1], -1).to(x.dtype)


class Wrapped(nn.Module):
    """phi_out^{-1} o S o phi_in: a contract-coordinate stage seen from raw coordinates."""

    def __init__(self, stage, phi_in, phi_out):
        super().__init__()
        self.stage, self.phi_in, self.phi_out = stage, phi_in, phi_out

    def forward(self, x):
        sdt = next(self.stage.parameters()).dtype
        return _phi(self.phi_out, self.stage(_phi(self.phi_in, x.float()).to(sdt)).float(), inverse=True).to(x.dtype)


class Chain(nn.Module):
    """stages: list of raw-coordinate stage modules (Wrapped students, teacher
    StageRunners, or a mix). forward(x0) -> list of interfaces [x0, x1, ..., xS]."""

    def __init__(self, stages):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x, upto: int | None = None):
        out = [x]
        for k, st in enumerate(self.stages[:upto]):
            x = st(x)
            out.append(x)
        return out


@torch.no_grad()
def drift_profile(chain: Chain, teacher_ifaces: list[torch.Tensor], phis) -> list[float]:
    """Per interface k >= 1: relative MSE between the chain's interface k (propagated
    from the teacher's interface 0) and the teacher's, in phi_k coordinates."""
    ours = chain(teacher_ifaces[0])
    out = []
    for k in range(1, len(teacher_ifaces)):
        a, b = _phi(phis[k], ours[k].float()), _phi(phis[k], teacher_ifaces[k].float())
        bb = b.reshape(-1, b.shape[-1])
        out.append(float(((a - b) ** 2).sum() / ((bb - bb.mean(0)) ** 2).sum()))
    return out
