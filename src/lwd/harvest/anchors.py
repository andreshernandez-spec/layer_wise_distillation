"""Anchor store (docs/01 C0.3): per-channel std normalization, per-position absmax
scale, int8 codes. Packed offline from the fp16 reference dumps once the interface
statistics exist.

Why this and not fp8 or a Hadamard rotation (measured 22 Aug 2026, synthetic residual
stream with a 200x massive-activation channel, per-channel relative error):

    fp8 plain 0.026   fp8 rotated 0.31   fp8 chan-norm 0.026
    int8 plain 0.42   int8 rotated 0.10  int8 chan-norm 0.007

Rotation spreads the outlier's quantization noise onto every small channel; that is a
win for integer codes with no per-channel scale and a loss for floating-point codes.
Per-channel normalization removes the outlier problem at the source, and int8 then
beats e4m3 by 4x at the same byte per element. `rotation()` stays for the ablation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

INT8_MAX = 127.0
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def rotation(d: int, seed: int = 0) -> torch.Tensor:
    """Haar-random orthogonal (d, d) float32 from a fixed seed (ablation only)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.to(torch.float32)


def quantize(x: torch.Tensor, chan_std: torch.Tensor | None = None, fmt: str = "int8",
             Q: torch.Tensor | None = None):
    """x (..., d) -> codes (..., d) int8|fp8, scales fp16 (..., 1).

    chan_std (d,): divide each channel by it first (None = no normalization).
    Q (d, d): optional rotation applied after normalization (ablation)."""
    y = x.to(torch.float32)
    if chan_std is not None:
        y = y / chan_std.to(y.device, torch.float32)
    if Q is not None:
        y = y @ Q.to(y.device)
    if fmt == "int8":
        s = y.abs().amax(-1, keepdim=True).clamp_min(1e-12) / INT8_MAX
        codes = torch.round(y / s).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    elif fmt == "fp8":
        s = y.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        codes = (y / s).to(FP8)
    else:
        raise ValueError(fmt)
    return codes, s.to(torch.float16)


def dequantize(codes: torch.Tensor, scales: torch.Tensor, chan_std: torch.Tensor | None = None,
               Q: torch.Tensor | None = None) -> torch.Tensor:
    y = codes.to(torch.float32) * scales.to(torch.float32)
    if Q is not None:
        y = y @ Q.T.to(y.device)
    if chan_std is not None:
        y = y * chan_std.to(y.device, torch.float32)
    return y


def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """||x - y|| / ||x||. Fooled by massive-activation channels; see channel_relative_error."""
    x, y = x.to(torch.float64), y.to(torch.float64)
    return float((x - y).norm() / x.norm())


def channel_relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """RMS over channels of (per-channel RMS error / per-channel std)."""
    x, y = x.reshape(-1, x.shape[-1]).to(torch.float64), y.reshape(-1, y.shape[-1]).to(torch.float64)
    err = ((x - y) ** 2).mean(0).sqrt()
    std = x.std(0).clamp_min(1e-12)
    return float(((err / std) ** 2).mean().sqrt())


def pack(ref_files, out_dir, chan_std: torch.Tensor, fmt: str = "int8", Q_seed: int | None = None):
    """fp16 ref dumps (each (b, L, d)) -> one shard per ref file plus meta."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = chan_std.numel()
    Q = rotation(d, Q_seed) if Q_seed is not None else None
    torch.save({"fmt": fmt, "Q_seed": Q_seed, "d": d, "chan_std": chan_std.float().cpu()}, out / "meta.pt")
    for i, f in enumerate(sorted(ref_files)):
        x = torch.from_numpy(np.load(f))
        codes, s = quantize(x, chan_std, fmt, Q)
        np.savez(out / f"shard_{i:05d}.npz", codes=codes.view(torch.uint8).numpy() if fmt == "fp8"
                 else codes.numpy(), scales=s.numpy())


def load(store_dir) -> torch.Tensor:
    """Dequantized anchors (n, L, d) float32 from a packed store."""
    store = Path(store_dir)
    meta = torch.load(store / "meta.pt")
    Q = rotation(meta["d"], meta["Q_seed"]) if meta["Q_seed"] is not None else None
    outs = []
    for f in sorted(store.glob("shard_*.npz")):
        z = np.load(f)
        codes = torch.from_numpy(z["codes"])
        if meta["fmt"] == "fp8":
            codes = codes.view(FP8)
        outs.append(dequantize(codes, torch.from_numpy(z["scales"]), meta["chan_std"], Q))
    return torch.cat(outs)
