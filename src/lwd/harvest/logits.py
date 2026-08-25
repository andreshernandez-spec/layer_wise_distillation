"""Top-k logit store (docs/01 C0.4): uint16 ids + fp16 logits, plus the teacher's
full-vocab loss so truncation error is checkable."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def topk_pack(logits: torch.Tensor, k: int = 64):
    """logits (b, L, V) -> ids uint16 (b, L, k), logp fp16 (b, L, k), logsumexp f32 (b, L).

    Log-probs, not raw logits: raw logits reach magnitude ~200 where fp16 spacing is
    0.25 nats. Log-probs sit in [-30, 0] where fp16 spacing is <= 0.016 nats."""
    assert logits.shape[-1] < 65536
    lf = logits.float()
    lse = torch.logsumexp(lf, -1)
    vals, ids = lf.topk(k, dim=-1)
    logp = vals - lse[..., None]
    return ids.to(torch.int32).to(torch.uint16), logp.to(torch.float16), lse


def topk_coverage(logp: torch.Tensor) -> torch.Tensor:
    """Probability mass held by the stored top-k, per position."""
    return torch.exp(logp.float()).sum(-1)


class TopKWriter:
    def __init__(self, path, k: int = 64):
        self.path, self.k, self._n = Path(path), k, 0
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, logits: torch.Tensor, ids_in: torch.Tensor):
        # one sequence at a time: a (b, L, V) float32 copy is 3 GB at b=8, L=2048
        parts = [topk_pack(logits[i:i + 1], self.k) for i in range(logits.shape[0])]
        ids, logp, lse = (torch.cat([p[j] for p in parts]) for j in range(3))
        np.savez(self.path / f"chunk_{self._n:05d}.npz", ids=ids.cpu().numpy(),
                 logp=logp.cpu().numpy(), lse=lse.cpu().numpy(), tokens=ids_in.cpu().numpy())
        self._n += 1
