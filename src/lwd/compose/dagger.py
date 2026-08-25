"""DAgger for stagewise distillation (docs/03 C2.3).

Each stage is trained on the teacher's clean interface and then evaluated on the
interface the *student* stack actually produces. That gap is exposure bias, and the
C2.2 decomposition says it is part of the ~0.54 of fresh error every stage adds.

The fix is the standard one from imitation learning: train stage k on the distribution
it will meet, i.e. anchors propagated through the already-trained stages below it.
Note this propagates **anchors, not noise** (`docs/03` "What Phase 1 changes here"):
a student trained on pure noise diverges on real activations, so noise pushed through
imperfect stages is further off-manifold than where it started.

The target stays the teacher's: for input x at interface k, the target is the teacher
stage k applied to that same x. Only the input distribution moves.
"""
from __future__ import annotations

import numpy as np
import torch


class PropagatedSampler:
    """Interface-k inputs as the student stack produces them, mixed with the teacher's.

    lower: callable ids -> teacher interface 0 (embedding).
    stages: the already-trained raw-coordinate stages 0..k-1 (inference only).
    teacher_lower: callable ids -> teacher interface k, for the clean fraction.
    on_policy: fraction of each batch drawn from the student's own propagation.
    """

    def __init__(self, rows, lower, stages, teacher_lower, on_policy: float = 0.5):
        self.rows, self.lower, self.stages = rows, lower, stages
        self.teacher_lower, self.on_policy = teacher_lower, on_policy

    @torch.no_grad()
    def sample(self, b, L, g, device="cpu"):
        n_on = int(round(b * self.on_policy))
        idx = torch.randint(0, len(self.rows), (b,), generator=g).numpy()
        ids = torch.from_numpy(self.rows[idx, :L].astype(np.int64)).to(device)
        out = []
        if n_on:
            h = self.lower(ids[:n_on])
            for st in self.stages:
                h = st(h)
            out.append(h.float())
        if b - n_on:
            out.append(self.teacher_lower(ids[n_on:]).float())
        return torch.cat(out) if len(out) > 1 else out[0]
