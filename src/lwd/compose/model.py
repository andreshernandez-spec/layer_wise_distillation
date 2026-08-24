"""A composed student language model: the teacher's embedding and head with the
distilled stages in between (docs/03 C2.1).

Phase 2's student is same-width, so the embedding, final norm and head transfer from
the teacher unchanged and only the blocks are distilled. Pythia-1.4B has 24 blocks in
6 stages of 4; the student has 2 blocks per stage, so 12 blocks: a 2x depth
compression at equal width, and every parameter that differs from the teacher was
produced by stagewise training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lwd.compose.chain import Wrapped


@dataclass
class LMOutput:
    logits: torch.Tensor


class StudentLM(nn.Module):
    """edges: lwd.harvest.model.Edges (teacher embedding + final norm + head).
    stages: raw-coordinate stage modules, e.g. Wrapped(student, phi_in, phi_out).
    Accepts `input_ids=` so it drops into the same eval helpers as a HF model."""

    def __init__(self, edges, stages):
        super().__init__()
        self.edges = edges
        self.stages = nn.ModuleList(stages)

    def forward(self, input_ids: torch.Tensor, **_) -> LMOutput:
        h = self.edges.embed(input_ids)
        for st in self.stages:
            h = st(h)
        return LMOutput(logits=self.edges.head(h))

    @torch.no_grad()
    def interfaces(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """[interface 0, ..., interface S] as the student produces them."""
        h = self.edges.embed(input_ids)
        out = [h]
        for st in self.stages:
            h = st(h)
            out.append(h)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.stages.parameters())


def build(model_id, stage_state_dicts, phis, cfgs, dtype=torch.float16, device="cpu"):
    """Assemble from saved stage state dicts and the per-interface contract maps."""
    from lwd.harvest.model import Edges
    from lwd.stage.student import StudentStage
    edges = Edges(model_id, dtype, device)
    stages = []
    for k, sd in enumerate(stage_state_dicts):
        st = StudentStage(cfgs[k], seed=0)
        st.load_state_dict(sd)
        st.to(device).eval()
        stages.append(Wrapped(st, phis[k], phis[k + 1]))
    return StudentLM(edges, stages).to(device).eval()
