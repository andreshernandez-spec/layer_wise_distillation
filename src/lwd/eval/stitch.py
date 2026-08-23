"""Stitching loss (docs/02 measurement 2): run the teacher with stage k replaced by a
student stage wrapped in the contract maps, and measure next-token loss on held-out
text against the unmodified teacher.

    x -> phi_in -> student -> phi_out^{-1} -> next teacher stage
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Stitched(torch.nn.Module):
    """Replaces blocks [a, b) of a GPTNeoXForCausalLM with `student` in contract coords."""

    def __init__(self, model, a: int, b: int, student, phi_in, phi_out):
        super().__init__()
        self.model, self.a, self.b = model, a, b
        self.student, self.phi_in, self.phi_out = student, phi_in, phi_out
        self._handles = []

    def _apply_phi(self, phi, x, inverse=False):
        shape = x.shape
        f = phi.inverse if inverse else phi.forward
        return f(x.reshape(-1, shape[-1]).to(phi.dtype)).reshape(*shape[:-1], -1).to(x.dtype)

    def __enter__(self):
        layers = self.model.gpt_neox.layers

        def replace(mod, args, kwargs):
            h = args[0]
            z = self._apply_phi(self.phi_in, h.float())
            s = self.student(z).float()
            out = self._apply_phi(self.phi_out, s, inverse=True).to(h.dtype)
            self._stash = out
            return (out,) + tuple(args[1:]), kwargs

        def skip(mod, args, kwargs):
            # blocks a+1..b-1 are bypassed: pass the stashed output straight through
            return (self._stash,) + tuple(args[1:]), kwargs

        def identity_out(mod, args, output):
            return self._stash

        # block a receives the student's output as its input and must not act: make it
        # an identity by stashing and returning the stash from its output hook
        self._handles.append(layers[self.a].register_forward_pre_hook(replace, with_kwargs=True))
        self._handles.append(layers[self.a].register_forward_hook(identity_out))
        for i in range(self.a + 1, self.b):
            self._handles.append(layers[i].register_forward_pre_hook(skip, with_kwargs=True))
            self._handles.append(layers[i].register_forward_hook(identity_out))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def next_token_loss(model, ids: torch.Tensor, batch: int = 4) -> float:
    """Mean next-token cross-entropy over all positions of ids (n, L+1)."""
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], batch):
        x = ids[i:i + batch].to(next(model.parameters()).device)
        logits = model(input_ids=x[:, :-1]).logits
        for j in range(logits.shape[0]):  # one sequence at a time: (b, L, V) float32 is 3 GB at b=8
            lj = logits[j].float()
            tot += float(F.cross_entropy(lj, x[j, 1:], reduction="sum"))
        n += x[:, 1:].numel()
        del logits
    return tot / n


@torch.no_grad()
def stitching_delta(model, a, b, student, phi_in, phi_out, ids, batch=4):
    """(teacher loss, stitched loss, delta) on held-out token rows."""
    base = next_token_loss(model, ids, batch)
    student.eval()
    with Stitched(model, a, b, student, phi_in, phi_out):
        st = next_token_loss(model, ids, batch)
    return {"teacher": base, "stitched": st, "delta": st - base}
