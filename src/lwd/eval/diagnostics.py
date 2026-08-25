"""Phase 1 diagnostics (docs/02 measurements 3 and 4).

attention_entropy_under(stage, x): mean attention entropy per head per block of a
  teacher StageRunner (eager attention) on inputs x, in nats. Compare against the same
  stage on real inputs: near log(L) means the arm gives attention nothing to do.

jacobian_agreement(student, teacher, phi_in, phi_out, x, n_dirs): cosine between
  student and transformed-teacher JVPs on real inputs x at random directions, per
  position averaged. Srinivas-Fleuret says output matching under input noise is
  Jacobian matching; this checks whether it happened.
"""
from __future__ import annotations

import torch

from lwd.harvest.stats import attention_entropy


@torch.no_grad()
def attention_entropy_under(stage, x: torch.Tensor) -> torch.Tensor:
    """stage: StageRunner built with attn='eager'. Returns (n_blocks, n_heads)."""
    assert stage.config._attn_implementation == "eager", "needs eager attention for weights"
    caught = []
    hs = [layer.attention.register_forward_hook(lambda m, a, o: caught.append(o[1]))
          for layer in stage.layers]
    try:
        stage(x)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([attention_entropy(w) for w in caught])


def _phi(phi, x, inverse=False):
    if phi is None:
        return x
    f = phi.inverse if inverse else phi.forward
    s = x.shape
    return f(x.reshape(-1, s[-1]).to(phi.dtype)).reshape(*s[:-1], -1).to(x.dtype)


def jacobian_agreement(student, teacher, phi_in, phi_out, x: torch.Tensor, n_dirs: int = 16,
                       seed: int = 0) -> dict:
    """x (b, L, d) raw inputs. Directions are random in the student's input space; the
    teacher's JVP is taken through phi_out o T o phi_in^-1 so both live in the same
    coordinates. Returns mean cosine and mean JVP norm ratio.

    Both models are switched to eager attention for the duration: forward-mode AD and
    double-backward are not implemented for the fused SDPA kernels (CPU flash at least),
    and the HF attention kernel is looked up from the config at forward time."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = _phi(phi_in, x.float()).detach()
    tdt = next(teacher.parameters()).dtype
    cfgs = [m.config for m in (student, teacher) if hasattr(m, "config")]
    saved = [c._attn_implementation for c in cfgs]
    for c in cfgs:
        c._attn_implementation = "eager"

    def f_teacher(zz):
        return _phi(phi_out, teacher(_phi(phi_in, zz, inverse=True).to(tdt)).float())

    def f_student(zz):
        return student(zz).float()

    cos, ratio = [], []
    try:
        for _ in range(n_dirs):
            v = torch.randn(z.shape, generator=g).to(z.device)
            _, jt = torch.func.jvp(f_teacher, (z,), (v,))
            _, js = torch.func.jvp(f_student, (z,), (v,))
            jt, js = jt.reshape(-1), js.reshape(-1)
            cos.append(float(torch.dot(jt, js) / (jt.norm() * js.norm() + 1e-12)))
            ratio.append(float(js.norm() / (jt.norm() + 1e-12)))
    finally:
        for c, a in zip(cfgs, saved):
            c._attn_implementation = a
    return {"cos_mean": sum(cos) / len(cos), "cos_min": min(cos), "norm_ratio": sum(ratio) / len(ratio)}
