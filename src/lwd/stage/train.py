"""Train one student stage against one teacher stage on a sampler (docs/02)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from lwd.stage.student import rel_mse


@dataclass
class TrainConfig:
    steps: int = 1000
    batch: int = 8
    seq_len: int = 2048
    lr: float = 3e-4
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    log_every: int = 20
    amp: bool = True
    seed: int = 0


def _apply(phi, x):
    if phi is None:
        return x
    shape = x.shape
    return phi.forward(x.reshape(-1, shape[-1]).to(phi.dtype)).reshape(*shape[:-1], -1).to(x.dtype)


@torch.no_grad()
def evaluate(student, teacher, X: torch.Tensor, phi_in, phi_out, batch: int, device) -> float:
    """Relative MSE in transformed coordinates on held-out real interface tensors X
    (n, L, d). Teacher outputs are recomputed (exact)."""
    student.eval()
    num = den = 0.0
    for i in range(0, X.shape[0], batch):
        x = X[i:i + batch].to(device)
        t = _apply(phi_out, teacher(x.to(next(teacher.parameters()).dtype)).float())
        p = student(_apply(phi_in, x.float())).float()
        num += float(((p - t) ** 2).sum())
        den += float(((t - t.reshape(-1, t.shape[-1]).mean(0)) ** 2).sum())
    student.train()
    return num / max(den, 1e-12)


def train_stage(student, teacher, sampler, cfg: TrainConfig, phi_in=None, phi_out=None,
                eval_X: torch.Tensor | None = None, device="cpu", log=print):
    """sampler.sample(b, L, g, device) -> (b, L, d) raw-coordinate inputs (or a
    (tensor, n_real) pair from AnchorMix). Returns the history list."""
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    student.to(device).train()
    teacher.eval()
    tdt = next(teacher.parameters()).dtype
    decay = [p for n, p in student.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in student.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    use_amp = cfg.amp and device != "cpu"
    hist, t0, seen = [], time.time(), 0
    for step in range(cfg.steps):
        x = sampler.sample(cfg.batch, cfg.seq_len, g, device)
        if isinstance(x, tuple):
            x = x[0]
        with torch.no_grad():
            t = _apply(phi_out, teacher(x.to(tdt)).float())
        z = _apply(phi_in, x.float())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            p = student(z)
        loss = rel_mse(p.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        seen += cfg.batch * cfg.seq_len
        rec = {"step": step, "loss": float(loss), "positions": seen, "lr": sched.get_last_lr()[0],
               "sec": time.time() - t0}
        if eval_X is not None and (step % cfg.eval_every == 0 or step == cfg.steps - 1):
            rec["eval"] = evaluate(student, teacher, eval_X, phi_in, phi_out, cfg.batch, device)
        hist.append(rec)
        if step % cfg.log_every == 0 or "eval" in rec:
            log({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()})
    return hist
