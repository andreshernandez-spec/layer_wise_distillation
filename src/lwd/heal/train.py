"""End-to-end heal (docs/03 C2.4): fine-tune a composed student on the real slice
against the teacher's stored top-k distribution plus the true next token.

The teacher is never resident. Everything it contributes was harvested once
(`docs/01` C0.4): top-64 ids and log-probs per position, covering 93.9% of the mass
on this slice.
"""
from __future__ import annotations

import glob
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class HealConfig:
    tokens: int = 10_000_000
    batch: int = 2
    lr: float = 1e-4
    warmup: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    w_kd: float = 0.9        # weight on the teacher term; 1 - w_kd on true-token CE
    eval_every: int = 200
    log_every: int = 20
    amp: bool = True
    seed: int = 0


def kd_and_ce(logits: torch.Tensor, tk_ids: torch.Tensor, tk_logp: torch.Tensor,
              tokens: torch.Tensor):
    """Forward KL to the teacher, over the k stored tokens plus one bucket for
    everything else, and CE to the true next token. Returns (kd, ce).

    The bucket is what makes this a real KL. The stored top-k carries ~94% of the
    teacher's mass, so the distribution is (p_1..p_k, 1 - sum p) against
    (q_1..q_k, 1 - sum q), which is zero exactly when the student matches the teacher
    on the stored support *and* leaks no more mass outside it. Renormalizing the
    teacher alone and comparing against the student's full softmax is not a KL at all:
    it floors at -log(top-k mass), 0.22 nats here, and gives the student no reason to
    keep its mass on the teacher's support."""
    eps = 1e-6
    lsm = logits.float().log_softmax(-1)
    s_logp = lsm.gather(-1, tk_ids.long())                      # (b, L, k)
    t_logp = tk_logp.float()
    t_p, s_p = t_logp.exp(), s_logp.exp()
    t_rest = (1 - t_p.sum(-1)).clamp_min(eps)
    s_rest = (1 - s_p.sum(-1)).clamp_min(eps)
    kd = ((t_p * (t_logp - s_logp)).sum(-1) + t_rest * (t_rest.log() - s_rest.log())).mean()
    ce = F.nll_loss(lsm[:, :-1].reshape(-1, lsm.shape[-1]), tokens[:, 1:].reshape(-1))
    return kd, ce


class TopKStore:
    """The harvested chunks, as batches of (tokens, top-k ids, top-k log-probs)."""

    def __init__(self, path: str, exclude_rows: set[int] | None = None):
        self.files = sorted(glob.glob(f"{path}/*.npz"))
        if not self.files:
            # an empty store used to yield nothing, so heal() ran zero steps and
            # reported success with loss_before == loss_after (24 Aug 2026: a pod
            # harvested with skip_topk produced twelve "finished" heal runs that had
            # trained on nothing at all)
            raise FileNotFoundError(
                f"no top-k chunks in {path}: the heal has no teacher to match. "
                "Harvest with skip_topk unset.")
        self.exclude = exclude_rows or set()

    def batches(self, batch: int, max_tokens: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.files))
        seen = 0
        for fi in order:
            z = np.load(self.files[fi])
            tok, ids, logp = z["tokens"], z["ids"], z["logp"]
            for i in range(0, tok.shape[0], batch):
                t = torch.from_numpy(tok[i:i + batch].astype(np.int64))
                yield (t, torch.from_numpy(ids[i:i + batch].astype(np.int64)),
                       torch.from_numpy(logp[i:i + batch]))
                seen += t.numel()
                if seen >= max_tokens:
                    return


def heal(model, store: TopKStore, cfg: HealConfig, eval_fn=None, device="cpu", log=print):
    """Fine-tune `model` (a StudentLM) end to end. Returns the history."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    nodecay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    steps = max(1, cfg.tokens // (cfg.batch * 2048))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    hist, t0, seen, skipped = [], time.time(), 0, 0
    for step, (tok, ids, logp) in enumerate(store.batches(cfg.batch, cfg.tokens, cfg.seed)):
        tok, ids, logp = tok.to(device), ids.to(device), logp.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp and device != "cpu"):
            logits = model(input_ids=tok).logits
        kd, ce = kd_and_ce(logits, ids, logp, tok)
        loss = cfg.w_kd * kd + (1 - cfg.w_kd) * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if torch.isfinite(loss) and torch.isfinite(gn):   # docs/02: one NaN poisons a run
            opt.step()
        else:
            skipped += 1
        sched.step()
        seen += tok.numel()
        rec = {"step": step, "kd": float(kd), "ce": float(ce), "tokens": seen,
               "lr": sched.get_last_lr()[0], "sec": time.time() - t0, "skipped": skipped}
        if eval_fn is not None and (step % cfg.eval_every == 0):
            model.eval(); rec["eval"] = eval_fn(model); model.train()
        if step % cfg.log_every == 0 or "eval" in rec:
            log({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()})
        hist.append(rec)
    if eval_fn is not None:
        model.eval(); hist[-1]["eval"] = eval_fn(model)
    return hist
