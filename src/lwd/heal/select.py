"""Heal a composed student to its validation-selected length (docs/08).

Same rule as the stage trainer and the same code: lwd.stage.select.select_length. Under a
data budget every arm recycles its tokens for as many epochs as help, and how many that is
differs by arm, so the length cannot be fixed in advance and a cosine cannot be sized.
Until this existed no heal had a validation trajectory at all, and nobody knew whether
17.5 epochs was the random arm's optimum or merely its budget.

A snapshot here is 600M trainable weights and their Adam moments. The weights are kept
exactly; the moments are kept in bfloat16, which is three significant digits on the fp32
exponent range and costs a cooldown nothing, and brings a snapshot from 7.2 GB to 4.8 GB.
The window of snapshots a rewind can need is what bounds host memory, which is why the
heal validates every 200 steps with a patience of 5 where a stage uses 100 and 10: the same
1000 steps of patience, half the snapshots.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from lwd.heal.train import TopKStore, kd_and_ce
from lwd.stage.select import select_length


@dataclass
class HealSelectConfig:
    cap_tokens: int = 200_000_000
    batch: int = 1
    seq_len: int = 2048
    lr: float = 1e-4
    warmup: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    w_kd: float = 0.9
    val_every: int = 200
    patience: int = 5
    cooldown_frac: float = 0.1
    min_cooldown: int = 50
    log_every: int = 25
    amp: bool = True
    seed: int = 0
    abort_after_skips: int = 50


def _pack(obj):
    """Host copy of an optimizer state: moments in bfloat16, everything else as it is."""
    if torch.is_tensor(obj):
        t = obj.detach().to("cpu", copy=True)
        return t.to(torch.bfloat16) if t.is_floating_point() and t.numel() > 1 else t
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pack(v) for v in obj)
    return obj


def _unpack(obj):
    if torch.is_tensor(obj):
        return obj.to(torch.float32) if obj.dtype == torch.bfloat16 else obj
    if isinstance(obj, dict):
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unpack(v) for v in obj)
    return obj


def heal_selected(model, store: TopKStore, cfg: HealSelectConfig, val_fn, device="cpu", log=print):
    """val_fn(model) -> float on the budget's validation rows, lower is better. Leaves
    `model` holding the annealed weights of the chosen length; returns (history, summary)."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                             {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    cap_steps = max(1, cfg.cap_tokens // (cfg.batch * cfg.seq_len))
    # one stream for both phases; the store re-permutes and loops, so it never runs dry
    stream = iter(store.batches(cfg.batch, 4 * cfg.cap_tokens + 10 * cfg.seq_len * cfg.batch, cfg.seed))
    t0 = time.time()
    c = {"tokens": 0, "skipped": 0, "consec": 0}

    def snapshot():
        return ({n: p.detach().to("cpu", copy=True) for n, p in named}, _pack(opt.state_dict()))

    def restore(snap):
        with torch.no_grad():
            for n, p in named:
                p.copy_(snap[0][n].to(p.device))
        opt.load_state_dict(_unpack(snap[1]))

    def one_step(step, lr, phase):
        for grp in opt.param_groups:
            grp["lr"] = lr
        tok, ids, logp = next(stream)
        tok, ids, logp = tok.to(device), ids.to(device), logp.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp and device != "cpu"):
            logits = model(input_ids=tok).logits
        kd, ce = kd_and_ce(logits, ids, logp, tok)
        loss = cfg.w_kd * kd + (1 - cfg.w_kd) * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step(); c["consec"] = 0
        else:
            c["skipped"] += 1; c["consec"] += 1
            if c["consec"] >= cfg.abort_after_skips:
                raise RuntimeError(f"heal diverged: {c['consec']} consecutive non-finite steps "
                                   f"from step {step - c['consec'] + 1}")
        c["tokens"] += tok.numel()
        return {"step": step, "phase": phase, "kd": float(kd), "ce": float(ce), "lr": lr,
                "tokens": c["tokens"], "sec": time.time() - t0, "skipped": c["skipped"]}

    def validate():
        model.eval()
        with torch.no_grad():
            v = float(val_fn(model))
        model.train()
        return v

    hist, info = select_length(one_step, validate, snapshot, restore, cap_steps=cap_steps, lr=cfg.lr,
                               warmup=cfg.warmup, val_every=cfg.val_every, patience=cfg.patience,
                               cooldown_frac=cfg.cooldown_frac, min_cooldown=cfg.min_cooldown,
                               log=log, log_every=cfg.log_every)
    model.eval()
    epoch = max(1, store.epoch_tokens())
    per_step = cfg.batch * cfg.seq_len
    summary = {**info, "tokens_seen": c["tokens"], "epochs": c["tokens"] / epoch,
               "tokens_to_best": info["best_step"] * per_step,
               "epochs_to_best": info["best_step"] * per_step / epoch,
               "skipped": c["skipped"], "seconds": time.time() - t0}
    return hist, summary
