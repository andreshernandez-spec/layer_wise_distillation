"""Heal a composed student to its validation-selected best (docs/08).

Same rule as lwd.stage.select, for the same reason: under a data budget every arm recycles
its tokens for as many epochs as help, and how many that is differs by arm, so the length
cannot be fixed in advance and a cosine cannot be sized. Warm up, hold the rate, validate
on the budget's validation rows, stop on patience, cool the best checkpoint to zero, keep
the better of the two. Until this existed no heal had a validation trajectory at all, and
nobody knew whether 17.5 epochs was the random arm's optimum or merely its budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from lwd.heal.train import TopKStore, kd_and_ce
from lwd.stage.select import to_cpu


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
    val_every: int = 100
    patience: int = 8
    cooldown_frac: float = 0.1
    min_cooldown: int = 50
    log_every: int = 25
    amp: bool = True
    seed: int = 0
    abort_after_skips: int = 50


def heal_selected(model, store: TopKStore, cfg: HealSelectConfig, val_fn, device="cpu", log=print):
    """val_fn(model) -> float on the budget's validation rows, lower is better. Leaves
    `model` holding the selected weights; returns (history, summary)."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                             {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))
    seq = cfg.seq_len
    cap_steps = max(1, cfg.cap_tokens // (cfg.batch * seq))
    # one stream for both phases; the store re-permutes and loops, so it never runs dry
    stream = iter(store.batches(cfg.batch, 4 * cfg.cap_tokens + 10 * seq * cfg.batch, cfg.seed))
    hist, t0 = [], time.time()
    c = {"tokens": 0, "skipped": 0, "consec": 0, "validations": 0}

    def snapshot():
        return to_cpu({n: p for n, p in named}), to_cpu(opt.state_dict())

    def restore(snap):
        with torch.no_grad():
            for n, p in named:
                p.copy_(snap[0][n].to(p.device))
        opt.load_state_dict(snap[1])

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
        c["validations"] += 1
        model.eval()
        with torch.no_grad():
            v = float(val_fn(model))
        model.train()
        return v

    best, best_step, snap, bad, stop, step = float("inf"), -1, None, 0, "cap", 0
    while step < cap_steps:
        rec = one_step(step, cfg.lr * min(1.0, (step + 1) / cfg.warmup), "stable")
        step += 1
        if step % cfg.val_every == 0 or step == cap_steps:
            rec["val"] = validate()
            if rec["val"] < best:
                best, best_step, bad, snap = rec["val"], step, 0, snapshot()
            else:
                bad += 1
        hist.append(rec)
        if (step - 1) % cfg.log_every == 0 or "val" in rec:
            log({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in rec.items()})
        if bad >= cfg.patience:
            stop = "patience"
            break
    assert snap is not None, "no validation ran: the cap is below val_every"
    stable_steps = step

    restore(snap)
    n_cool = max(cfg.min_cooldown, int(round(cfg.cooldown_frac * best_step)))
    for i in range(n_cool):
        hist.append(one_step(stable_steps + i, cfg.lr * (1.0 - (i + 1) / n_cool), "cooldown"))
    cooled = validate()
    hist[-1]["val"] = cooled
    selected = "cooled"
    if cooled > best:
        with torch.no_grad():
            for n, p in named:
                p.copy_(snap[0][n].to(p.device))
        selected = "stable-best"
    model.eval()
    epoch = max(1, store.epoch_tokens())
    summary = {"stop": stop, "best_step": best_step, "best_val": best, "cooled_val": cooled,
               "selected": selected, "selected_val": min(best, cooled),
               "stable_steps": stable_steps, "cooldown_steps": n_cool,
               "tokens_seen": c["tokens"], "epochs": c["tokens"] / epoch,
               "tokens_to_best": hist[best_step - 1]["tokens"],
               "epochs_to_best": hist[best_step - 1]["tokens"] / epoch,
               "validations": c["validations"], "skipped": c["skipped"], "seconds": time.time() - t0}
    log({k: v for k, v in summary.items()})
    return hist, summary
