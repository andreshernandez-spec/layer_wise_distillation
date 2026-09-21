"""Train a stage to its validation-selected best (docs/08).

train.py anneals a cosine over a length fixed in advance. That is the wrong tool once
every arm has to be stopped where ITS validation says: a checkpoint lifted out of the
middle of a long cosine run is at near-peak learning rate, so the arm whose optimum comes
early (anchors only, at scarce data: step 600 of 6104) is read at its worst. The schedule
here does not presuppose the length: warm up, hold the rate, validate as it goes, stop on
patience, then cool the best checkpoint down to zero and keep whichever of the two
validates better. Every arm gets the same rule, so none is handicapped by when it peaks.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch

from lwd.stage.student import rel_mse
from lwd.stage.train import _apply


@dataclass
class SelectConfig:
    cap_steps: int = 6104            # 1e8 positions at batch 8 x 2048, the source document's budget
    batch: int = 8
    seq_len: int = 2048
    lr: float = 3e-4
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 100
    patience: int = 8                # validations without a new best before stopping
    cooldown_frac: float = 0.1       # linear decay to zero over this fraction of best_step
    min_cooldown: int = 50
    cf_lambda: float = 0.0
    log_every: int = 20
    amp: bool = True
    seed: int = 0


def to_cpu(obj):
    """A deep copy with every tensor on the host. deepcopy alone leaves an optimizer's
    moments on the GPU, which doubles them there for as long as the snapshot lives."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _snapshot(student, opt):
    return to_cpu(student.state_dict()), to_cpu(opt.state_dict())


def _restore(student, opt, snap, device):
    student.load_state_dict(snap[0])
    student.to(device)
    opt.load_state_dict(snap[1])


def train_selected(student, teacher, sampler, cfg: SelectConfig, val_fn, phi_in=None,
                   phi_out=None, cf=None, device="cpu", log=print):
    """val_fn(student) -> float, lower is better, deterministic. Leaves `student` holding
    the selected weights and returns (history, summary)."""
    assert cfg.cf_lambda == 0.0 or cf is not None, "cf_lambda > 0 needs a CFDistance"
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    student.to(device).train()
    teacher.eval()
    tdt = next(teacher.parameters()).dtype
    decay = [p for _, p in student.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in student.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    use_amp = cfg.amp and device != "cpu"
    hist, t0 = [], time.time()
    count = {"real": 0, "noise": 0, "skipped": 0, "validations": 0}

    def set_lr(v):
        for grp in opt.param_groups:
            grp["lr"] = v

    def one_step(step, lr, phase):
        set_lr(lr)
        x = sampler.sample(cfg.batch, cfg.seq_len, g, device)
        n_real = cfg.batch
        if isinstance(x, tuple):
            x, n_real = x
        with torch.no_grad():
            t = _apply(phi_out, teacher(x.to(tdt)).float())
        z = _apply(phi_in, x.float())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            p = student(z)
        p = p.float()
        mse = rel_mse(p, t)
        cfd = cf(p, t) if cfg.cf_lambda > 0 else None
        loss = mse if cfd is None else mse + cfg.cf_lambda * cfd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        if torch.isfinite(loss) and torch.isfinite(gnorm):      # see train.py: one bad
            opt.step()                                           # batch must not poison
        else:                                                    # the weights for good
            count["skipped"] += 1
            opt.zero_grad(set_to_none=True)
        count["real"] += n_real * cfg.seq_len
        count["noise"] += (cfg.batch - n_real) * cfg.seq_len
        rec = {"step": step, "phase": phase, "loss": float(loss), "mse": float(mse), "lr": lr,
               "positions": count["real"] + count["noise"], "sec": time.time() - t0}
        if cfd is not None:
            rec["cf"] = float(cfd)
        return rec

    def validate():
        count["validations"] += 1
        student.eval()
        with torch.no_grad():
            v = float(val_fn(student))
        student.train()
        return v

    best, best_step, snap, bad, stop = float("inf"), -1, None, 0, "cap"
    step = 0
    while step < cfg.cap_steps:
        rec = one_step(step, cfg.lr * min(1.0, (step + 1) / cfg.warmup), "stable")
        step += 1
        if step % cfg.val_every == 0 or step == cfg.cap_steps:
            rec["val"] = validate()
            if rec["val"] < best:
                best, best_step, bad = rec["val"], step, 0
                snap = _snapshot(student, opt)
            else:
                bad += 1
        hist.append(rec)
        if (step - 1) % cfg.log_every == 0 or "val" in rec:
            log({k: (round(v, 5) if isinstance(v, float) and k != "lr" else v) for k, v in rec.items()})
        if bad >= cfg.patience:
            stop = "patience"
            break
    assert snap is not None, "no validation ran: cap_steps is below val_every"
    stable_steps = step

    # Cool the best checkpoint down. The data stream simply continues; only the weights
    # and the optimizer go back to where validation was best.
    _restore(student, opt, snap, device)
    n_cool = max(cfg.min_cooldown, int(round(cfg.cooldown_frac * best_step)))
    for i in range(n_cool):
        rec = one_step(stable_steps + i, cfg.lr * (1.0 - (i + 1) / n_cool), "cooldown")
        hist.append(rec)
    cooled = validate()
    hist[-1]["val"] = cooled
    log({"cooldown_steps": n_cool, "val_before": round(best, 5), "val_after": round(cooled, 5)})
    selected = "cooled"
    if cooled > best:                    # cooling did not help: hand back the snapshot
        student.load_state_dict(snap[0])
        student.to(device)
        selected = "stable-best"
    student.eval()
    total = count["real"] + count["noise"]
    summary = {"stop": stop, "best_step": best_step, "best_val": best, "cooled_val": cooled,
               "selected": selected, "selected_val": min(best, cooled),
               "stable_steps": stable_steps, "cooldown_steps": n_cool,
               "positions": total, "real_positions": count["real"], "noise_positions": count["noise"],
               "achieved_m": (count["noise"] / count["real"]) if count["real"] else float("inf"),
               "teacher_positions": total, "validations": count["validations"],
               "skipped": count["skipped"], "seconds": time.time() - t0}
    return hist, summary
