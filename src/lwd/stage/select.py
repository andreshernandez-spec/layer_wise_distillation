"""Train a stage to its validation-selected length (docs/08).

train.py anneals a cosine over a length fixed in advance. That is the wrong tool once
every arm has to be stopped where ITS validation says: a checkpoint lifted out of the
middle of a long cosine run is at near-peak learning rate, so the arm whose optimum comes
early (anchors only, at scarce data) is read at its worst.

So the length is chosen and then annealed to. Warm up, hold the rate, validate as it goes,
stop on patience. The length is the step where the SMOOTHED validation curve is lowest,
and the model is the one obtained by rewinding to the checkpoint a cooldown before that
step and annealing the rate to zero so the run ENDS there. That is what a schedule sized
for that length would have produced, without knowing the length in advance.

The first version of this rule (21 Sep 2026, one afternoon) cooled down from the best
checkpoint itself and kept whichever of the two validated better. On the first cell it
finished, that went from 1.20 to 2.21: the best step sat at the edge of overfitting 44
real sequences, and a cooldown that starts there trains further in. Worse, the 1.20 was a
single dip between 1.76 and 2.36. Both faults point the same way. An arm that peaks early
could never be annealed under that rule, while an arm that runs to the cap always was, and
the arm that peaks earliest is the control. Smoothing, rewinding and always taking the
annealed model removes the asymmetry and the lucky dip together.
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
    patience: int = 8                # validations without a new smoothed best before stopping
    cooldown_frac: float = 0.1       # the anneal is this fraction of the chosen length,
    min_cooldown: int = 50           # rounded to the validation grid (it starts on a snapshot)
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

    def n_cool(length):
        grid = cfg.val_every
        return max(grid, int(round(cfg.cooldown_frac * length / grid)) * grid, 
                   -(-cfg.min_cooldown // grid) * grid)

    def smoothed(vals, i):
        """Centred three-point mean; two points at either end. One validation of five
        sequences at a held learning rate moves by a nat between neighbours."""
        lo, hi = max(0, i - 1), min(len(vals) - 1, i + 1)
        return sum(vals[lo:hi + 1]) / (hi - lo + 1)

    snaps = {0: _snapshot(student, opt)}             # step -> state; a rewind can reach step 0
    vsteps, vals = [], []
    best_i, stop, step = None, "cap", 0
    while step < cfg.cap_steps:
        rec = one_step(step, cfg.lr * min(1.0, (step + 1) / cfg.warmup), "stable")
        step += 1
        at_cap = step == cfg.cap_steps
        if step % cfg.val_every == 0 or at_cap:
            rec["val"] = validate()
            vsteps.append(step); vals.append(rec["val"])
            snaps[step] = _snapshot(student, opt)
            # a smoothed value is final once its right-hand neighbour exists
            final = len(vals) if at_cap else len(vals) - 1
            if final > 0:
                best_i = min(range(final), key=lambda i: smoothed(vals, i))
                need = vsteps[best_i] - n_cool(vsteps[best_i])      # where its rewind starts;
                for k in [k for k in snaps if k < need]:            # later bests rewind from
                    del snaps[k]                                    # later still, so prune
                if final - 1 - best_i >= cfg.patience:
                    stop = "patience"
        hist.append(rec)
        if (step - 1) % cfg.log_every == 0 or "val" in rec:
            log({k: (round(v, 5) if isinstance(v, float) and k != "lr" else v) for k, v in rec.items()})
        if stop == "patience":
            break
    assert best_i is not None, "fewer than two validations ran: cap_steps is below 2 x val_every"
    stable_steps = step
    length = vsteps[best_i]
    cool = min(n_cool(length), length)
    start = length - cool
    start = max(k for k in snaps if k <= start)      # on the grid by construction; 0 at worst
    cool = length - start

    # Rewind and anneal so that the run ends at the chosen length. The data stream simply
    # continues; only the weights and the optimizer go back.
    _restore(student, opt, snaps[start], device)
    del snaps
    for i in range(cool):
        s_abs = start + i
        lr = cfg.lr * min(1.0, (s_abs + 1) / cfg.warmup) * (1.0 - (i + 1) / cool)
        hist.append(one_step(s_abs, lr, "cooldown"))
    cooled = validate()
    hist[-1]["val"] = cooled
    log({"length": length, "rewound_to": start, "cooldown_steps": cool,
         "smoothed_best": round(smoothed(vals, best_i), 5), "val_annealed": round(cooled, 5)})
    student.eval()
    total = count["real"] + count["noise"]
    summary = {"stop": stop, "best_step": length, "rewound_to": start, "cooldown_steps": cool,
               "smoothed_best_val": smoothed(vals, best_i), "raw_val_at_best_step": vals[best_i],
               "cooled_val": cooled, "selected": "annealed", "selected_val": cooled,
               "stable_steps": stable_steps, "stable_vals": list(zip(vsteps, vals)),
               "positions": total, "real_positions": count["real"], "noise_positions": count["noise"],
               "achieved_m": (count["noise"] / count["real"]) if count["real"] else float("inf"),
               "teacher_positions": total, "validations": count["validations"],
               "skipped": count["skipped"], "seconds": time.time() - t0}
    return hist, summary
