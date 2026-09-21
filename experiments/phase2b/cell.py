"""One single-stage cell of the data-limited protocol (docs/08, step 1).

    python experiments/phase2b/cell.py CONFIG --m 8 --cf-lambda 0 --seed 0

Everything real comes from one budgeted harvest (experiments/phase0/run.py with
budget_rows): the contract and the noise moments from its statistics, the anchors from its
training rows, and the validation rows that decide when to stop. The noise ratio m is the
dose and m = 0 is the control. Selection is on validation stitching delta; the number that
is reported is the stitching delta on held-out Pile-test rows that nothing here selects on.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from run import build_contract, load_refs                      # noqa: E402

from lwd.eval.stitch import Stitched, next_token_loss
from lwd.harvest.model import StageRunner, stage_bounds
from lwd.noise.samplers import ContractGaussianized, DoseMix
from lwd.stage.cf import CFDistance
from lwd.stage.select import SelectConfig, train_selected
from lwd.stage.student import StudentStage, student_config
from lwd.stage.train import _apply, evaluate

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def cell_name(budget_rows, stage, m, lam, wd, seed):
    return f"b{budget_rows}_st{stage}_m{m:g}_cf{lam:g}_wd{wd:g}_s{seed}"


def main(a):
    c = yaml.safe_load(open(a.config))
    # step 2 trains every stage of a stack from one config: the stage, and the harvest
    # that carries anchors for all six inputs, come from the command line
    if a.stage is not None:
        c["stage"] = a.stage
    if a.harvest:
        c["harvest"] = a.harvest
    if a.out:
        c["out"] = a.out
    out = Path(c["out"]); out.mkdir(parents=True, exist_ok=True)
    harvest = Path(c["harvest"])
    hmeta = json.load(open(harvest / "run.json"))
    budget = hmeta["budget"]
    assert budget, f"{harvest} is not a budgeted harvest (docs/08): statistics would leak"
    k, dev, dt = c["stage"], c["device"], DT[c["dtype"]]
    wd = a.wd if a.wd is not None else c["train"].get("weight_decay", 0.1)
    name = cell_name(budget["budget_rows"], k, a.m, a.cf_lambda, wd, a.seed) + (f"_{a.tag}" if a.tag else "")
    dest = out / f"{name}.json"
    if dest.exists() and not a.overwrite:                  # a result is a measurement
        raise SystemExit(f"{dest} exists; pass --tag or --overwrite")

    from transformers import AutoConfig, GPTNeoXForCausalLM
    n_layers = AutoConfig.from_pretrained(c["model"]).num_hidden_layers
    lo, hi = stage_bounds(n_layers, c["n_stages"])[k]
    teacher = StageRunner(c["model"], lo, hi, dt, "sdpa", dev)
    full = GPTNeoXForCausalLM.from_pretrained(c["model"], dtype=dt, attn_implementation="sdpa").to(dev).eval()

    st_in, st_out = np.load(harvest / f"stats_iface{k}.npz"), np.load(harvest / f"stats_iface{k + 1}.npz")
    assert int(st_in["mc_n"]) == budget["train_tokens"], "statistics are not from the training rows alone"
    phi_in = build_contract(st_in, "zca", None).to(dev)
    phi_out = build_contract(st_out, "zca", None).to(dev)
    anchors = load_refs(harvest / "anchors" / f"iface{k}")
    val_X = load_refs(harvest / "anchors_val" / f"iface{k}")
    assert anchors.shape[0] == budget["train_rows"] and val_X.shape[0] == budget["val_rows"]
    L = c["train"]["seq_len"]
    # validation reads at most val_seqs_max rows: 32 x 2048 tokens is ample for a paired
    # comparison of checkpoints and keeps a validation under a tenth of the step time
    n_val = min(budget["val_rows"], c.get("val_seqs_max", budget["val_rows"]))
    val_ids = torch.from_numpy(np.load(harvest / "val_rows.npy")[:n_val, : L + 1].astype(np.int64))
    held_ids = torch.from_numpy(np.load(c["heldout"])[: c["heldout_seqs"], : L + 1].astype(np.int64))

    noise = ContractGaussianized(phi_in, "iid", None) if a.m > 0 else None
    sampler = DoseMix(anchors, noise, a.m)
    d = int(st_in["mc_mean"].shape[0])
    student = StudentStage(student_config(c["model"], d, c["student_layers"], c["student_heads"]), seed=a.seed)
    cfg = SelectConfig(**{**c["train"], "weight_decay": wd, "cf_lambda": a.cf_lambda, "seed": a.seed})
    cf = CFDistance(d, M=c["cf"]["M"], freqs=tuple(c["cf"]["freqs"]), seed=c["cf"]["seed"]).to(dev) \
        if a.cf_lambda > 0 else None

    vb = min(2, cfg.batch)
    base_val = next_token_loss(full, val_ids, vb)           # the teacher's own loss, once

    def stitched_loss(stu, ids):
        with Stitched(full, lo, hi, stu, phi_in, phi_out):
            return next_token_loss(full, ids, vb)

    def val_fn(stu):
        return stitched_loss(stu, val_ids) - base_val

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not sha and Path("SHA").exists():       # a pod gets an rsync of the tree, not the repo
        sha = Path("SHA").read_text().strip()
    logf = open(out / f"{name}.log", "w")

    def log(r):
        print(r, flush=True); logf.write(json.dumps(r) + "\n"); logf.flush()

    log({"cell": name, "m": a.m, "cf_lambda": a.cf_lambda, "weight_decay": wd, "lr": cfg.lr,
         "cap_steps": cfg.cap_steps, "patience": cfg.patience, "budget": budget, "sha": sha})
    t0 = time.time()
    hist, summary = train_selected(student, teacher, sampler, cfg, val_fn, phi_in, phi_out, cf=cf,
                                   device=dev, log=log)

    # reported numbers: nothing below was used to stop or to select
    with torch.no_grad():
        student.eval()
        base_held = next_token_loss(full, held_ids, vb)
        held = stitched_loss(student, held_ids) - base_held
        val_eps = evaluate(student, teacher, val_X, phi_in, phi_out, vb, dev)
        x = val_X[:vb, :L].to(dev)
        t = _apply(phi_out, teacher(x.to(dt)).float())
        p = student(_apply(phi_in, x.float())).float()
        probe = CFDistance(d, M=64, seed=0).to(dev)
        ps, pt = p.reshape(-1, d) @ probe.U.T, t.reshape(-1, d) @ probe.U.T
        dispersion = float((ps.var(0) / pt.var(0)).mean())
        cf_val = float(probe(p, t))

    p_t = sum(q.numel() for q in teacher.parameters())
    p_s = sum(q.numel() for q in student.parameters())
    p_full = sum(q.numel() for q in full.parameters())
    flops_train = summary["positions"] * (2 * p_t + 6 * p_s)
    flops_val = summary["validations"] * val_ids[:, :-1].numel() * 2 * p_full
    rec = {"cell": name, "stage": k, "m": a.m, "cf_lambda": a.cf_lambda, "weight_decay": wd,
           "seed": a.seed, "budget": budget, "select": summary,
           "held_stitch_delta": held, "held_teacher_loss": base_held,
           "val_stitch_delta": summary["selected_val"], "val_eps": val_eps,
           "dispersion_student_over_teacher": dispersion, "cf_distance_val": cf_val,
           "noise_per_distinct_real_position": summary["noise_positions"] / max(1, budget["train_tokens"]),
           "real_passes": summary["real_positions"] / max(1, budget["train_tokens"]),
           "flops": {"train": flops_train, "validation": flops_val, "total": flops_train + flops_val,
                     "teacher_stage_params": p_t, "student_params": p_s},
           "schedule": {"lr": cfg.lr, "warmup": cfg.warmup, "val_every": cfg.val_every,
                        "patience": cfg.patience, "cooldown_frac": cfg.cooldown_frac, "cap_steps": cfg.cap_steps},
           "config": c, "history": hist, "seconds": time.time() - t0,
           "env": {"sha": sha, "torch": torch.__version__, "python": platform.python_version(),
                   "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   "harvest_sha": hmeta["env"]["sha"], "sha256_train": budget["sha256_train"]}}
    json.dump(rec, open(dest, "w"))
    torch.save(student.state_dict(), out / f"{name}.pt")
    print(json.dumps({q: v for q, v in rec.items() if q not in ("history", "config")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--m", type=float, required=True, help="noise positions per real position; 0 is the control")
    p.add_argument("--cf-lambda", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=None, help="weight decay override")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--harvest", default="")
    p.add_argument("--out", default="")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
