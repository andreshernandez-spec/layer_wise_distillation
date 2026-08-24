"""C2.3: retrain a stage on-policy and measure what it buys.

The C2.2 decomposition (docs/03) says each stage adds ~0.54 of fresh error regardless
of depth. Part of that is exposure bias: the stage was trained on the teacher's clean
interface and meets a drifted one. This retrains stage k on the propagated interface
and reports the change in the two numbers that matter, so the prediction is testable:

  large drop  -> exposure bias dominates the fresh term, DAgger is the mitigation
  small drop  -> the fresh term is irreducible stage error, and better stages, not
                 better propagation, is the route to a better composed model

    python experiments/phase2/dagger.py CONFIG --stage 3 --on-policy 0.5
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from lwd.compose.chain import Wrapped
from lwd.compose.dagger import PropagatedSampler
from lwd.contract.whiten import Affine, Contract
from lwd.harvest.model import Lower, StageRunner, stage_bounds
from lwd.stage.student import StudentStage, student_config
from lwd.stage.train import TrainConfig, evaluate, train_stage

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"]); dev, dt = c["device"], DT[c["dtype"]]
    S, model_id, k = c["n_stages"], c["model"], a.stage
    struct = "mix" if a.measure == "C" else "iid"
    harvest = Path(c["harvest"])
    phis = []
    for j in range(S + 1):
        st = np.load(harvest / f"stats_iface{j}.npz")
        phis.append(Contract(None, Affine.zca(torch.from_numpy(st["mc_mean"]),
                                              torch.from_numpy(st["mc_cov_shrunk"]))).to(dev, torch.float32))
    d_s = c.get("d_s") or phis[0].affine.mean.numel()
    cfg = student_config(model_id, d_s, c["student_layers"], c["student_heads"])
    n_layers = __import__("transformers").AutoConfig.from_pretrained(model_id).num_hidden_layers
    lo, hi = stage_bounds(n_layers, S)[k]

    below = []
    for j in range(k):
        stu = StudentStage(cfg, seed=a.seed)
        stu.load_state_dict(torch.load(out / f"{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}_stage{j}.pt".replace("+", ""), map_location=dev))
        below.append(Wrapped(stu.to(dev).eval(), phis[j], phis[j + 1]))
    rows = np.load(c["slice"])
    if c.get("decontam"):
        rows = np.delete(rows, json.load(open(c["decontam"]))["drop"], axis=0)
    used = set(np.load(harvest / "anchor_idx.npy")[k].tolist())
    train_rows = rows[[i for i in range(len(rows)) if i in used]]

    teacher = StageRunner(model_id, lo, hi, dt, "sdpa", dev)
    sampler = PropagatedSampler(train_rows, Lower(model_id, 0, dt, "sdpa", dev), below,
                                Lower(model_id, lo, dt, "sdpa", dev), a.on_policy)
    eval_X = torch.cat([torch.from_numpy(np.load(f)) for f in
                        sorted((harvest / "anchors" / f"iface{k}").glob("ref_*.npy"))])[: c["eval_seqs"]]
    student = StudentStage(cfg, seed=a.seed)
    student.load_state_dict(torch.load(out / f"{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}_stage{k}.pt".replace("+", ""), map_location=dev))
    student.to(dev)
    tc = TrainConfig(**c["train"], seed=a.seed)
    tc.steps = int(round(a.q_retrain / (tc.batch * tc.seq_len)))
    tc.lr = c["train"]["lr"] * a.lr_scale
    # the drifted eval set: held-out rows pushed through the stages below, which is
    # the interface this stage actually meets in the composed model. DAgger targets
    # error HERE; clean eps is the control that says what it cost.
    free = [i for i in range(len(rows)) if i not in used][: c["eval_seqs"]]
    with torch.no_grad():
        h = Lower(model_id, 0, dt, "sdpa", dev)(torch.from_numpy(rows[free, : tc.seq_len].astype(np.int64)).to(dev))
        for st_below in below:
            h = st_below(h)
        drift_X = h.float().cpu()
    before = evaluate(student, teacher, eval_X, phis[k], phis[k + 1], tc.batch, dev)
    before_d = evaluate(student, teacher, drift_X, phis[k], phis[k + 1], tc.batch, dev)
    hist = train_stage(student, teacher, sampler, tc, phis[k], phis[k + 1], eval_X=eval_X,
                       device=dev, log=lambda r: print(r, flush=True))
    after_d = evaluate(student, teacher, drift_X, phis[k], phis[k + 1], tc.batch, dev)
    rec = {"stage": k, "on_policy": a.on_policy, "q_retrain": a.q_retrain,
           "eps_drifted_before": before_d, "eps_drifted_after": after_d,
           "eps_clean_before": before, "eps_clean_after": hist[-1]["eval"],
           "sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    name = f"dagger_{a.measure}_stage{k}_p{a.on_policy}_q{a.q_retrain:.0e}".replace("+", "")
    torch.save(student.state_dict(), out / f"{name}.pt")
    json.dump({**rec, "history": hist}, open(out / f"{name}.json", "w"))
    print(json.dumps(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config"); p.add_argument("--stage", type=int, required=True)
    p.add_argument("--measure", default="R"); p.add_argument("--q", type=float, default=1e8)
    p.add_argument("--q-retrain", type=float, default=1e7); p.add_argument("--on-policy", type=float, default=0.5)
    p.add_argument("--lr-scale", type=float, default=0.3); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
