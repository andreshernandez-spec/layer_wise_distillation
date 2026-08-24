"""C2.2: drift through the composed student, against the teacher's own sensitivity.

Two quantities per stage, on held-out real text:

  realized drift   how far the student's interface k is from the teacher's, in
                   whitened coordinates (so channels are comparable)
  Lipschitz ratio  how much the *teacher's* stage k amplifies an input perturbation
                   of the size the student actually produces

Drift that merely tracks the product of the ratios is the teacher's sensitivity, not
the student's fault. `tests/test_compose.py` already showed the teacher amplifies
float-level perturbations 10-1000x per stage, so the floor is real and has to be
subtracted by eye before any mitigation is credited.

    python experiments/phase2/drift.py experiments/phase2/configs/stack-1.4b.yaml --measure R
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from lwd.compose.chain import Wrapped
from lwd.compose.model import StudentLM
from lwd.contract.whiten import Affine, Contract
from lwd.harvest.model import Edges, ResidentModel, StageRunner, stage_bounds
from lwd.stage.student import StudentStage, student_config

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def whitened_rel_mse(a, b, phi):
    """||phi(a) - phi(b)||^2 / var(phi(b)), both (n, L, d)."""
    x = phi.forward(a.reshape(-1, a.shape[-1]).to(phi.dtype)).float()
    y = phi.forward(b.reshape(-1, b.shape[-1]).to(phi.dtype)).float()
    return float(((x - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


@torch.no_grad()
def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"]); dev, dt = c["device"], DT[c["dtype"]]
    S, model_id = c["n_stages"], c["model"]
    struct = "mix" if a.measure == "C" else "iid"
    phis, cfgs, sds = [], [], []
    for k in range(S + 1):
        st = np.load(Path(c["harvest"]) / f"stats_iface{k}.npz")
        phis.append(Contract(None, Affine.zca(torch.from_numpy(st["mc_mean"]),
                                              torch.from_numpy(st["mc_cov_shrunk"]))).to(dev, torch.float32))
    for k in range(S):
        name = f"{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}_stage{k}".replace("+", "")
        sds.append(torch.load(out / f"{name}.pt", map_location=dev))
        cfgs.append(student_config(model_id, c.get("d_s") or phis[0].affine.mean.numel(),
                                   c["student_layers"], c["student_heads"]))
    stages = []
    for k in range(S):
        stu = StudentStage(cfgs[k], seed=a.seed); stu.load_state_dict(sds[k]); stu.to(dev).eval()
        stages.append(Wrapped(stu, phis[k], phis[k + 1]))
    student = StudentLM(Edges(model_id, dt, dev), stages).to(dev).eval()

    rows = np.load(c["slice"])
    if c.get("decontam"):
        rows = np.delete(rows, json.load(open(c["decontam"]))["drop"], axis=0)
    used = set(np.load(Path(c["harvest"]) / "anchor_idx.npy").ravel().tolist())
    free = [i for i in range(len(rows)) if i not in used][: a.rows]
    ids = torch.from_numpy(rows[free, : c["train"]["seq_len"]].astype(np.int64))

    teacher = ResidentModel(model_id, S, dt, "sdpa", dev)
    rep = {"realized_drift": [], "lipschitz": [], "stage_eps": []}
    for i in range(0, ids.shape[0], a.batch):
        t_if, _ = teacher.forward(ids[i:i + a.batch].to(dev))
        s_if = student.interfaces(ids[i:i + a.batch].to(dev))
        d = [whitened_rel_mse(s_if[k], t_if[k], phis[k]) for k in range(S + 1)]
        rep["realized_drift"].append(d)
    rep["realized_drift"] = np.mean(rep["realized_drift"], 0).tolist()
    del teacher
    torch.cuda.empty_cache() if dev == "cuda" else None

    # Lipschitz: perturb the teacher stage's input by the drift the student produces
    g = torch.Generator().manual_seed(0)
    bounds = stage_bounds(24 if "1.4b" in model_id else 6, S)
    for k, (lo, hi) in enumerate(bounds):
        run = StageRunner(model_id, lo, hi, dt, "sdpa", dev)
        x = torch.from_numpy(np.load(sorted((Path(c["harvest"]) / "anchors" / f"iface{k}").glob("ref_*.npy"))[0]))[:2].to(dev)
        scale = max(rep["realized_drift"][k], 1e-8) ** 0.5
        noise = torch.randn(x.shape, generator=g).to(dev) * x.float().std() * scale
        y0 = run(x.to(dt)).float(); y1 = run((x.float() + noise).to(dt)).float()
        din = whitened_rel_mse(x.float() + noise, x.float(), phis[k])
        dout = whitened_rel_mse(y1, y0, phis[k + 1])
        rep["lipschitz"].append(dout / max(din, 1e-12))
        del run
    prod = np.cumprod([1.0] + rep["lipschitz"]).tolist()
    rep["predicted_from_stage0_drift"] = [rep["realized_drift"][1] * p for p in prod]
    json.dump(rep, open(out / f"drift_{a.measure}.json", "w"), indent=1)
    print(json.dumps({k: [round(x, 5) for x in v] if isinstance(v, list) else v for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config"); p.add_argument("--measure", default="R")
    p.add_argument("--q", type=float, default=1e8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows", type=int, default=16); p.add_argument("--batch", type=int, default=2)
    main(p.parse_args())
