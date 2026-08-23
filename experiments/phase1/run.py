"""Phase 1 cell runner (docs/02): one (measure, structure, Q) cell of Exp 1.

    python experiments/phase1/run.py experiments/phase1/configs/cell.yaml \
        --measure C --structure ar1 --q 1e6 --seed 0

Reads the Phase 0 harvest (stats + fp16 anchor refs) for the teacher stage's input
interface, builds the contract, the sampler and the student, trains for Q positions,
evaluates on held-out real activations, and writes one JSON per cell.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lwd.contract.whiten import Affine, Contract, MarginalGaussianize
from lwd.eval.stitch import stitching_delta
from lwd.eval.diagnostics import attention_entropy_under, jacobian_agreement
from lwd.harvest.model import Lower, StageRunner, stage_bounds
from lwd.noise.samplers import AnchorMix, ContractGaussianized, Gaussian, Isotropic, LiveReal
from lwd.stage.student import StudentStage, student_config
from lwd.stage.train import TrainConfig, evaluate, train_stage

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_refs(d: Path, max_seqs: int | None = None) -> torch.Tensor:
    x = torch.cat([torch.from_numpy(np.load(f)) for f in sorted(d.glob("ref_*.npy"))])
    return x[:max_seqs] if max_seqs else x


def build_contract(st, kind: str, d_s: int | None):
    mean, cov = torch.from_numpy(st["mc_mean"]), torch.from_numpy(st["mc_cov_shrunk"])
    if kind == "zca":
        return Contract(None, Affine.zca(mean, cov))
    if kind == "pca":
        return Contract(None, Affine.pca(mean, cov, d_s))
    if kind == "gauss":
        m = MarginalGaussianize(torch.from_numpy(st["q_q"]), torch.from_numpy(st["q_values"]))
        # whiten the Gaussianized data: its covariance is unknown without a second pass,
        # so estimate it from the anchor refs (done by the caller via fit_gauss_affine)
        return Contract(m, None)
    raise ValueError(kind)


def fit_gauss_affine(contract: Contract, refs: torch.Tensor) -> Contract:
    from lwd.harvest.stats import MeanCov
    y = contract.gauss.forward(refs.reshape(-1, refs.shape[-1]).double())
    mc = MeanCov(y.shape[1]); mc.update(y); st = mc.finalize()
    return Contract(contract.gauss, Affine.zca(st["mean"], st["cov_shrunk"]))


class RealSampler:
    def __init__(self, X):
        self.X = X

    def sample(self, b, L, g, device):
        idx = torch.randint(0, self.X.shape[0], (b,), generator=g)
        return self.X[idx, :L].to(device)


def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"]); out.mkdir(parents=True, exist_ok=True)
    name = f"{a.measure}_{a.structure}_q{a.q:.0e}_s{a.seed}".replace("+", "") + (f"_{a.tag}" if a.tag else "")
    dev, dt = c["device"], DT[c["dtype"]]
    harvest = Path(c["harvest"])
    k = c["stage"]
    n_layers = __import__("transformers").AutoConfig.from_pretrained(c["model"]).num_hidden_layers
    lo, hi = stage_bounds(n_layers, c["n_stages"])[k]
    teacher = StageRunner(c["model"], lo, hi, dt, "sdpa", dev)
    st_in = np.load(harvest / f"stats_iface{k}.npz")
    st_out = np.load(harvest / f"stats_iface{k + 1}.npz")
    refs_in = load_refs(harvest / "anchors" / f"iface{k}")
    n_eval = c["eval_seqs"]
    eval_X, anchor_X = refs_in[:n_eval], refs_in[n_eval:]
    if c.get("anchor_seqs_used"):  # scarce-anchor regime: use only the first k anchor sequences
        anchor_X = anchor_X[: c["anchor_seqs_used"]]
    rho = torch.from_numpy(st_in["lag1_rho"]).float()

    phi_in = build_contract(st_in, c["contract"], c.get("d_s"))
    # bridge cell: a narrower student reads and writes the top-d_s PCA coordinates of
    # both interfaces; eps is then also reported lifted back to the full target
    phi_out = build_contract(st_out, "pca" if c.get("d_s") else "zca", c.get("d_s"))
    phi_out_full = build_contract(st_out, "zca", None).to(dev)
    if phi_in.gauss is not None:
        phi_in = fit_gauss_affine(phi_in, anchor_X)
    phi_in, phi_out = phi_in.to(dev), phi_out.to(dev)

    mean, cov = torch.from_numpy(st_in["mc_mean"]), torch.from_numpy(st_in["mc_cov_shrunk"])
    noise = {"G": lambda: Gaussian(mean, cov, a.structure, rho),
             "I": lambda: Isotropic(mean, cov, a.structure, rho),
             "C": lambda: ContractGaussianized(phi_in, a.structure, rho)}
    if a.measure == "R":
        sampler = RealSampler(anchor_X)
    elif a.measure == "L":
        # live real activations from the slice rows, excluding this interface's anchors
        # (the eval set is drawn from them) and the stitching rows
        rows_all = np.load(c["slice"])
        if c.get("decontam"):
            rows_all = np.delete(rows_all, json.load(open(c["decontam"]))["drop"], axis=0)
        used = set(np.load(harvest / "anchor_idx.npy")[k].tolist())
        free_rows = [i for i in range(len(rows_all)) if i not in used]
        sampler = LiveReal(rows_all, Lower(c["model"], lo, dt, "sdpa", dev), exclude=used | set(free_rows[: c["stitch_seqs"]]))
    elif a.structure == "mix":
        # inside a mix the noise component is always i.i.d.: the mixing is the
        # structure. Passing a.structure through here built Gaussian(..., "mix"),
        # which the sampler does not know (found on the pod, 23 Aug 2026).
        inner = {"G": lambda: Gaussian(mean, cov, "iid", rho),
                 "I": lambda: Isotropic(mean, cov, "iid", rho),
                 "C": lambda: ContractGaussianized(phi_in, "iid", rho)}[a.measure]()
        sampler = AnchorMix(anchor_X, inner, c["mix_real_frac"])
    else:
        sampler = noise[a.measure]()

    scfg = student_config(c["model"], c.get("d_s") or st_in["mc_mean"].shape[0],
                          c["student_layers"], c["student_heads"])
    student = StudentStage(scfg, seed=a.seed)
    tc = TrainConfig(**c["train"], seed=a.seed)
    tc.steps = int(round(a.q / (tc.batch * tc.seq_len)))
    rec = {"cell": name, "measure": a.measure, "structure": a.structure, "q": a.q, "seed": a.seed,
           "steps": tc.steps, "student_params": student.n_params(), "config": c,
           "env": {"sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                   "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   "python": platform.python_version()}}
    logf = open(out / f"{name}.log", "w")
    t0 = time.time()
    hist = train_stage(student, teacher, sampler, tc, phi_in, phi_out, eval_X=eval_X, device=dev,
                       log=lambda r: (print(r, flush=True), logf.write(json.dumps(r) + "\n"), logf.flush()))
    rec["final_eval"] = hist[-1]["eval"]
    if c.get("d_s"):
        # full-space eps: lift the student's d_s output by the PCA pseudo-inverse and
        # compare with the full teacher output in full ZCA coordinates
        class Lifted(torch.nn.Module):
            def __init__(s, st): super().__init__(); s.st = st
            def forward(s, z):
                y = s.st(z); sh = y.shape
                return phi_out_full.forward(phi_out.inverse(y.reshape(-1, sh[-1])).float()).reshape(*sh[:-1], -1)
        rec["final_eval_projected"] = rec["final_eval"]
        rec["final_eval"] = evaluate(Lifted(student), teacher, eval_X, phi_in, phi_out_full, tc.batch, dev)
        print("eval projected", rec["final_eval_projected"], "full", rec["final_eval"], flush=True)
    # diagnostics (docs/02 measurements 3 and 4)
    with torch.no_grad():
        torch.cuda.empty_cache()
        # fp32 eager: fp16 eager attention overflows on massive activations (NaN from
        # block 13 of the 1.4B); batch 2 keeps the (b, h, L, L) map small
        eager = StageRunner(c["model"], lo, hi, torch.float32, "eager", dev)
        gd = torch.Generator(device="cpu").manual_seed(1234)
        nb = min(2, tc.batch)
        xb = eval_X[:nb].to(dev)
        sb = sampler.sample(max(nb, 2 if a.structure != "mix" else tc.batch), tc.seq_len, gd, dev)
        sb = (sb[0] if isinstance(sb, tuple) else sb)[:nb]
        rec["attn_entropy"] = {"real": attention_entropy_under(eager, xb.float()).mean(1).tolist(),
                               "arm": attention_entropy_under(eager, sb.float()).mean(1).tolist(),
                               "uniform": float(np.log(tc.seq_len))}
    # Jacobian agreement through the fp32 runner as well: forward-mode AD through the
    # fp16 teacher fails (float32 dual tensors against half weights)
    rec["jacobian"] = jacobian_agreement(student, eager, phi_in, phi_out,
                                         eval_X[:2, : min(256, tc.seq_len)].to(dev), n_dirs=c.get("jvp_dirs", 8))
    del eager
    torch.cuda.empty_cache()
    print("attn", rec["attn_entropy"], "jac", rec["jacobian"], flush=True)
    # stitching delta on slice rows that were not anchors of this interface
    from transformers import GPTNeoXForCausalLM
    rows = np.load(c["slice"])
    if c.get("decontam"):
        rows = np.delete(rows, json.load(open(c["decontam"]))["drop"], axis=0)
    used = set(np.load(harvest / "anchor_idx.npy")[k].tolist())
    free = [i for i in range(len(rows)) if i not in used][: c["stitch_seqs"]]
    ids = torch.from_numpy(rows[free, : c["train"]["seq_len"] + 1].astype(np.int64))
    del teacher; torch.cuda.empty_cache()
    full = GPTNeoXForCausalLM.from_pretrained(c["model"], dtype=dt, attn_implementation="sdpa").to(dev).eval()
    rec["stitch"] = stitching_delta(full, lo, hi, student, phi_in, phi_out, ids, batch=min(2, c["train"]["batch"]))
    del full; torch.cuda.empty_cache()
    print("stitch", rec["stitch"], flush=True)
    rec["history"] = hist
    rec["seconds"] = time.time() - t0
    json.dump(rec, open(out / f"{name}.json", "w"))
    torch.save(student.state_dict(), out / f"{name}.pt")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("history", "config")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--measure", choices=["R", "L", "G", "I", "C"], required=True)
    p.add_argument("--structure", choices=["iid", "ar1", "mix"], default="iid")
    p.add_argument("--q", type=float, required=True, help="noise positions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the cell name (e.g. a46 for 46 anchors)")
    main(p.parse_args())
