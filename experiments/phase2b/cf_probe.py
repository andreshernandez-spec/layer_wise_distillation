"""How large is the CF distance on students that already exist, and are they
under-dispersed? Run once, before the CF arm's lambda grid is pre-registered (docs/08),
on artifacts from Phase 2: nothing here trains anything.

    python experiments/phase2b/cf_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from run import build_contract, load_refs                      # noqa: E402

from lwd.harvest.model import StageRunner, stage_bounds
from lwd.noise.samplers import ContractGaussianized
from lwd.stage.cf import CFDistance
from lwd.stage.student import StudentStage, rel_mse, student_config

MODEL, K, DEV = "EleutherAI/pythia-1.4b", 2, "cuda"
H = Path("out/harvest-1.4b")
CK = Path("out/phase2-1.4b-a100/ckpt")


def ap(phi, x):
    sh = x.shape
    return phi.forward(x.reshape(-1, sh[-1]).to(phi.dtype)).reshape(*sh[:-1], -1).float()


@torch.no_grad()
def main():
    lo, hi = stage_bounds(24, 6)[K]
    teacher = StageRunner(MODEL, lo, hi, torch.float16, "sdpa", DEV)
    st_in, st_out = np.load(H / f"stats_iface{K}.npz"), np.load(H / f"stats_iface{K + 1}.npz")
    phi_in, phi_out = build_contract(st_in, "zca", None).to(DEV), build_contract(st_out, "zca", None).to(DEV)
    real = load_refs(H / "anchors" / f"iface{K}", 16)[:, :2048]
    g = torch.Generator().manual_seed(0)
    noise = ContractGaussianized(phi_in, "iid", None).sample(16, 2048, g, DEV).cpu()
    cf = CFDistance(2048, M=64, seed=0).to(DEV)
    out = {}
    for name, f in [("R (anchors only)", CK / "R_iid_q1e08_s0_stage2.pt"),
                    ("C_mix (noise recipe)", CK / "C_mix_q1e08_s0_stage2_preDAgger.pt")]:
        stu = StudentStage(student_config(MODEL, 2048, 2, 16), seed=0)
        stu.load_state_dict(torch.load(f, map_location="cpu")); stu.to(DEV).eval()
        for kind, X in (("real", real), ("noise", noise)):
            mse, cfd, ratio = [], [], []
            for i in range(0, X.shape[0], 2):
                x = X[i:i + 2].to(DEV)
                t = ap(phi_out, teacher(x.half()).float())
                p = stu(ap(phi_in, x.float())).float()
                mse.append(float(rel_mse(p, t))); cfd.append(float(cf(p, t)))
                ps, pt = p.reshape(-1, 2048) @ cf.U.T, t.reshape(-1, 2048) @ cf.U.T
                ratio.append(float((ps.var(0) / pt.var(0)).mean()))
            out[f"{name} | {kind}"] = {"rel_mse": np.mean(mse), "cf": np.mean(cfd),
                                       "var_ratio_student_over_teacher": np.mean(ratio)}
            r = out[f"{name} | {kind}"]
            print(f"{name:22s} {kind:6s} rel_mse {r['rel_mse']:.4f}  cf {r['cf']:.3e}  "
                  f"var(student)/var(teacher) {r['var_ratio_student_over_teacher']:.3f}  "
                  f"lambda for cf = 10% / 100% of mse: {0.1 * r['rel_mse'] / r['cf']:.0f} / {r['rel_mse'] / r['cf']:.0f}")
        del stu
    Path("out/phase2b").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("out/phase2b/cf_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
