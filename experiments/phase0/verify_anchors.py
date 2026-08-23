"""G0.3: pack the int8 anchor store from the fp16 refs and the harvested channel std,
then check that dequantized anchors pushed through teacher stage k reproduce the exact
stage output within a tolerance derived from the quantization floor.

Streams one ref file at a time. Peak host memory is one ref file (~40 MB) plus the
per-channel error accumulators; it never holds an interface in float32 (that held
~30 GB per interface and took the laptop down on 22 Aug 2026).

    python experiments/phase0/verify_anchors.py out/harvest-1.4b --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from lwd.harvest.anchors import dequantize, quantize
from lwd.harvest.model import StageRunner, stage_bounds

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class ChanErr:
    """Accumulates per-channel sum of squared error and per-channel variance of the
    reference, for channel_relative_error without holding the data."""

    def __init__(self, d):
        self.se = torch.zeros(d, dtype=torch.float64)
        self.s1 = torch.zeros(d, dtype=torch.float64)
        self.s2 = torch.zeros(d, dtype=torch.float64)
        self.n = 0

    def update(self, ref, other):
        r = ref.reshape(-1, ref.shape[-1]).to(torch.float64)
        o = other.reshape(-1, other.shape[-1]).to(torch.float64)
        self.se += ((r - o) ** 2).sum(0)
        self.s1 += r.sum(0); self.s2 += (r * r).sum(0); self.n += r.shape[0]

    def value(self):
        var = (self.s2 / self.n - (self.s1 / self.n) ** 2).clamp_min(1e-24)
        return float(((self.se / self.n) / var).mean().sqrt())


@torch.no_grad()
def main(out, device, batch, fmt):
    out = Path(out)
    cfg = json.load(open(out / "run.json"))["config"]
    S, dt = cfg["n_stages"], DT[cfg["dtype"]]
    n_layers = AutoConfig.from_pretrained(cfg["model"]).num_hidden_layers
    report = {}
    for k, (a, b) in enumerate(stage_bounds(n_layers, S)):
        st = np.load(out / f"stats_iface{k}.npz")
        chan_std = torch.from_numpy(np.sqrt(np.diag(st["mc_cov"]))).float()
        runner = StageRunner(cfg["model"], a, b, dt, cfg["attn"], device)
        files = sorted((out / "anchors" / f"iface{k}").glob("ref_*.npy"))
        d = chan_std.numel()
        floor, prop = ChanErr(d), ChanErr(d)
        n_seqs = 0
        for f in files:
            ref = torch.from_numpy(np.load(f))  # (b, L, d) fp16
            deq = dequantize(*quantize(ref, chan_std, fmt), chan_std)  # float32, one file
            floor.update(ref, deq)
            for i in range(0, ref.shape[0], batch):
                oe = runner(ref[i:i + batch].to(device).to(dt)).float().cpu()
                oq = runner(deq[i:i + batch].to(device).to(dt)).float().cpu()
                prop.update(oe, oq)
            n_seqs += ref.shape[0]
        fl, pr = floor.value(), prop.value()
        tol = 4.0 * fl
        report[f"stage{k}"] = {"quant_floor": fl, "propagated": pr, "amplification": pr / fl,
                               "tolerance": tol, "pass": bool(pr <= tol), "n_seqs": n_seqs, "fmt": fmt}
        print(json.dumps({f"stage{k}": report[f"stage{k}"]}), flush=True)
        del runner
        torch.cuda.empty_cache()
    json.dump(report, open(out / f"verify_anchors_{fmt}.json", "w"), indent=1)
    assert all(v["pass"] for v in report.values()), "G0.3 failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--device", default="cpu"); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--fmt", default="int8", choices=["int8", "fp8"])
    a = p.parse_args()
    main(a.out, a.device, a.batch, a.fmt)
