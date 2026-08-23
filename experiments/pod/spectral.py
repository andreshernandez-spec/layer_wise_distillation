"""Spectral metrics for every saved student on the pod, into one small JSON.

    python experiments/pod/spectral.py out/phase1-1.4b-a100
"""
import glob
import json
import os
import sys

import torch

from lwd.eval.spectral import metrics_for


def main(d):
    # resume: the pass is ~30 s a checkpoint, so losing it to a crash at the end is
    # the expensive failure. Write after every cell and skip what is already there.
    path = f"{d}/spectral.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for f in sorted(glob.glob(f"{d}/*.pt")):
        cell = os.path.basename(f)[:-3]
        if cell in out:
            continue
        m = metrics_for(torch.load(f, map_location="cpu"))
        al = [v["alpha"] for v in m.values() if "alpha" in v]
        sr = [v["stable_rank"] for v in m.values() if "stable_rank" in v]
        if not al:
            out[cell] = {"per_matrix": m, "non_finite": True}
            print(cell, "non-finite weights, skipped", flush=True)
            json.dump(out, open(path, "w"), indent=1)
            continue
        out[cell] = {"per_matrix": m, "alpha_mean": sum(al) / len(al), "alpha_min": min(al),
                     "alpha_max": max(al), "stable_rank_mean": sum(sr) / len(sr)}
        print(cell, round(out[cell]["alpha_mean"], 2), round(out[cell]["stable_rank_mean"], 1), flush=True)
        json.dump(out, open(path, "w"), indent=1)
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
