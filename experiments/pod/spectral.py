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
    out = {}
    for f in sorted(glob.glob(f"{d}/*.pt")):
        cell = os.path.basename(f)[:-3]
        m = metrics_for(torch.load(f, map_location="cpu"))
        al = [v["alpha"] for v in m.values()]; sr = [v["stable_rank"] for v in m.values()]
        out[cell] = {"per_matrix": m, "alpha_mean": sum(al) / len(al), "alpha_min": min(al),
                     "alpha_max": max(al), "stable_rank_mean": sum(sr) / len(sr)}
        print(cell, round(out[cell]["alpha_mean"], 2), round(out[cell]["stable_rank_mean"], 1), flush=True)
    json.dump(out, open(f"{d}/spectral.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
