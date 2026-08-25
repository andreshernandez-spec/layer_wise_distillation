"""Spectral metrics for every saved student on the pod, into one small JSON.

    python experiments/pod/spectral.py out/phase1-1.4b-a100
"""
import glob
import json
import os
import sys

import torch

from lwd.eval.spectral import metrics_for


def write(obj, path):
    """Atomic: a reader (or a resume) never sees a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def main(d):
    # resume: the pass is ~30 s a checkpoint, so losing it to a crash at the end is
    # the expensive failure. Write after every cell and skip what is already there.
    path = f"{d}/spectral.json"
    try:
        out = json.load(open(path)) if os.path.exists(path) else {}
    except json.JSONDecodeError:
        # a non-atomic write interrupted mid-flush leaves a truncated file, and then
        # every resume dies on it. Start over rather than refuse to run.
        print("existing spectral.json is truncated, starting fresh", flush=True)
        out = {}
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
            write(out, path)
            continue
        out[cell] = {"per_matrix": m, "alpha_mean": sum(al) / len(al), "alpha_min": min(al),
                     "alpha_max": max(al), "stable_rank_mean": sum(sr) / len(sr)}
        print(cell, round(out[cell]["alpha_mean"], 2), round(out[cell]["stable_rank_mean"], 1), flush=True)
        write(out, path)
    write(out, path)


if __name__ == "__main__":
    main(sys.argv[1])
