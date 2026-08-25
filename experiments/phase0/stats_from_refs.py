"""Interim interface statistics from the fp16 anchor refs (1M positions per interface)
instead of the full slice. Used when the full-pass statistics are not available yet;
labelled interim in the output directory and never cited as the C0.2 numbers.

    python experiments/phase0/stats_from_refs.py out/harvest-1.4b out/harvest-1.4b-interim [harvest.yaml]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from lwd.harvest.stats import CFSketch, Lag1, MeanCov, Quantiles, to_numpy


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "run.json").exists():
        cfg = json.load(open(src / "run.json"))["config"]
    else:  # the source harvest crashed before writing run.json: take the config file
        import yaml
        cfg = yaml.safe_load(open(sys.argv[3]))
    for link in ("anchors", "topk", "anchor_idx.npy"):
        if not (dst / link).exists() and (src / link).exists():
            os.symlink(os.path.relpath(src / link, dst), dst / link)
    S = len(list(src.glob("anchors/iface*")))
    for k in range(S):
        files = sorted((src / "anchors" / f"iface{k}").glob("ref_*.npy"))
        d = np.load(files[0]).shape[-1]
        acc = {"mc": MeanCov(d), "cf": CFSketch(d, seed=cfg.get("seed", 0)), "q": Quantiles(d, seed=cfg.get("seed", 0)), "lag1": Lag1(d)}
        n = 0
        for f in files:
            h = torch.from_numpy(np.load(f))
            for a in acc.values():
                a.update(h)
            n += h.shape[0] * h.shape[1]
        np.savez(dst / f"stats_iface{k}.npz", **{f"{g}_{nm}": v for g, a in acc.items() for nm, v in to_numpy(a.finalize()).items()})
        print(f"iface{k}: {n} positions from {len(files)} files", flush=True)
    json.dump({"interim": True, "source": str(src), "config": cfg}, open(dst / "run.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
