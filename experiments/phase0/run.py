"""Phase 0 harvest driver (docs/01 C0.2-C0.4).

    python experiments/phase0/run.py experiments/phase0/configs/harvest-1.4b.yaml

Resident mode: one forward per batch, all interfaces captured by hooks, statistics and
anchors updated in the same pass, top-k logits written. Streaming mode: stage by stage,
interface tensors round-trip through disk in fp16, one interface resident at a time.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lwd.harvest.logits import TopKWriter
from lwd.harvest.model import Edges, ResidentModel, StageRunner, stage_bounds
from lwd.harvest.stats import CFSketch, Lag1, MeanCov, Quantiles, to_numpy

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_config(p):
    c = yaml.safe_load(open(p))
    ints = ["n_stages", "batch", "seq_len", "anchor_seqs", "topk", "seed", "max_rows"]
    for k in ints:
        if k in c:
            assert isinstance(c[k], int), (k, c[k])
    assert c["dtype"] in DT and c["mode"] in ("resident", "streaming")
    return c


def env_record():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    import transformers
    return {"sha": sha, "dirty": bool(dirty.strip()), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


class IfaceStats:
    def __init__(self, d, cfg, device):
        self.mc, self.cf = MeanCov(d), CFSketch(d, seed=cfg["seed"])
        self.q, self.lag = Quantiles(d, seed=cfg["seed"]), Lag1(d)
        for acc in (self.mc, self.cf, self.lag):
            for name, t in vars(acc).items():
                if torch.is_tensor(t):
                    setattr(acc, name, t.to(device))

    def update(self, h):
        self.mc.update(h); self.cf.update(h); self.q.update(h.cpu()); self.lag.update(h)

    def finalize(self):
        return {"mc": to_numpy(self.mc.finalize()), "cf": to_numpy(self.cf.finalize()),
                "q": to_numpy(self.q.finalize()), "lag1": to_numpy(self.lag.finalize())}


def batches(rows, cfg):
    L = cfg["seq_len"]
    n = min(len(rows), cfg.get("max_rows", len(rows)))
    for i in range(0, n, cfg["batch"]):
        yield i, torch.from_numpy(rows[i:i + cfg["batch"], :L].astype(np.int64))


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    rows = np.load(cfg["slice"])
    if cfg.get("decontam"):
        drop = json.load(open(cfg["decontam"]))["drop"]
        rows = np.delete(rows, drop, axis=0)
        print(f"dropped {len(drop)} contaminated rows, {len(rows)} left", flush=True)
    rng = np.random.default_rng(cfg["seed"])
    n_rows = min(len(rows), cfg.get("max_rows", len(rows)))
    S = cfg["n_stages"]
    # each interface gets its own seeded subset of sequences as anchors
    anchor_idx = [np.sort(rng.choice(n_rows, cfg["anchor_seqs"], replace=False)) for _ in range(S + 1)]
    anchor_sets = [set(a.tolist()) for a in anchor_idx]
    np.save(out / "anchor_idx.npy", np.stack(anchor_idx))
    dev, dt = cfg["device"], DT[cfg["dtype"]]
    t0 = time.time()
    meta = {"config": cfg, "env": env_record(), "slice_sha256":
            hashlib.sha256(rows.tobytes()).hexdigest(), "n_rows": n_rows}

    if cfg["mode"] == "resident":
        m = ResidentModel(cfg["model"], S, dt, cfg["attn"], dev)
        d = m.model.config.hidden_size
        stats = [IfaceStats(d, cfg, dev) for _ in range(S + 1)]
        for k in range(S + 1):
            (out / "anchors" / f"iface{k}").mkdir(parents=True, exist_ok=True)
        stats_only = cfg.get("stats_only", False)  # refs and top-k already on disk
        # ref_ifaces limits which interfaces get anchor refs (4.1 GB each at d=2048);
        # the statistics come from the same pass for every interface regardless
        ref_ifaces = set(cfg.get("ref_ifaces") or range(S + 1))
        topk = None if (stats_only or cfg.get("skip_topk")) else TopKWriter(out / "topk", cfg["topk"])
        for i, ids in batches(rows, cfg):
            ifaces, logits = m.forward(ids.to(dev))
            for k, h in enumerate(ifaces):
                stats[k].update(h)
                if stats_only:
                    continue
                sel = [j for j in range(h.shape[0]) if i + j in anchor_sets[k]] if k in ref_ifaces else []
                if sel:
                    np.save(out / "anchors" / f"iface{k}" / f"ref_{i:06d}.npy",
                            h[sel].to(torch.float16).cpu().numpy())
            if topk is not None:
                topk.write(logits, ids)
            if i == 0:
                for st in stats:  # a finalize bug must fail here, not after the whole pass
                    st.finalize()
            if (i // cfg["batch"]) % 20 == 0:
                print(f"row {i}/{n_rows} {time.time() - t0:.0f}s", flush=True)
        for k in range(S + 1):
            np.savez(out / f"stats_iface{k}.npz", **{f"{g}_{n}": v for g, dd in
                                                    stats[k].finalize().items() for n, v in dd.items()})
            print(f"stats_iface{k} written", flush=True)
    else:
        from transformers import AutoConfig
        edges = Edges(cfg["model"], dt, dev)
        bounds = stage_bounds(AutoConfig.from_pretrained(cfg["model"]).num_hidden_layers, S)
        ifdir = out / "ifaces"; ifdir.mkdir(exist_ok=True)
        # interface 0
        for i, ids in batches(rows, cfg):
            np.save(ifdir / f"iface0_{i:06d}.npy", edges.embed(ids.to(dev)).cpu().numpy())
        for k, (a, b) in enumerate(bounds):
            runner = StageRunner(cfg["model"], a, b, dt, cfg["attn"], dev)
            for i, ids in batches(rows, cfg):
                h = torch.from_numpy(np.load(ifdir / f"iface{k}_{i:06d}.npy")).to(dev)
                with torch.no_grad():
                    np.save(ifdir / f"iface{k + 1}_{i:06d}.npy", runner(h).cpu().numpy())
            del runner
            if not cfg.get("keep_ifaces", False) and k > 0:
                for f in ifdir.glob(f"iface{k}_*.npy"):
                    f.unlink()
            print(f"stage {k} done {time.time() - t0:.0f}s", flush=True)
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(out / "run.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}))


if __name__ == "__main__":
    main(sys.argv[1])
