"""C2.4: the heal-budget curve, from three starting points (docs/03).

  stagewise   the composed student from a trained stack
  random      the same architecture, untrained stages: the "better than nothing" bar
  oracle      the stack trained on real activations

The teacher's embedding, final norm and head transfer unchanged and are frozen: they
are not what stagewise training produced, and Phase 2's question is about the stages.
That also keeps AdamW state on a 16 GB card.

    python experiments/phase2/heal.py CONFIG --init stagewise --measure C --tokens 1e6
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lwd.compose.chain import Wrapped
from lwd.compose.model import StudentLM
from lwd.contract.whiten import Affine, Contract
from lwd.eval.stitch import next_token_loss
from lwd.harvest.model import Edges
from lwd.heal.train import HealConfig, TopKStore, heal
from lwd.stage.student import StudentStage, student_config

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build(c, a, dev, dt):
    S, model_id = c["n_stages"], c["model"]
    struct = "mix" if a.measure == "C" else "iid"
    loaded = {}
    phis = []
    for k in range(S + 1):
        st = np.load(Path(c["harvest"]) / f"stats_iface{k}.npz")
        phis.append(Contract(None, Affine.zca(torch.from_numpy(st["mc_mean"]),
                                              torch.from_numpy(st["mc_cov_shrunk"]))).to(dev, torch.float32))
    d_s = c.get("d_s") or phis[0].affine.mean.numel()
    stages = []
    for k in range(S):
        cfg = student_config(model_id, d_s, c["student_layers"], c["student_heads"])
        stu = StudentStage(cfg, seed=a.seed + (1000 if a.init == "random" else 0))
        if a.init != "random":
            name = f"{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}_stage{k}".replace("+", "")
            # Load a variant stack (e.g. _preDAgger) by name rather than by copying files
            # over the promoted ones: in-place promotion is what cost three results.
            ck = Path(c["out"]) / f"{name}{a.stack_suffix}.pt"
            stu.load_state_dict(torch.load(ck, map_location="cpu"))
            loaded[f"stage{k}"] = digest(ck)
        stages.append(Wrapped(stu.to(dev), phis[k], phis[k + 1]))
    edges = Edges(model_id, dt, dev)
    for p in edges.parameters():
        p.requires_grad_(False)          # the teacher's, and frozen
    return StudentLM(edges, stages).to(dev), loaded


def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"]); out.mkdir(parents=True, exist_ok=True)
    dev, dt = c["device"], DT[c["dtype"]]
    hs = a.heal_seed if a.heal_seed is not None else a.seed
    name = (f"heal_{a.init}{a.tag}_{a.measure}_t{a.tokens:.0e}_s{a.seed}"
            + (f"_h{hs}" if hs != a.seed else "")).replace("+", "")
    # The stack checkpoints are promoted in place (dagger_stack.py), so the same command
    # can load a different model on a later day. A result file is a measurement, not a
    # cache entry: never overwrite one silently.
    dest = out / f"{name}.json"
    if dest.exists() and not a.overwrite:
        raise SystemExit(f"{dest} exists; pass --tag or --overwrite")
    model, loaded = build(c, a, dev, dt)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = np.load(c["heldout"])
    ids = torch.from_numpy(rows[: a.eval_rows, : c["train"]["seq_len"] + 1].astype(np.int64))
    def ev(m):
        return next_token_loss(m, ids, batch=1)

    before = ev(model.eval())
    hist, lr, warm, seen = [], None, None, 0
    if a.tokens > 0:
        # A cold start needs a gentler schedule than a warm one. Using the same lr for
        # both does not compare stagewise against random init, it compares a tuned
        # learning rate against an untuned one.
        lr = a.lr if a.lr else c.get("heal_lr_random" if a.init == "random" else "heal_lr", 1e-4)
        warm = c.get("heal_warmup_random", 500) if a.init == "random" else c.get("heal_warmup", 500)
        globals()["_lr"], globals()["_warm"] = lr, warm
        hc = HealConfig(tokens=int(a.tokens), batch=c.get("heal_batch", 1), lr=lr, warmup=warm,
                        seed=hs, eval_every=10**9, log_every=25, amp=(dev == "cuda"))
        print(f"heal lr={lr} warmup={warm}", flush=True)
        store = TopKStore(str(Path(c["harvest"]) / "topk"))
        hist, seen = heal(model, store, hc, eval_fn=None, device=dev,
                          log=lambda r: print(r, flush=True))
    after = ev(model.eval())
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"name": name, "init": a.init, "tag": a.tag, "measure": a.measure,
           "tokens": a.tokens, "tokens_seen": seen, "seed": a.seed, "heal_seed": hs,
           "lr": lr if a.tokens > 0 else None, "warmup": warm if a.tokens > 0 else None,
           "trainable_params": n_train, "loss_before": before, "loss_after": after,
           "history": hist, "sha": sha, "stage_ckpt": loaded,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    json.dump(rec, open(dest, "w"))
    print(json.dumps({k: v for k, v in rec.items() if k != "history"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--init", choices=["stagewise", "random", "oracle"], required=True)
    p.add_argument("--measure", default="C"); p.add_argument("--q", type=float, default=1e8)
    p.add_argument("--tokens", type=float, default=1e6); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heal-seed", type=int, default=None,
                   help="vary the heal trajectory only; --seed still picks the stack")
    p.add_argument("--stack-suffix", default="",
                   help="load stage checkpoints with this suffix, e.g. _preDAgger")
    p.add_argument("--eval-rows", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0, help="override; 0 uses the config")
    p.add_argument("--tag", default="", help="suffix on the run name, e.g. _dagger")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
