"""Step 2 of the data-limited protocol (docs/08): compose a stack and heal it on the
budget, to its validation-selected best; or heal plain KD from random init the same way.

    python experiments/phase2b/heal.py CONFIG --init stack --m 8 --cf-lambda 0 --lr 1e-4 --seed 0
    python experiments/phase2b/heal.py CONFIG --init random --lr 1e-4 --seed 0

The heal reads only the budget: its top-k store holds the training rows and nothing else,
and it stops on next-token loss over the budget's validation rows. The number reported is
next-token loss on held-out Pile-test rows that nothing selects on.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell import cell_name                                       # noqa: E402

from lwd.compose.chain import Wrapped
from lwd.compose.model import StudentLM
from lwd.contract.whiten import Affine, Contract
from lwd.eval.stitch import next_token_loss
from lwd.harvest.model import Edges
from lwd.heal.select import HealSelectConfig, heal_selected
from lwd.heal.train import TopKStore
from lwd.stage.student import StudentStage, student_config

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main(a):
    c = yaml.safe_load(open(a.config))
    harvest, cells = Path(c["harvest"]), Path(c["cells"])
    out = Path(c["out"]); out.mkdir(parents=True, exist_ok=True)
    hmeta = json.load(open(harvest / "run.json"))
    budget = hmeta["budget"]
    assert budget, f"{harvest} is not a budgeted harvest"
    B, S, dev, dt = budget["budget_rows"], c["n_stages"], c["device"], DT[c["dtype"]]
    stack = "random" if a.init == "random" else f"m{a.m:g}_cf{a.cf_lambda:g}_wd{a.wd:g}_s{a.stack_seed}"
    name = f"heal_b{B}_{stack}_lr{a.lr:g}_h{a.seed}" + (f"_{a.tag}" if a.tag else "")
    dest = out / f"{name}.json"
    if dest.exists() and not a.overwrite:
        raise SystemExit(f"{dest} exists; pass --tag or --overwrite")

    phis = []
    for k in range(S + 1):
        st = np.load(harvest / f"stats_iface{k}.npz")
        assert int(st["mc_n"]) == budget["train_tokens"], "statistics are not from the training rows alone"
        phis.append(Contract(None, Affine.zca(torch.from_numpy(st["mc_mean"]),
                                              torch.from_numpy(st["mc_cov_shrunk"]))).to(dev, torch.float32))
    d = phis[0].affine.mean.numel()
    stages, stage_recs = [], []
    for k in range(S):
        stu = StudentStage(student_config(c["model"], d, c["student_layers"], c["student_heads"]),
                           seed=a.seed + 1000 if a.init == "random" else a.stack_seed)
        if a.init == "stack":
            cn = cell_name(B, k, a.m, a.cf_lambda, a.wd, a.stack_seed)
            stu.load_state_dict(torch.load(cells / f"{cn}.pt", map_location="cpu"))
            r = json.load(open(cells / f"{cn}.json"))
            assert r["budget"]["sha256_train"] == budget["sha256_train"], f"{cn} was trained on another budget"
            stage_recs.append({"cell": cn, "flops": r["flops"]["total"], "positions": r["select"]["positions"],
                               "noise_positions": r["select"]["noise_positions"],
                               "best_step": r["select"]["best_step"], "stop": r["select"]["stop"],
                               "held_stitch_delta": r["held_stitch_delta"]})
        stages.append(Wrapped(stu.to(dev), phis[k], phis[k + 1]))
    edges = Edges(c["model"], dt, dev)
    for p in edges.parameters():
        p.requires_grad_(False)
    model = StudentLM(edges, stages).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    L = 2048
    n_val = min(budget["val_rows"], c.get("val_seqs_max", budget["val_rows"]))
    val_ids = torch.from_numpy(np.load(harvest / "val_rows.npy")[:n_val, : L + 1].astype(np.int64))
    held_ids = torch.from_numpy(np.load(c["heldout"])[: c["eval_rows"], : L + 1].astype(np.int64))
    store = TopKStore(str(harvest / "topk"))
    assert abs(store.epoch_tokens() - budget["train_tokens"]) <= 2048 * 8, \
        f"the heal store holds {store.epoch_tokens()} tokens, the budget's training rows {budget['train_tokens']}"

    before = next_token_loss(model.eval(), held_ids, batch=1)
    hc = HealSelectConfig(**{**c["heal"], "lr": a.lr, "seed": a.seed})
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not sha and Path("SHA").exists():       # a pod gets an rsync of the tree, not the repo
        sha = Path("SHA").read_text().strip()
    print(json.dumps({"heal": name, "lr": a.lr, "init": a.init, "budget": budget, "sha": sha}), flush=True)
    t0 = time.time()
    hist, summary = heal_selected(model, store, hc, lambda m: next_token_loss(m, val_ids, batch=1),
                                  device=dev, log=lambda r: print(r, flush=True))
    after = next_token_loss(model.eval(), held_ids, batch=1)
    flops_heal = summary["tokens_seen"] * 6 * n_train
    flops_stages = sum(r["flops"] for r in stage_recs)
    rec = {"name": name, "init": a.init, "m": a.m if a.init == "stack" else None,
           "cf_lambda": a.cf_lambda if a.init == "stack" else None,
           "weight_decay_stages": a.wd if a.init == "stack" else None, "stack_seed": a.stack_seed,
           "lr": a.lr, "heal_seed": a.seed, "budget": budget, "select": summary,
           "held_loss_before": before, "held_loss": after, "val_loss": summary["selected_val"],
           "stages": stage_recs, "trainable_params": n_train,
           "flops": {"heal": flops_heal, "stages": flops_stages, "total": flops_heal + flops_stages},
           "noise_positions": sum(r["noise_positions"] for r in stage_recs),
           "schedule": {k: getattr(hc, k) for k in ("lr", "warmup", "val_every", "patience",
                                                     "cooldown_frac", "cap_tokens", "batch")},
           "config": c, "history": hist, "seconds": time.time() - t0,
           "env": {"sha": sha, "torch": torch.__version__, "python": platform.python_version(),
                   "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                   "sha256_train": budget["sha256_train"]}}
    json.dump(rec, open(dest, "w"))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("history", "config", "stages")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--init", choices=["stack", "random"], required=True)
    p.add_argument("--m", type=float, default=0.0)
    p.add_argument("--cf-lambda", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=0.1, help="weight decay the stack's stages were trained with")
    p.add_argument("--stack-seed", type=int, default=0)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0, help="heal seed")
    p.add_argument("--tag", default="")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
