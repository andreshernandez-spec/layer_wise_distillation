"""Phase 2 step 1: train every stage of a student, not just one (docs/03).

Phase 1 answered "how good can one stage get". Phase 2 needs all of them, in both
recipes: R (anchors only) for the abundant regime and C_mix (anchors + noise) for the
scarce one. Each stage is an independent process, resumable by skipping finished ones.

    python experiments/phase2/train_stages.py experiments/phase2/configs/stack-1.4b.yaml --measure R
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"])
    struct = "mix" if a.measure == "C" else "iid"
    todo = []
    for k in range(c["n_stages"]):
        name = f"stage{k}_{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}".replace("+", "")
        if not (out / f"{name}.json").exists():
            todo.append((k, name))
    print(f"{len(todo)} of {c['n_stages']} stages to train", flush=True)
    for k, name in todo:
        cfg = dict(c)
        cfg["stage"] = k
        cfg["out"] = str(out)
        tmp = out / f"_cfg_stage{k}.yaml"
        out.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(tmp, "w"))
        cmd = [sys.executable, "experiments/phase1/run.py", str(tmp), "--measure", a.measure,
               "--structure", struct, "--q", str(a.q), "--seed", str(a.seed), "--tag", f"stage{k}"]
        print(" ".join(cmd), flush=True)
        if a.dry:
            continue
        if subprocess.run(cmd).returncode != 0:
            print(f"stage {k} failed; continuing", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--measure", choices=["R", "L", "C", "G"], required=True)
    p.add_argument("--q", type=float, default=1e8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry", action="store_true")
    main(p.parse_args())
