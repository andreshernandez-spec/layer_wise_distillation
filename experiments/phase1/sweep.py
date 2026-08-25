"""Resumable Phase 1 sweep: runs every (measure, structure, q, seed) cell in a grid
config that has no JSON yet. One process per cell so a crash costs one cell.

    python experiments/phase1/sweep.py experiments/phase1/configs/cell-1.4b.yaml \
        experiments/phase1/configs/grid-1.4b.yaml [--dry]
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def cells(grid):
    for arm in grid["arms"]:
        m, s = arm.split("_")
        # q_by_arm lets a known-negative arm stop early instead of spending the same
        # budget as the arms still in the race
        qs = grid.get("q_by_arm", {}).get(arm) or (grid["q_real"] if m in ("R", "L") else grid["q"])
        for q in qs:
            seeds = grid["seeds"] if float(q) <= float(grid.get("multi_seed_below", 0)) else [0]
            for seed in seeds:
                yield m, s, float(q), seed


def name(m, s, q, seed, tag=""):
    return f"{m}_{s}_q{q:.0e}_s{seed}".replace("+", "") + (f"_{tag}" if tag else "")


def main(a):
    c = yaml.safe_load(open(a.config)); grid = yaml.safe_load(open(a.grid))
    out = Path(c["out"])
    todo = [(m, s, q, seed) for m, s, q, seed in cells(grid) if not (out / f"{name(m, s, q, seed, grid.get('tag', ''))}.json").exists()]
    total = sum(1 for _ in cells(grid))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        todo = todo[i::n]
        print(f"shard {i}/{n}: {len(todo)} cells", flush=True)
    print(f"{len(todo)} of {total} cells to run", flush=True)
    for m, s, q, seed in todo:
        cmd = [sys.executable, "experiments/phase1/run.py", a.config, "--measure", m, "--structure", s,
               "--q", str(q), "--seed", str(seed)]
        if grid.get("tag"):
            cmd += ["--tag", grid["tag"]]
        print(" ".join(cmd), flush=True)
        if a.dry:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"cell {name(m, s, q, seed)} failed ({r.returncode}); continuing", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config"); p.add_argument("grid"); p.add_argument("--dry", action="store_true")
    p.add_argument("--shard", default="", help="i/n: run every nth cell, for concurrent workers")
    main(p.parse_args())
