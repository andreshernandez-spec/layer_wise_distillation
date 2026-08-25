"""Retrain a whole stack on-policy, in order, then leave it composable (docs/03 C2.3).

Stage k propagates through the stages below it *as already retrained*, which is what
iterative DAgger means: each stage sees the distribution the final stack will actually
produce, not the one the original stages produced.

    python experiments/phase2/dagger_stack.py CONFIG --measure C --q-retrain 1e7
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main(a):
    c = yaml.safe_load(open(a.config))
    out = Path(c["out"])
    struct = "mix" if a.measure == "C" else "iid"
    for k in range(c["n_stages"]):
        base = f"{a.measure}_{struct}_q{a.q:.0e}_s{a.seed}_stage{k}".replace("+", "")
        cmd = [sys.executable, "experiments/phase2/dagger.py", a.config, "--stage", str(k),
               "--measure", a.measure, "--q", str(a.q), "--q-retrain", str(a.q_retrain),
               "--on-policy", str(a.on_policy), "--seed", str(a.seed)]
        print(" ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"stage {k} failed; stopping (later stages depend on it)", flush=True)
            return
        # promote the retrained stage so the next one propagates through it
        dg = out / f"dagger_{a.measure}_stage{k}_p{a.on_policy}_q{a.q_retrain:.0e}.pt".replace("+", "")
        if not (out / f"{base}_preDAgger.pt").exists():
            shutil.copy(out / f"{base}.pt", out / f"{base}_preDAgger.pt")
        shutil.copy(dg, out / f"{base}.pt")
        print(f"stage {k} promoted", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config"); p.add_argument("--measure", default="C")
    p.add_argument("--q", type=float, default=1e8); p.add_argument("--q-retrain", type=float, default=1e7)
    p.add_argument("--on-policy", type=float, default=0.5); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
