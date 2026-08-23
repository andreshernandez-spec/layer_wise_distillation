"""Same cell, two platforms: how much of an eps difference is the hardware?

Training is chaotic, so two GPUs running the identical config diverge. This says by
how much, which is the bar any cross-platform claim has to clear.

    python experiments/phase1/compare_platforms.py out/phase1-1.4b out/phase1-1.4b-a100
"""
import glob
import json
import os
import sys


def load(d):
    out = {}
    for f in glob.glob(f"{d}/*_s*.json"):
        r = json.load(open(f))
        if "history" not in r:
            continue
        out[os.path.basename(f)[:-5]] = {
            "eps": r["final_eval"],
            "best": min(x["eval"] for x in r["history"] if "eval" in x),
            "stitch": r["stitch"]["delta"],
            "device": r["env"].get("device", "?"),
            "torch": r["env"].get("torch", "?"),
        }
    return out


def main(a, b):
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    if not common:
        print(f"no cells in common ({len(A)} vs {len(B)})")
        return
    print(f"{len(common)} cells in common")
    print(f"{'cell':28s} {'eps A':>8s} {'eps B':>8s} {'rel diff':>9s} {'stitch A':>9s} {'stitch B':>9s}")
    rels = []
    for k in common:
        ra, rb = A[k], B[k]
        rel = abs(ra["eps"] - rb["eps"]) / max(ra["eps"], 1e-9)
        rels.append(rel)
        print(f"{k:28s} {ra['eps']:8.4f} {rb['eps']:8.4f} {100*rel:8.2f}% {ra['stitch']:9.4f} {rb['stitch']:9.4f}")
    print(f"\nA: {A[common[0]]['device']} torch {A[common[0]]['torch']}")
    print(f"B: {B[common[0]]['device']} torch {B[common[0]]['torch']}")
    print(f"platform effect on eps: median {100*sorted(rels)[len(rels)//2]:.2f}%, max {100*max(rels):.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
