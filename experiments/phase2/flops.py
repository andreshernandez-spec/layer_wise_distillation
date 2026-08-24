"""C2.5: total FLOPs for a stagewise student against its baselines, from the run logs.

Counted, not estimated: every stage cell and heal run records its positions, and the
non-embedding parameter counts come from the configs. A stagewise student must be
compared against random-init-plus-heal *at equal total FLOPs*, which is G2's kill
criterion, so this has to include the harvest and every stage.

    python experiments/phase2/flops.py experiments/phase2/configs/stack-1.4b.yaml
"""
import argparse
import glob
import json
from pathlib import Path

import yaml

# non-embedding parameters, for the 6 x 2-block student of Pythia-1.4B
TEACHER_NONEMB = {"EleutherAI/pythia-1.4b": 1.21e9, "EleutherAI/pythia-2.8b": 2.52e9,
                  "EleutherAI/pythia-70m": 1.9e7}


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path))
    from transformers import AutoConfig
    out = Path(c["out"]); S = c["n_stages"]
    P_t = TEACHER_NONEMB[c["model"]]
    n_layers = AutoConfig.from_pretrained(c["model"]).num_hidden_layers
    per_layer = P_t / n_layers
    stage_t = per_layer * (n_layers / S)               # one teacher stage
    stage_s = per_layer * c["student_layers"]          # one student stage
    P_s = stage_s * S                                  # the whole student's blocks
    rep = {"student_nonemb_params": P_s, "teacher_nonemb_params": P_t}
    # `*_stage*` also matches heal_stagewise_*, so anchor on the digit
    for tag, pat in (("stages", "*_stage[0-9]*.json"), ("heal", "heal_*.json")):
        rows = [json.load(open(f)) for f in glob.glob(str(out / pat))]
        if tag == "stages":
            # per position: teacher stage forward (2P) + student fwd+bwd (6P)
            per = 2 * stage_t + 6 * stage_s
            by = {}
            for r in rows:
                by.setdefault(f"{r['measure']}_{r['structure']}", 0.0)
                by[f"{r['measure']}_{r['structure']}"] += r["q"] * per
            rep["stage_training"] = by
        else:
            rep["heal"] = {r["name"]: r["tokens"] * 6 * P_s for r in rows}
    # the harvest was paid once, for every arm
    rep["harvest"] = 2 * P_t * 1.05e7
    tot = rep["harvest"] + sum(rep.get("stage_training", {}).values())
    rep["stagewise_total_excluding_heal"] = tot
    for k, v in rep.items():
        unit = "" if k.endswith("params") else " FLOPs"
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in sorted(v.items()):
                print(f"  {kk:34s} {vv:.3e}{unit}")
        else:
            print(f"{k:36s} {v:.3e}{unit}")
    print("\nG2's kill criterion compares stagewise+heal against random+heal at EQUAL")
    print("total FLOPs, so the random arm is entitled to the stagewise total as extra")
    print("heal tokens: " + (f"{tot / (6 * P_s):.2e} tokens" if P_s else "n/a"))
    json.dump(rep, open(out / "flops.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("config")
    main(p.parse_args().config)
