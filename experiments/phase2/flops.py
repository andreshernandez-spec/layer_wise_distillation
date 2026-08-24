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


def main(cfg_path, a_heal_tokens=1e7):
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
    # `*_stage*` also matches heal_stagewise_* and dagger_C_stage0_*, so pin the
    # whole stage-cell shape: measure_structure_q..._s..._stageK. Counting a DAgger
    # retrain here would inflate the stagewise budget 10% and hand the random arm
    # tokens the plain stack never spent.
    for tag, pat in (("stages", "*_q[0-9]*_s[0-9]*_stage[0-9].json"), ("heal", "heal_*.json")):
        files = sorted(glob.glob(str(out / pat)))
        rows = [json.load(open(f)) for f in files]
        if tag == "stages":
            bad = [f for f, r in zip(files, rows) if "measure" not in r or "q" not in r]
            if bad:
                raise SystemExit(f"not stage cells: {bad}")
            if len(rows) != S:
                raise SystemExit(f"expected {S} stage cells, found {len(rows)}: {files}")
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
    # G2 compares the two arms at equal FLOPs *end to end*, so the stagewise arm's own
    # heal counts on its side of the ledger and the random arm has to be paid for it too.
    # Reporting only tot/(6*P_s) reads as the whole budget and is short by that heal:
    # it was, by 5.3%, in the first 1.74e8 run.
    heal_sw = a_heal_tokens * 6 * P_s
    end_to_end = tot + heal_sw
    rep["stagewise_heal_tokens"] = a_heal_tokens
    rep["stagewise_total_end_to_end"] = end_to_end
    rep["equal_flops_tokens"] = end_to_end / (6 * P_s)
    print(f"{'stagewise_total_end_to_end':36s} {end_to_end:.3e} FLOPs"
          f"  (incl. its {a_heal_tokens:.0e}-token heal)")
    print("\nG2's kill criterion compares stagewise+heal against random+heal at EQUAL")
    print("total FLOPs end to end. The random arm's whole budget is")
    print(f"  {rep['equal_flops_tokens']:.4e} tokens")
    print(f"which is {tot / (6 * P_s):.3e} for the harvest and stages plus "
          f"{a_heal_tokens:.0e} for the heal.")
    json.dump(rep, open(out / "flops.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("config")
    p.add_argument("--heal-tokens", type=float, default=1e7,
                   help="the stagewise arm's heal budget, which the random arm is also paid for")
    a = p.parse_args()
    main(a.config, a.heal_tokens)
