"""G2 assessment (docs/03): reads the heal runs, the drift profiles and the DAgger
cells, and states each criterion.

The kill criterion is the one with a trap in it. It compares the stagewise student
against random-init **at equal total FLOPs**, not at equal heal tokens: the stagewise
arm has already spent the harvest and every stage training, so the random arm is
entitled to that budget as extra heal tokens. Comparing at equal heal tokens silently
favours stagewise and the criterion could never fire.

    python experiments/phase2/gate.py experiments/phase2/configs/stack-1.4b.yaml
"""
import argparse
import glob
import json
from pathlib import Path

import yaml

TEACHER_NONEMB = {"EleutherAI/pythia-1.4b": 1.21e9, "EleutherAI/pythia-2.8b": 2.52e9,
                  "EleutherAI/pythia-70m": 1.9e7}


def main(cfg_path):
    c = yaml.safe_load(open(cfg_path))
    out = Path(c["out"]); S = c["n_stages"]
    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(c["model"]).num_hidden_layers
    P_t = TEACHER_NONEMB[c["model"]]
    P_s = P_t / n_layers * c["student_layers"] * S

    heals = {}
    for f in glob.glob(str(out / "heal_*.json")):
        r = json.load(open(f))
        heals.setdefault(r["init"], {})[r["tokens"]] = r
    print("=== criterion 4 (kill): stagewise vs random-init, held-out next-token loss\n")
    print("Each cell's schedule is printed: arms tuned separately are not on the same")
    print("axis, and a comparison across different learning rates has to say so.\n")
    print(f"{'heal tokens':>12s} {'stagewise':>10s} {'random':>10s} {'oracle':>10s} {'verdict':s}")
    budgets = sorted({t for v in heals.values() for t in v})
    fired = []
    for t in budgets:
        sw = heals.get("stagewise", {}).get(t)
        rd = heals.get("random", {}).get(t)
        orc = heals.get("oracle", {}).get(t)
        row = [f"{t:12.0e}"]
        for r in (sw, rd, orc):
            if r:
                lr = r.get("lr")
                row.append(f"{r['loss_after']:10.4f}" + (f"@{lr:.0e}" if lr else ""))
            else:
                row.append(f"{'-':>10s}")
        if sw and rd:
            ok = sw["loss_after"] < rd["loss_after"]
            row.append("stagewise better" if ok else "KILL: random is at least as good")
            fired.append(not ok)
        print(" ".join(row))

    # equal-FLOPs entitlement
    # only the arm being compared: both stacks live in the same directory, and
    # counting them together doubles the entitlement
    arm = "C_mix"
    stage_flops = 0.0
    n_stage_cells = 0
    for f in glob.glob(str(out / f"{arm}_*_stage[0-9]*.json")):
        r = json.load(open(f))
        stage_flops += r["q"] * (2 * P_t / S + 6 * P_s / S)
        n_stage_cells += 1
    harvest = 2 * P_t * 1.05e7
    extra = (harvest + stage_flops) / (6 * P_s)
    print(f"\nstagewise ({arm}, {n_stage_cells} stage cells) spent {harvest + stage_flops:.2e} FLOPs before healing;")
    print(f"at equal total FLOPs the random arm is entitled to {extra:.2e} extra heal tokens.")
    if budgets:
        big = max(budgets)
        print(f"So the honest comparison is stagewise@{big:.0e} against random@{big + extra:.2e}.")
        if not any(t >= big + extra for t in heals.get("random", {})):
            print("  NOT RUN YET: that random cell is missing, so criterion 4 is not settled.")

    print("\n=== criterion 5: gap from stagewise to the oracle at the largest budget")
    if budgets:
        big = max(budgets)
        sw, orc = heals.get("stagewise", {}).get(big), heals.get("oracle", {}).get(big)
        if sw and orc:
            print(f"  {sw['loss_after'] - orc['loss_after']:+.4f} nats "
                  f"(stagewise {sw['loss_after']:.4f}, oracle {orc['loss_after']:.4f})")
        else:
            print("  pending")

    print("\n=== criterion 2: drift profile")
    for m in ("R", "C"):
        p = out / f"drift_{m}.json"
        if p.exists():
            d = json.load(open(p))
            print(f"  {m}: drift {' '.join(f'{x:.3f}' for x in d['realized_drift'])}")
            print(f"     lipschitz {' '.join(f'{x:.2f}' for x in d['lipschitz'])}")
        else:
            print(f"  {m}: pending")

    print("\n=== criterion 3: mitigations")
    dg = sorted(glob.glob(str(out / "dagger_*.json")))
    if not dg:
        print("  pending")
    for f in dg:
        r = json.load(open(f))
        print(f"  stage {r['stage']} on-policy {r['on_policy']}: drifted eps "
              f"{r['eps_drifted_before']:.4f} -> {r['eps_drifted_after']:.4f} "
              f"(clean {r['eps_clean_before']:.4f} -> {r['eps_clean_after']:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("config")
    main(p.parse_args().config)
