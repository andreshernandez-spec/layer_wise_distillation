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

    # Key on the tag as well as the init: dagger_stack.py's re-heal is also
    # init="stagewise" at 1e6 and 1e7, and keying on init alone silently kept
    # whichever the glob returned last.
    heals = {}
    for f in sorted(glob.glob(str(out / "heal_*.json"))):
        r = json.load(open(f))
        tag = r.get("tag")
        if tag is None:                       # older records: recover it from the name
            stem = Path(f).stem.split("_")
            tag = "_" + stem[2] if len(stem) > 4 and stem[2] not in ("C", "R") else ""
        # a repeat on a second heal trajectory is the same arm, not a duplicate cell
        hs = r.get("heal_seed", r["seed"])
        key = r["init"] + tag
        if hs != r["seed"]:
            key += f"_h{hs}"
        if r["tokens"] in heals.get(key, {}):
            raise SystemExit(f"two records for {key} at {r['tokens']:.0e}: {f}")
        heals.setdefault(key, {})[r["tokens"]] = r
    print("=== the heal curve (DIAGNOSTIC, equal heal tokens: NOT the kill criterion)\n")
    print("Each cell's schedule is printed: arms tuned separately are not on the same")
    print("axis, and a comparison across different learning rates has to say so.\n")
    cols = ["stagewise", "stagewise_dagger", "random", "random_eqflops", "oracle"]
    cols = [k for k in cols if k in heals]
    print(f"{'heal tokens':>12s} " + " ".join(f"{k:>16s}" for k in cols) + " verdict")
    budgets = sorted({t for v in heals.values() for t in v})
    fired = []
    for t in budgets:
        sw = heals.get("stagewise", {}).get(t)
        rd = heals.get("random", {}).get(t)
        row = [f"{t:12.0e}"]
        for r in (heals[k].get(t) for k in cols):
            if r:
                lr = r.get("lr")
                cell = f"{r['loss_after']:.4f}" + (f"@{lr:.0e}" if lr else "")
                row.append(f"{cell:>16s}")
            else:
                row.append(f"{'-':>16s}")
        if sw and rd:
            row.append("stagewise better" if sw["loss_after"] < rd["loss_after"] else "random at least as good")
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
    print("\n=== criterion 4 (KILL): stagewise+heal vs random+heal at equal total FLOPs\n")
    sw_budgets = sorted(heals.get("stagewise", {}))
    if not sw_budgets:
        print("  no stagewise cell, criterion 4 not settled.")
    else:
        big = max(sw_budgets)
        need = big + extra
        sw = heals["stagewise"][big]
        print(f"  stagewise@{big:.0e} heal + {harvest + stage_flops:.3e} FLOPs of harvest+stages")
        print(f"  random needs {need:.4e} heal tokens to match, end to end.")
        # any random-init cell counts, whatever tag it carries: the equal-FLOPs cell is
        # tagged _eqflops so it cannot collide with the curve's 1e7 cell.
        rand_all = {t: r for k, v in heals.items() if k.startswith("random")
                    for t, r in v.items()}
        cand = [(t, r) for t, r in rand_all.items() if t >= need * 0.99]
        if not cand:
            have = max(rand_all, default=0)
            print(f"  NOT RUN YET: largest random cell is {have:.4e} tokens "
                  f"({have / need - 1:+.1%} of what it is owed). Criterion 4 is not settled.")
        else:
            t, rd = min(cand)
            gap = rd["loss_after"] - sw["loss_after"]
            print(f"  random@{t:.4e} = {rd['loss_after']:.4f}   stagewise@{big:.0e} = {sw['loss_after']:.4f}")
            print(f"  gap {gap:+.4f} nats (positive means the stagewise init is worth something)")
            if gap <= 0:
                print("  *** KILL FIRES: random init at equal FLOPs is at least as good. ***")
                fired.append(True)
            else:
                print("  kill does not fire.")
            # One seed per arm. Phase 1 measured 0.106 nats of seed spread on the
            # stitching delta over 42 same-arm pairs; there is no repeat here, so that
            # is the only scale available and a gap inside it is a tie, either way.
            # if the stagewise arm has repeats, the comparison should use their mean
            reps = [v[big]["loss_after"] for k, v in heals.items()
                    if k.startswith("stagewise_h") and big in v]
            if reps:
                vals = [sw["loss_after"]] + reps
                m = sum(vals) / len(vals)
                print(f"  stagewise over {len(vals)} heal trajectories: "
                      + ", ".join(f"{v:.4f}" for v in vals)
                      + f" -> mean {m:.4f}, spread {max(vals) - min(vals):.4f}")
                print(f"  gap against the mean: {rd['loss_after'] - m:+.4f} nats")
            if abs(gap) < 0.106:
                print(f"  but |{gap:+.4f}| is inside Phase 1's 0.106-nat seed spread and there is")
                print("  no repeated seed on this cell: read it as a tie, not as a margin.")

    print("\n=== criterion 5: gap from stagewise to the oracle at the largest budget")
    # the largest budget where BOTH arms exist, not the largest overall: the equal-FLOPs
    # cell is random-only and pushed max(budgets) somewhere criterion 5 cannot be read
    shared = sorted(set(heals.get("stagewise", {})) & set(heals.get("oracle", {})))
    if shared:
        big = shared[-1]
        sw, orc = heals["stagewise"][big], heals["oracle"][big]
        if sw and orc:
            print(f"  at {big:.0e} heal tokens (the largest both arms ran):")
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

    print("\n=== repeats: same cell, second heal trajectory")
    for base in sorted(k for k in heals if not k.endswith(tuple(f"_h{i}" for i in range(1, 9)))):
        for t, r0 in sorted(heals[base].items()):
            reps = [v[t]["loss_after"] for k, v in heals.items()
                    if k.startswith(base + "_h") and t in v]
            if reps:
                vals = [r0["loss_after"]] + reps
                print(f"  {base:22s} @{t:.0e}: " + ", ".join(f"{v:.4f}" for v in vals)
                      + f"   spread {max(vals) - min(vals):.4f}")

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
