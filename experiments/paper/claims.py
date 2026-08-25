"""Every number the paper asserts, recomputed from the result files.

The paper must not carry a transcribed number. One was transcribed from the wrong line
of a FLOPs report during Phase 2 and cost a 35-minute run; the docs are written by hand
and can drift the same way. This emits each claim with its value, the files it came
from, and (where the docs state it) a check against the recorded figure.

    python experiments/paper/claims.py                 # table
    python experiments/paper/claims.py --json out.json # machine readable
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from fit import fit as fit_beta            # noqa: E402

P1 = "out/phase1-1.4b-a100"
P2 = "out/phase2-1.4b-a100"
TEACHER_NONEMB, N_LAYERS, S, STUDENT_LAYERS = 1.21e9, 24, 6, 2


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def claim(reg, key, value, source, doc=None, tol=5e-3):
    """doc is what a docs/ table says; a mismatch is a bug in one of the two."""
    value = float(value)                       # numpy scalars do not serialize
    ok = None
    if doc is not None:
        ok = bool(abs(value - doc) <= tol * max(1.0, abs(doc)))
    reg[key] = {"value": value, "source": source, "doc": doc, "matches_doc": ok}
    return value


# what docs/02's beta table published, to catch it drifting from the data
DOC_BETA = {"L_iid": 0.326, "R_iid": 0.336, "C_mix": 0.313, "C_ar1": 0.279,
            "G_iid": 0.334, "I_iid": 0.206}


def phase1(reg):
    """Same path as experiments/phase1/fit.py --metric best_eval: one point per cell,
    no aggregation, eps read as the minimum held-out value over the trajectory."""
    rows = [r for r in load(f"{P1}/*.json") if "final_eval" in r]
    by_arm = defaultdict(list)
    for r in rows:
        best = min(x["eval"] for x in r["history"] if "eval" in x)
        by_arm[f"{r['measure']}_{r['structure']}"].append((r["q"], best))
    for arm, pts in sorted(by_arm.items()):
        if len({q for q, _ in pts}) < 4:
            continue
        f = fit_beta([q for q, _ in pts], [e for _, e in pts])
        claim(reg, f"beta[{arm}]", f["beta"], f"{P1}/*.json, {len(pts)} cells",
              doc=DOC_BETA.get(arm), tol=0.01)
        reg[f"beta[{arm}]"]["ci"] = [round(x, 4) for x in f["beta_ci"]]
        reg[f"beta[{arm}]"]["eps_inf"] = f["eps_inf"]
        reg[f"beta[{arm}]"]["n_cells"] = len(pts)
    claim(reg, "phase1_cells", float(len(rows)), P1)


def phase2(reg):
    heals = {}
    for r in load(f"{P2}/heal_*.json"):
        tag = r.get("tag", "")
        hs = r.get("heal_seed", r["seed"])
        key = r["init"] + tag + (f"_h{hs}" if hs != r["seed"] else "")
        heals[(key, r["tokens"])] = r

    def L(key, tok):
        return heals[(key, tok)]["loss_after"]

    sw = [L("stagewise", 1e7), L("stagewise_h1", 1e7)]
    dg = [L("stagewise_dagger", 1e7), L("stagewise_dagger_h1", 1e7)]
    rnd_eq = L("random_eqflops", 183670000.0)

    claim(reg, "heal.stagewise@1e7.mean", float(np.mean(sw)), f"{P2}/heal_stagewise_*", doc=3.7560)
    claim(reg, "heal.stagewise@1e7.spread", float(np.ptp(sw)), f"{P2}/heal_stagewise_*", doc=0.0679)
    claim(reg, "heal.dagger@1e7.mean", float(np.mean(dg)), f"{P2}/heal_stagewise_dagger_*", doc=3.6602)
    claim(reg, "heal.dagger@1e7.spread", float(np.ptp(dg)), f"{P2}/heal_stagewise_dagger_*", doc=0.0454)
    claim(reg, "heal.random_equalflops", rnd_eq, f"{P2}/heal_random_eqflops_*", doc=3.5430)
    claim(reg, "heal.oracle@1e7", L("oracle", 1e7), f"{P2}/heal_oracle_*", doc=3.4850)
    claim(reg, "heal.random@1e7", L("random", 1e7), f"{P2}/heal_random_C_t1e07*", doc=5.2693)

    claim(reg, "kill.gap_vs_seed0", rnd_eq - L("stagewise", 1e7), "derived", doc=-0.1792)
    claim(reg, "kill.gap_vs_mean", rnd_eq - float(np.mean(sw)), "derived", doc=-0.2131)
    claim(reg, "kill.gap_vs_dagger", rnd_eq - float(np.mean(dg)), "derived")
    claim(reg, "criterion5.gap", L("stagewise", 1e7) - L("oracle", 1e7), "derived", doc=0.2371)
    claim(reg, "dagger.healed_effect", float(np.mean(sw)) - float(np.mean(dg)), "derived")
    claim(reg, "dagger.unhealed_effect",
          heals[("stagewise", 1e7)]["loss_before"] - heals[("stagewise_dagger", 1e7)]["loss_before"],
          "derived", doc=3.3379)

    # equal-FLOPs budget, from the same arithmetic the runs used
    per_layer = TEACHER_NONEMB / N_LAYERS
    stage_t, stage_s = per_layer * (N_LAYERS / S), per_layer * STUDENT_LAYERS
    P_s = stage_s * S
    per = 2 * stage_t + 6 * stage_s
    stages = [r for r in load(f"{P2}/*_q[0-9]*_s[0-9]*_stage[0-9].json")]
    tot = 2 * TEACHER_NONEMB * 1.05e7 + sum(r["q"] * per for r in stages)
    claim(reg, "flops.stagewise_end_to_end", tot + 1e7 * 6 * P_s, "derived", doc=6.667e17, tol=1e-3)
    claim(reg, "flops.equal_tokens", (tot + 1e7 * 6 * P_s) / (6 * P_s), "derived", doc=1.8367e8, tol=1e-3)
    claim(reg, "flops.n_stage_cells", float(len(stages)), f"{P2}/*_stage*.json", doc=6.0)

    for m in ("R", "C", "C_dagger"):
        p = Path(f"{P2}/drift_{m}.json")
        if p.exists():
            d = json.load(open(p))
            claim(reg, f"drift.{m}.output", d["realized_drift"][-1], str(p))
            reg[f"drift.{m}.output"]["profile"] = [round(x, 4) for x in d["realized_drift"]]
    # Two platforms measured the same R stack. The contractive stages agree; stage 0,
    # which is strongly expansive, does not, because the estimator perturbs two
    # sequences with one noise draw. Quote it as a range or not at all.
    a100 = json.load(open(f"{P2}/drift_R.json"))["lipschitz"]
    lap = json.load(open("out/phase2-1.4b/drift_R.json"))["lipschitz"]
    claim(reg, "lipschitz.stage0.a100", a100[0], f"{P2}/drift_R.json", doc=72.67, tol=1e-3)
    claim(reg, "lipschitz.stage0.laptop", lap[0], "out/phase2-1.4b/drift_R.json", doc=48.53, tol=1e-3)
    claim(reg, "lipschitz.stage0.disagreement", abs(a100[0] - lap[0]) / min(a100[0], lap[0]),
          "two platforms, n=2 sequences each")
    claim(reg, "lipschitz.contractive_max_disagreement",
          max(abs(x - y) / min(x, y) for x, y in zip(a100[1:], lap[1:])),
          "stages 1 to 5, two platforms")
    claim(reg, "lipschitz.rest_max", max(a100[1:]), f"{P2}/drift_R.json")
    claim(reg, "drift.C.ratio_product_at_iface2",
          json.load(open(f"{P2}/drift_C.json"))["predicted_from_stage0_drift"][1], f"{P2}/drift_C.json")

    dg_cells = load(f"{P2}/dagger_C_stage*.json")
    cuts = [1 - r["eps_drifted_after"] / r["eps_drifted_before"] for r in dg_cells]
    claim(reg, "dagger.cut_stage0", cuts[0], f"{P2}/dagger_C_stage0*")
    claim(reg, "dagger.cut_min_stages1to5", min(cuts[1:]), f"{P2}/dagger_C_stage*")
    claim(reg, "dagger.cut_max_stages1to5", max(cuts[1:]), f"{P2}/dagger_C_stage*")


def main(a):
    reg = {}
    phase1(reg)
    phase2(reg)
    bad = [k for k, v in reg.items() if v["matches_doc"] is False]
    if a.json:
        json.dump(reg, open(a.json, "w"), indent=1)
    w = max(len(k) for k in reg)
    for k, v in reg.items():
        mark = {True: "ok", False: "MISMATCH", None: ""}[v["matches_doc"]]
        doc = f"  doc={v['doc']:.4g}" if v["doc"] is not None else ""
        print(f"{k:{w}s} {v['value']:>14.6g}{doc:>16s}  {mark}")
    print(f"\n{len(reg)} claims, {len(bad)} disagreeing with the docs")
    if bad:
        print("MISMATCHED:", ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    sys.exit(main(p.parse_args()))
