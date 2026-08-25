"""G1 assessment (docs/02): reads the cells and the early-stopped fit, states each
criterion against the pre-registered margins, and writes the budget forecast.

    python experiments/phase1/gate.py out/phase1-1.4b [--eps-margin 1.5] [--stitch-margin 2.0]
"""
import argparse
import glob
import json
from collections import defaultdict

# per-position FLOPs for the stage (teacher forward + student fwd/bwd), docs/compute.md
TIERS = {"tier0 (2.8B->410M, 8 stages)": 8.5e8 * 8, "tier1 (OLMo-7B->1B, 8 stages)": 2.4e9 * 8,
         "tier2 (32B->2B, 16 stages)": 6e9 * 16}


def main(out, eps_margin, stitch_margin):
    cells = [json.load(open(f)) for f in glob.glob(f"{out}/*_s*.json")]
    by = defaultdict(dict)
    for r in cells:
        arm = f"{r['measure']}_{r['structure']}"
        best = min(x["eval"] for x in r["history"] if "eval" in x)
        by[arm].setdefault(r["q"], []).append({"best": best, "final": r["final_eval"], "stitch": r["stitch"]["delta"], "seed": r["seed"]})
    ref_arm = "L_iid" if "L_iid" in by else "R_iid"
    ref = by[ref_arm].get(1e7) or by[ref_arm][max(by[ref_arm])]
    eps_ref = min(x["best"] for x in ref); stitch_ref = min(x["stitch"] for x in ref)
    print(f"reference {ref_arm} at 1e7: eps {eps_ref:.3f}, stitch {stitch_ref:.3f} nats")
    print(f"criterion 3 (kill): best noise cell needs eps_inf <= {eps_margin}x{eps_ref:.3f} = {eps_margin*eps_ref:.3f}"
          f" and stitch at its largest Q <= {stitch_margin}x{stitch_ref:.3f} = {stitch_margin*stitch_ref:.3f}")
    try:
        fit = json.load(open(f"{out}/fit_best_eval.json"))
    except FileNotFoundError:
        fit = {}
    for arm in sorted(by):
        if arm in (ref_arm, "R_iid"):
            continue
        qs = sorted(by[arm]); top = by[arm][qs[-1]]
        best_eps = min(min(x["best"] for x in by[arm][q]) for q in qs)
        stitch_top = min(x["stitch"] for x in top)
        f = fit.get(arm, {})
        eps_inf = f.get("eps_inf"); ci = f.get("eps_inf_ci"); qstar = f.get("Q_star")
        verdict = []
        lim = eps_margin * eps_ref
        if ci and ci[1] <= lim:
            verdict.append("eps PASS (CI upper bound clears the margin)")
        elif ci and ci[0] > lim:
            verdict.append("eps FAIL (CI lower bound misses the margin)")
        else:
            verdict.append("eps UNDETERMINED (CI straddles the margin; more Q / seeds)")
        verdict.append("stitch PASS" if stitch_top <= stitch_margin * stitch_ref else "stitch FAIL")
        print(f"{arm:8s} Qmax {qs[-1]:.0e} best eps {best_eps:.3f} stitch@Qmax {stitch_top:.3f} | fit eps_inf {eps_inf} CI {ci} beta {f.get('beta')} Q* {qstar} | {', '.join(verdict)}")
        if qstar:
            print("   budget if Q* positions per stage:")
            for name, flops in TIERS.items():
                total = qstar * flops
                print(f"     {name}: {total:.1e} FLOPs total ({total / 8e12 / 3600:.0f} T4-hours at 8 TFLOP/s effective)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--eps-margin", type=float, default=1.5); p.add_argument("--stitch-margin", type=float, default=2.0)
    a = p.parse_args()
    main(a.out, a.eps_margin, a.stitch_margin)
