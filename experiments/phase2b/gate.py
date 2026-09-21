"""Gate S1 of the data-limited protocol (docs/08): does any dose of noise, or the CF term,
beat the control on held-out stitching delta at the same real-data budget?

The control is the best configuration that uses neither (m = 0, lambda = 0), over the
weight decays it was given: beating an unregularized strawman would prove nothing.

Which configuration is "best" is decided on VALIDATION stitching delta, for the control
and for the treatment alike, and only then is the held-out number of the chosen one read.
There are seven treatment configurations against two controls; choosing among them on the
held-out score would hand the treatment the best of seven draws.

    python experiments/phase2b/gate.py out/phase2b/cells
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict


def main(d):
    cells = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f))
        cells[(r["budget"]["budget_rows"], r["m"], r["cf_lambda"], r["weight_decay"])].append(r)
    verdicts = {}
    for B in sorted({k[0] for k in cells}):
        rows = {k[1:]: v for k, v in cells.items() if k[0] == B}
        tokens = next(iter(rows.values()))[0]["budget"]["budget_tokens"]
        print(f"\n=== budget {B} rows ({tokens:,} real tokens), stage 2, held-out stitching delta (nats)\n")
        print(f"{'m':>4s} {'lambda':>6s} {'wd':>4s} {'n':>2s} {'held mean':>9s} {'spread':>7s} {'val':>7s} "
              f"{'best step':>9s} {'stop':>8s} {'positions':>10s} {'noise/real tok':>14s} {'passes':>7s} {'disp':>5s}")
        stat = {}
        for key in sorted(rows):
            rs = rows[key]
            h = [r["held_stitch_delta"] for r in rs]
            mean = sum(h) / len(h)
            var = sum((x - mean) ** 2 for x in h) / (len(h) - 1) if len(h) > 1 else float("nan")
            stat[key] = {"mean": mean, "var": var, "n": len(h), "vals": h,
                         "val": sum(r["val_stitch_delta"] for r in rs) / len(rs)}
            s0 = rs[0]["select"]
            print(f"{key[0]:4g} {key[1]:6g} {key[2]:4g} {len(h):2d} {mean:9.4f} {max(h) - min(h):7.4f} "
                  f"{sum(r['val_stitch_delta'] for r in rs) / len(rs):7.4f} "
                  f"{sum(r['select']['best_step'] for r in rs) / len(rs):9.0f} {s0['stop']:>8s} "
                  f"{sum(r['select']['positions'] for r in rs) / len(rs):10.3g} "
                  f"{sum(r['noise_per_distinct_real_position'] for r in rs) / len(rs):14.1f} "
                  f"{sum(r['real_passes'] for r in rs) / len(rs):7.1f} "
                  f"{sum(r['dispersion_student_over_teacher'] for r in rs) / len(rs):5.2f}")
        controls = {k: v for k, v in stat.items() if k[0] == 0 and k[1] == 0}
        treats = {k: v for k, v in stat.items() if not (k[0] == 0 and k[1] == 0)}
        if not controls or not treats:
            print("  control or treatment missing, gate not evaluable"); continue
        ck, c = min(controls.items(), key=lambda kv: kv[1]["val"])       # chosen on validation,
        tk, t = min(treats.items(), key=lambda kv: kv[1]["val"])         # judged on held-out
        pooled = [v["var"] for v in stat.values() if v["n"] > 1 and not math.isnan(v["var"])]
        s_pooled = math.sqrt(sum(pooled) / len(pooled)) if pooled else float("nan")
        gap = c["mean"] - t["mean"]
        separated = max(t["vals"]) < min(c["vals"])
        ok = gap > 2 * s_pooled and separated
        verdicts[B] = ok
        print(f"\n  control chosen on validation ({c['val']:.4f}): wd={ck[2]:g}  held-out {c['mean']:.4f}  {['%.4f' % x for x in c['vals']]}")
        print(f"  treatment chosen on validation ({t['val']:.4f}): m={tk[0]:g} lambda={tk[1]:g} wd={tk[2]:g}  held-out {t['mean']:.4f}  {['%.4f' % x for x in t['vals']]}")
        print(f"  gap {gap:+.4f}, pooled seed sd {s_pooled:.4f}, 2x = {2 * s_pooled:.4f}, ranges separated: {separated}")
        print(f"  S1 at this budget: {'PASS' if ok else 'fail'}")
    print("\nS1:", "PASS at budgets " + ", ".join(str(b) for b, v in verdicts.items() if v)
          if any(verdicts.values()) else "FAIL at every budget: the Phase 1 effect does not survive a fair baseline")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dir")
    sys.exit(main(p.parse_args().dir))
