"""All Phase 1 cells as one table (markdown with --md).

    python experiments/phase1/summarize.py out/phase1-1.4b [--md] [--min-q 1e6]
"""
import argparse
import glob
import json


def main(out, md, min_q):
    rows = []
    for f in sorted(glob.glob(f"{out}/*_s*.json")):
        r = json.load(open(f))
        if "history" not in r or r["q"] < min_q:
            continue
        h = r["history"]; ev = [(x["step"], x["eval"]) for x in h if "eval" in x]
        best = min(ev, key=lambda t: t[1])
        ae = r.get("attn_entropy", {}); harm = sum(ae["arm"]) / len(ae["arm"]) if ae else float("nan")
        tag = r["cell"].split("_s")[-1].split("_", 1)[1] if "_" in r["cell"].split("_s")[-1] else ""
        rows.append((r["measure"] + "_" + r["structure"], tag, r["q"], r["seed"], best[1], best[0], r["final_eval"],
                     r["stitch"]["delta"], r["jacobian"]["cos_mean"], harm, r["steps"], r["seconds"] / 60))
    rows.sort(key=lambda t: (t[1], t[2], t[4]))
    hdr = ["arm", "tag", "Q", "seed", "eps best", "@step", "eps final", "stitch", "jac", "H arm", "steps", "min"]
    if md:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for t in rows:
            print("| " + " | ".join([t[0], t[1] or "-", f"{t[2]:.0e}", str(t[3]), f"{t[4]:.3f}", str(t[5]), f"{t[6]:.3f}", f"{t[7]:.3f}", f"{t[8]:.3f}", f"{t[9]:.2f}", str(t[10]), f"{t[11]:.0f}"]) + " |")
    else:
        print(f"{'arm':8s} {'tag':4s} {'Q':>6s} {'sd':>2s} {'eps_best':>8s} {'@step':>6s} {'eps_fin':>8s} {'stitch':>7s} {'jac':>6s} {'H_arm':>6s} {'steps':>6s} {'min':>4s}")
        for t in rows:
            print(f"{t[0]:8s} {t[1] or '-':4s} {t[2]:6.0e} {t[3]:2d} {t[4]:8.3f} {t[5]:6d} {t[6]:8.3f} {t[7]:7.3f} {t[8]:6.3f} {t[9]:6.2f} {t[10]:6d} {t[11]:4.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--md", action="store_true"); p.add_argument("--min-q", type=float, default=1e6)
    a = p.parse_args()
    main(a.out, a.md, a.min_q)
