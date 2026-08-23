"""Fit eps(Q) = c Q^-beta + eps_inf per arm with bootstrap CIs (docs/02 'Fit').

    python experiments/phase1/fit.py out/phase1-1.4b [--metric final_eval]
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit


def model(Q, c, beta, eps_inf):
    return c * Q ** (-beta) + eps_inf


def fit(Q, e, n_boot=200, seed=0):
    Q, e = np.asarray(Q, float), np.asarray(e, float)
    p0 = [e[0] * Q[0] ** 0.3, 0.3, max(e.min() * 0.5, 1e-6)]
    bounds = ([0, 0, 0], [np.inf, 3, np.inf])
    popt, _ = curve_fit(model, Q, e, p0=p0, bounds=bounds, maxfev=20000)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Q), len(Q))
        if len(set(Q[i])) < 3:
            continue
        try:
            b, _ = curve_fit(model, Q[i], e[i], p0=popt, bounds=bounds, maxfev=20000)
            boots.append(b)
        except RuntimeError:
            pass
    boots = np.array(boots) if boots else popt[None]
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    return {"c": popt[0], "beta": popt[1], "eps_inf": popt[2],
            "beta_ci": [lo[1], hi[1]], "eps_inf_ci": [lo[2], hi[2]], "n_points": len(Q)}


def main(out, metric):
    by_arm = defaultdict(list)
    for f in glob.glob(f"{out}/*.json"):
        r = json.load(open(f))
        if "final_eval" not in r:
            continue
        if metric == "best_eval":  # early-stopped: minimum held-out eps over the trajectory
            val = min(x["eval"] for x in r["history"] if "eval" in x)
        elif metric == "stitch":
            val = r["stitch"]["delta"]
        else:
            val = r[metric]
        by_arm[f"{r['measure']}_{r['structure']}"].append((r["q"], val))
    report = {}
    e_real = None
    if "R_iid" in by_arm:
        pts = sorted(by_arm["R_iid"])
        e_real = min(e for q, e in pts if q >= 1e7) if any(q >= 1e7 for q, _ in pts) else min(e for _, e in pts)
    for arm, pts in sorted(by_arm.items()):
        Q, e = zip(*sorted(pts))
        if len(set(Q)) < 3:
            report[arm] = {"n_points": len(Q), "note": "fewer than 3 Q values"}
            continue
        r = fit(Q, e)
        if e_real is not None and r["eps_inf"] < e_real:
            r["Q_star"] = (r["c"] / (e_real - r["eps_inf"])) ** (1 / max(r["beta"], 1e-6))
        else:
            r["Q_star"] = None
        report[arm] = r
        print(arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    report["_eps_real_ref"] = e_real
    json.dump(report, open(f"{out}/fit_{metric}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--metric", default="final_eval")
    a = p.parse_args()
    main(a.out, a.metric)
