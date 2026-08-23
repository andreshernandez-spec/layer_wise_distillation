"""Compare two statistics sets interface by interface (e.g. interim 1M-position refs vs
the full 10M-token pass). Relative differences in whitened units where that makes sense.

    python experiments/phase0/compare_stats.py out/harvest-1.4b-interim out/harvest-1.4b
"""
import glob
import json
import sys

import numpy as np


def main(a, b):
    out = {}
    for fa in sorted(glob.glob(f"{a}/stats_iface*.npz")):
        k = fa.split("stats_")[-1][:-4]
        za, zb = np.load(fa), np.load(f"{b}/stats_{k}.npz")
        sd = np.sqrt(np.diag(zb["mc_cov"]))
        r = {
            "mean_rel": float(np.sqrt(np.mean(((za["mc_mean"] - zb["mc_mean"]) / sd) ** 2))),
            "cov_fro_rel": float(np.linalg.norm(za["mc_cov"] - zb["mc_cov"]) / np.linalg.norm(zb["mc_cov"])),
            "cf_rmse": float(np.sqrt(np.mean((za["cf_cf"] - zb["cf_cf"]) ** 2))),
            "lag1_rmse": float(np.sqrt(np.mean((za["lag1_rho"] - zb["lag1_rho"]) ** 2))),
            "median_rel": float(np.sqrt(np.mean(((za["q_values"][512] - zb["q_values"][512]) / sd) ** 2))),
            "n_a": int(za["mc_n"]), "n_b": int(zb["mc_n"]),
        }
        out[k] = r
        print(k, json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in r.items()}))
    json.dump(out, open(f"{b}/compare_{a.rstrip('/').split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
