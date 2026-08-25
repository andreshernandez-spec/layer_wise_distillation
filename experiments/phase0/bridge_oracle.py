"""C0.5: bridge floor per interface and candidate width.

Streams the fp16 refs, keeps at most `max_pos` positions (a fixed random subset of each
file) in float32, regenerates the next interface with the stage runner in batches, and
solves the ridge from f64 Gram matrices accumulated in chunks. Peak host memory is
about 2 x max_pos x d x 4 bytes (3.3 GB at the default 200k, d=2048).

    python experiments/phase0/bridge_oracle.py out/harvest-1.4b --widths 1024 512
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from lwd.harvest.model import StageRunner, stage_bounds
from lwd.harvest.stats import zca_from_cov

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def gram(Z, Y, chunk=16384):
    """Z (n, p), Y (n, q) float32 -> (Z^T Z, Z^T Y, sum Y, sum Y^2) in f64."""
    p, q = Z.shape[1], Y.shape[1]
    G = torch.zeros(p, p, dtype=torch.float64); C = torch.zeros(p, q, dtype=torch.float64)
    for i in range(0, Z.shape[0], chunk):
        z, y = Z[i:i + chunk].double(), Y[i:i + chunk].double()
        G += z.T @ z; C += z.T @ y
    return G, C


@torch.no_grad()
def main(out, widths, max_pos, test_frac, device):
    out = Path(out)
    cfg = json.load(open(out / "run.json"))["config"]
    S, dt = cfg["n_stages"], DT[cfg["dtype"]]
    n_layers = AutoConfig.from_pretrained(cfg["model"]).num_hidden_layers
    g = torch.Generator().manual_seed(0)
    report = {}
    for k, (a, b) in enumerate(stage_bounds(n_layers, S)):
        files = sorted((out / "anchors" / f"iface{k}").glob("ref_*.npy"))
        per_file = max(1, max_pos // len(files))
        runner = StageRunner(cfg["model"], a, b, dt, cfg["attn"], device)
        Xs, Ys = [], []
        for f in files:
            x = torch.from_numpy(np.load(f))  # (bf, L, d) fp16
            y = torch.cat([runner(x[i:i + 4].to(device).to(dt)).float().cpu() for i in range(0, x.shape[0], 4)])
            x = x.reshape(-1, x.shape[-1]); y = y.reshape(-1, y.shape[-1])
            idx = torch.randperm(x.shape[0], generator=g)[:per_file]
            Xs.append(x[idx].float()); Ys.append(y[idx])
        del runner; torch.cuda.empty_cache()
        X, Y = torch.cat(Xs), torch.cat(Ys); del Xs, Ys
        st = np.load(out / f"stats_iface{k}.npz")
        W, _, lam, U = zca_from_cov(torch.from_numpy(st["mc_cov_shrunk"]))
        mean = torch.from_numpy(st["mc_mean"]).float()
        order = torch.argsort(lam, descending=True)
        Ue = U[:, order].float(); scale = (1 / torch.sqrt(lam[order].clamp_min(1e-12))).float()
        Z = ((X - mean) @ Ue) * scale  # PCA-whitened, columns ranked by eigenvalue
        n = Z.shape[0]; nt = int(n * test_frac)
        Ztr, Zte, Ytr, Yte = Z[nt:], Z[:nt], Y[nt:], Y[:nt]
        ym = Ytr.mean(0); Ytr_c = Ytr - ym
        den = float(((Yte.double() - Yte.double().mean(0)) ** 2).sum())
        res = {}
        for name, cols in [("full", slice(None))] + [(f"top{w}", slice(0, w)) for w in widths]:
            A = Ztr[:, cols]
            G, C = gram(A, Ytr_c)
            lam_r = 1e-3 * A.shape[0]
            Wr = torch.linalg.solve(G + lam_r * torch.eye(G.shape[0], dtype=torch.float64), C)
            pred = Zte[:, cols].double() @ Wr + ym.double()
            res[name] = float(((Yte.double() - pred) ** 2).sum() / den)
        for w in widths:
            res[f"floor{w}"] = res[f"top{w}"] - res["full"]
        res["n_train"], res["n_test"] = int(n - nt), int(nt)
        report[f"iface{k}"] = res
        print(json.dumps({f"iface{k}": {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res.items()}}), flush=True)
        del X, Y, Z
    json.dump(report, open(out / "bridge_oracle.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--widths", type=int, nargs="+", default=[1024])
    p.add_argument("--max-pos", type=int, default=200_000); p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.out, a.widths, a.max_pos, a.test_frac, a.device)
