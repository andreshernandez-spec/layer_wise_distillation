"""Bridge oracle (docs/01 C0.5): how much of interface k+1 is linearly predictable from
interface k, from the full input and from its top-d_S ZCA directions. The gap is the
floor a d_S-wide student inherits from the projection alone."""
from __future__ import annotations

import torch


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3):
    """X (n, p), Y (n, q) float64, centered by caller. Returns W (p, q)."""
    p = X.shape[1]
    G = X.T @ X + lam * X.shape[0] * torch.eye(p, dtype=X.dtype, device=X.device)
    return torch.linalg.solve(G, X.T @ Y)


def rel_mse(Y: torch.Tensor, Yhat: torch.Tensor) -> float:
    return float(((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def bridge_floor(Xtr, Ytr, Xte, Yte, W_in, mean_in, lam_eig, d_s: int, lam: float = 1e-3,
                 U_in=None):
    """Relative test MSE of ridge from (a) the full input, (b) its top-d_s principal
    directions. Y predicted in raw coordinates.

    mean_in, lam_eig, U_in: from stats.zca_from_cov (W_in is accepted for signature
    compatibility and unused: selecting columns of ZCA output by eigenvalue is wrong,
    ZCA rotates back into the input basis). Directions are ranked by eigenvalue.
    """
    assert U_in is not None, "pass U_in=U from zca_from_cov"
    order = torch.argsort(lam_eig, descending=True)
    Ue = U_in[:, order].to(torch.float64)
    scale = 1 / torch.sqrt(lam_eig[order].clamp_min(1e-12)).to(torch.float64)

    def pca(X):
        return ((X.to(torch.float64) - mean_in) @ Ue) * scale

    Ztr, Zte = pca(Xtr), pca(Xte)
    Ytr, Yte = Ytr.to(torch.float64), Yte.to(torch.float64)
    ym = Ytr.mean(0)
    out = {}
    for name, cols in (("full", slice(None)), (f"top{d_s}", slice(0, d_s))):
        A, B = Ztr[:, cols], Zte[:, cols]
        Wr = ridge_fit(A, Ytr - ym, lam)
        out[name] = rel_mse(Yte, B @ Wr + ym)
    out["floor"] = out[f"top{d_s}"] - out["full"]
    return out
