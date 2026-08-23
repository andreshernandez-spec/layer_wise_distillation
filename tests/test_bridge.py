import torch

from lwd.harvest.bridge import bridge_floor, rel_mse, ridge_fit
from lwd.harvest.stats import zca_from_cov


def test_bridge_floor_on_a_linear_map():
    """Y = X W with W acting only on the top-k covariance directions: the top-k probe is
    exact and the floor is zero; with W acting on all directions the top-k probe loses
    exactly the variance it cannot see."""
    g = torch.Generator().manual_seed(0)
    n, d = 20000, 32
    scales = torch.linspace(4, 0.25, d, dtype=torch.float64)
    X = torch.randn(n, d, generator=g, dtype=torch.float64) * scales
    cov = torch.from_numpy(__import__("numpy").cov(X.numpy().T, bias=True))
    W, _, lam, U = zca_from_cov(cov, eps=0.0)
    mean = X.mean(0)
    # map that only reads the 8 highest-variance directions
    order = torch.argsort(lam, descending=True)
    P = U[:, order[:8]]
    Wmap = P @ torch.randn(8, d, generator=g, dtype=torch.float64)
    Y = X @ Wmap
    r = bridge_floor(X[4000:], Y[4000:], X[:4000], Y[:4000], W, mean, lam, 8, lam=1e-8, U_in=U)
    assert r["full"] < 1e-3 and r["top8"] < 1e-3 and abs(r["floor"]) < 1e-3, r
    # map reading everything: top-8 misses the rest
    Y2 = X @ torch.randn(d, d, generator=g, dtype=torch.float64)
    r2 = bridge_floor(X[4000:], Y2[4000:], X[:4000], Y2[:4000], W, mean, lam, 8, lam=1e-8, U_in=U)
    assert r2["full"] < 1e-3 and r2["floor"] > 0.1, r2
