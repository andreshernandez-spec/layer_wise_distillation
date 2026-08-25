import math

import numpy as np
import pytest
import torch

from lwd.harvest.stats import (CFSketch, Lag1, MeanCov, Quantiles, cf_distance,
                               zca_from_cov)


def test_meancov_matches_numpy():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5000, 8, generator=g, dtype=torch.float64) @ torch.randn(8, 8, generator=g, dtype=torch.float64)
    mc = MeanCov(8)
    for chunk in x.split(700):
        mc.update(chunk)
    r = mc.finalize(shrink=0.0)
    np.testing.assert_allclose(r["mean"].numpy(), x.numpy().mean(0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["cov"].numpy(), np.cov(x.numpy().T, bias=True), rtol=1e-8, atol=1e-8)


def test_zca_whitens():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(6, 6, generator=g, dtype=torch.float64)
    cov = A @ A.T + 0.1 * torch.eye(6, dtype=torch.float64)
    W, Winv, lam, U = zca_from_cov(cov, eps=0.0)
    np.testing.assert_allclose((W @ cov @ W.T).numpy(), np.eye(6), atol=1e-9)
    np.testing.assert_allclose((W @ Winv).numpy(), np.eye(6), atol=1e-9)


def test_cf_sketch_of_gaussian_is_exp_minus_half_t2():
    g = torch.Generator().manual_seed(2)
    d = 16
    cf = CFSketch(d, M=64, K=8, seed=0)
    for _ in range(20):
        cf.update(torch.randn(5000, d, generator=g, dtype=torch.float64))
    r = cf.finalize()
    expect = torch.exp(-0.5 * r["t"] ** 2)  # (K,)
    # cos part ~ exp(-t^2/2), sin part ~ 0, Monte Carlo error ~ 1/sqrt(1e5)
    assert (r["cf"][..., 0] - expect[None, :]).abs().max() < 0.02
    assert r["cf"][..., 1].abs().max() < 0.02
    assert cf_distance(r["cf"], torch.stack([expect.expand(64, -1), torch.zeros(64, 8)], -1)) < 1e-4


def test_lag1_recovers_ar1_rho():
    g = torch.Generator().manual_seed(3)
    rho, L, d = 0.7, 4096, 4
    x = torch.zeros(8, L, d, dtype=torch.float64)
    e = torch.randn(8, L, d, generator=g, dtype=torch.float64)
    for t in range(1, L):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    lag = Lag1(d)
    lag.update(x)
    assert (lag.finalize()["rho"] - rho).abs().max() < 0.03


def test_quantiles_of_normal():
    g = torch.Generator().manual_seed(4)
    q = Quantiles(3, levels=11, reservoir=50000)
    for _ in range(4):
        q.update(torch.randn(20000, 3, generator=g))
    r = q.finalize()
    med = r["values"][5]
    assert med.abs().max() < 0.03
    assert (r["values"][-2] - 1.2816).abs().max() < 0.05  # 90th percentile
