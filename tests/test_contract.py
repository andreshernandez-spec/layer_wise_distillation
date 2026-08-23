import numpy as np
import torch

from lwd.contract.whiten import Affine, Contract, MarginalGaussianize
from lwd.noise.samplers import ContractGaussianized, Gaussian, Isotropic


def _heavy_tailed(n=20000, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    x[:, 0] = x[:, 0] ** 3 * 5        # heavy tail, massive activation flavour
    x[:, 1] = torch.exp(x[:, 1])      # skewed
    return x @ torch.randn(d, d, generator=g, dtype=torch.float64) + 3


def test_affine_zca_is_a_bijection_and_whitens():
    x = _heavy_tailed()
    cov = torch.from_numpy(np.cov(x.numpy().T, bias=True))
    a = Affine.zca(x.mean(0), cov, eps=0.0)
    z = a.forward(x)
    np.testing.assert_allclose(np.cov(z.numpy().T, bias=True), np.eye(6), atol=1e-8)
    np.testing.assert_allclose(a.inverse(z).numpy(), x.numpy(), rtol=1e-9, atol=1e-9)


def test_affine_pca_top_k_inverse_is_projection():
    x = _heavy_tailed()
    cov = torch.from_numpy(np.cov(x.numpy().T, bias=True))
    a = Affine.pca(x.mean(0), cov, d_s=3, eps=0.0)
    z = a.forward(x)
    assert z.shape[1] == 3
    xr = a.inverse(z)
    # projection: applying forward again gives the same z
    np.testing.assert_allclose(a.forward(xr).numpy(), z.numpy(), atol=1e-8)


def test_marginal_gaussianize_is_monotone_invertible_and_gaussian():
    x = _heavy_tailed()
    q = torch.linspace(0, 1, 1024, dtype=torch.float64)
    v = torch.quantile(x, q, dim=0)
    m = MarginalGaussianize(q, v)
    z = m.forward(x)
    # marginals ~ N(0,1): check a few quantiles of every channel
    for p, zp in ((0.1, -1.2816), (0.5, 0.0), (0.9, 1.2816)):
        assert (torch.quantile(z, p, dim=0) - zp).abs().max() < 0.05
    # monotone per channel
    xs, _ = x.sort(0)
    assert (m.forward(xs).diff(dim=0) >= 0).all()
    # invertible on the table's range
    xr = m.inverse(z)
    inside = (x > v[1]).all(1) & (x < v[-2]).all(1)
    assert (xr[inside] - x[inside]).abs().max() < 1e-6 * x.abs().max()


def test_contract_roundtrip_and_samplers_match_moments():
    x = _heavy_tailed()
    q = torch.linspace(0, 1, 1024, dtype=torch.float64)
    m = MarginalGaussianize(q, torch.quantile(x, q, dim=0))
    y = m.forward(x)
    a = Affine.zca(y.mean(0), torch.from_numpy(np.cov(y.numpy().T, bias=True)), eps=0.0)
    c = Contract(m, a)
    z = c.forward(x)
    np.testing.assert_allclose(z.mean(0).numpy(), 0, atol=1e-8)
    inside = (x > m.v[1]).all(1) & (x < m.v[-2]).all(1)
    assert (c.inverse(z)[inside] - x[inside]).abs().max() < 1e-5 * x.abs().max()

    g = torch.Generator().manual_seed(1)
    cov = torch.from_numpy(np.cov(x.numpy().T, bias=True))
    s = Gaussian(x.mean(0), cov).sample(64, 512, g).reshape(-1, 6).double()
    se = x.std(0) / np.sqrt(s.shape[0])  # Monte Carlo error of the sample mean
    assert ((s.mean(0) - x.mean(0)).abs() < 4 * se).all(), (s.mean(0) - x.mean(0)) / se
    cs = torch.from_numpy(np.cov(s.numpy().T, bias=True))
    assert ((cs - cov).abs() < 0.1 * cov.diag().max()).all()
    s2 = Isotropic(x.mean(0), cov).sample(8, 64, g)
    assert s2.shape == (8, 64, 6)
    s3 = ContractGaussianized(c).sample(64, 512, g).reshape(-1, 6).double()
    # marginal medians match the teacher's
    assert (torch.quantile(s3, 0.5, dim=0) - torch.quantile(x, 0.5, dim=0)).abs().max() < 0.2 * x.std(0).max()


def test_ar1_structure_has_the_requested_autocorrelation():
    from lwd.harvest.stats import Lag1
    x = _heavy_tailed()
    cov = torch.from_numpy(np.cov(x.numpy().T, bias=True))
    rho = torch.full((6,), 0.6)
    g = torch.Generator().manual_seed(2)
    s = Gaussian(x.mean(0), cov, structure="ar1", rho=rho).sample(16, 1024, g)
    # whiten back and measure lag-1
    from lwd.harvest.stats import zca_from_cov
    W, _, _, _ = zca_from_cov(cov, eps=0.0)
    z = ((s.double() - x.mean(0)) @ W).float()
    l = Lag1(6); l.update(z)
    assert (l.finalize()["rho"] - 0.6).abs().max() < 0.05
