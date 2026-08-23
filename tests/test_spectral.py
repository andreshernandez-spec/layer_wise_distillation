import numpy as np

from lwd.eval.spectral import alpha_hill_ks, stable_rank


def test_alpha_recovers_power_law_tail():
    rng = np.random.default_rng(0)
    for a_true in (2.0, 3.0, 5.0):
        # Pareto tail with exponent a: P(x > t) ~ t^(1 - a)
        x = rng.pareto(a_true - 1, size=4000) + 1.0
        a, ks, n = alpha_hill_ks(np.sort(x), min_tail=200)
        assert abs(a - a_true) < 0.3 * a_true, (a_true, a, ks, n)
        assert ks < 0.05


def test_stable_rank():
    lam = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
    assert stable_rank(lam) == 2.0


def test_non_finite_weights_are_reported_not_raised():
    """A NaN-poisoned checkpoint used to surface as LAPACK's 'SVD did not converge',
    which says nothing about the real problem."""
    import torch
    from lwd.eval.spectral import esd, metrics_for
    w = torch.randn(300, 300)
    assert esd(w) is not None
    w[5, 7] = float("nan")
    assert esd(w) is None
    m = metrics_for({"good": torch.randn(300, 300), "bad": w})
    assert "alpha" in m["good"] and m["bad"].get("non_finite") is True
