"""The CF term must share the mimic optimum and must see under-dispersion."""
import torch

from lwd.stage.cf import CFDistance


def test_zero_at_the_mimic_optimum_and_positive_off_it():
    torch.manual_seed(0)
    cf = CFDistance(d=32, M=16, seed=1)
    t = torch.randn(4, 64, 32)
    assert float(cf(t.clone(), t)) < 1e-12
    assert float(cf(t + 0.5, t)) > 1e-3            # a shifted mean is seen
    assert float(cf(0.7 * t, t)) > 1e-3            # and so is under-dispersion


def test_under_dispersion_is_penalized_monotonically():
    """An MSE-trained student shrinks toward the conditional mean. The more it shrinks,
    the larger the distance has to be, or the term cannot pull against it."""
    torch.manual_seed(0)
    cf = CFDistance(d=32, M=32, seed=1)
    t = torch.randn(8, 128, 32)
    d = [float(cf(s * t, t)) for s in (1.0, 0.9, 0.8, 0.6)]
    assert d == sorted(d) and d[0] < 1e-12


def test_gradient_pushes_a_shrunk_student_back_out():
    torch.manual_seed(0)
    cf = CFDistance(d=16, M=32, seed=1)
    t = torch.randn(4, 256, 16)
    scale = torch.tensor(0.7, requires_grad=True)
    cf(scale * t, t).backward()
    assert scale.grad < 0                           # descending the loss increases the scale


def test_blind_to_a_permutation_of_positions():
    """It is a statement about the marginal, not about which position got which value.
    That is why it cannot replace the MSE and is only ever added to it."""
    torch.manual_seed(0)
    cf = CFDistance(d=16, M=16, seed=1)
    t = torch.randn(2, 64, 16)
    perm = t.reshape(-1, 16)[torch.randperm(128)].reshape(2, 64, 16)
    assert float(cf(perm, t)) < 1e-10
