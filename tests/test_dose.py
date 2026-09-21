"""The dose has to be the dose: exact ratio, m = 0 expressible, counts recorded."""
import pytest
import torch

from lwd.noise.samplers import DoseMix


class Flat:
    """Noise that is recognisably not an anchor."""
    def sample(self, b, L, g, device="cpu"):
        return torch.full((b, L, 4), -1.0)


ANCH = torch.ones(10, 16, 4)


@pytest.mark.parametrize("m", [0, 2, 8, 32, 5 / 3])
def test_the_long_run_ratio_is_the_requested_one(m):
    s, g = DoseMix(ANCH, Flat(), m), torch.Generator().manual_seed(0)
    real = noise = 0
    for _ in range(33 * 9 * 3):                      # a whole number of cycles for each m
        x, n_real = s.sample(8, 16, g)
        assert x.shape == (8, 16, 4)
        assert int((x[:, 0, 0] == 1).sum()) == n_real      # the real ones are the anchors
        real += n_real; noise += 8 - n_real
    if m == 0:
        assert noise == 0 and s.achieved_m() == 0
    else:
        assert abs(noise / real - m) < 1e-6 * max(1, m), (m, noise / real)
        assert abs(s.achieved_m() - m) < 1e-6 * max(1, m)


def test_m_zero_never_touches_the_noise_sampler():
    s = DoseMix(ANCH, None, 0)
    x, n_real = s.sample(8, 16, torch.Generator().manual_seed(0))
    assert n_real == 8 and bool((x == 1).all())


def test_a_dose_above_the_batch_size_still_delivers_real_sequences():
    """m = 32 with a batch of 8 has batches with no real sequence at all. They must not be
    every batch."""
    s, g = DoseMix(ANCH, Flat(), 32), torch.Generator().manual_seed(0)
    counts = [s.sample(8, 16, g)[1] for _ in range(33)]
    assert sum(counts) == 8 and max(counts) == 1
