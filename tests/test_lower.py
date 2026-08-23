"""Lower (embedding + blocks [0, a)) must equal the resident model's interface a bitwise."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


def test_lower_matches_resident():
    from lwd.harvest.model import Lower, ResidentModel
    from lwd.noise.samplers import LiveReal
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:3, :64].astype(np.int64))
    ifaces, _ = ResidentModel(MODEL, 3, torch.float32).forward(ids)
    for a, k in ((0, 0), (2, 1), (4, 2)):
        assert torch.equal(Lower(MODEL, a, torch.float32)(ids), ifaces[k]), (a, k)
    rows = np.load("out/slice/anchor_rows.npy")[:8]
    s = LiveReal(rows, Lower(MODEL, 2, torch.float32), exclude={0, 1})
    assert len(s.rows) == 6
    g = torch.Generator().manual_seed(0)
    x = s.sample(2, 32, g)
    assert x.shape == (2, 32, 512) and x.dtype == torch.float32
