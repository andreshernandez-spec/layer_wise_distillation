"""Resident vs streaming must agree bitwise (docs/01 G0.1). Uses pythia-70m on CPU."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 50000, (2, 64), generator=g)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@torch.no_grad()
def test_resident_equals_streaming_bitwise(ids, attn):
    from lwd.harvest.model import Edges, ResidentModel, StageRunner, stage_bounds
    dt = torch.float32
    res = ResidentModel(MODEL, n_stages=3, dtype=dt, attn=attn)
    ifaces, logits = res.forward(ids)
    edges = Edges(MODEL, dt)
    h = edges.embed(ids)
    assert torch.equal(h, ifaces[0])
    for k, (a, b) in enumerate(stage_bounds(6, 3)):
        h = StageRunner(MODEL, a, b, dt, attn)(h)
        assert torch.equal(h, ifaces[k + 1]), f"interface {k + 1} differs ({attn})"
    assert torch.equal(edges.head(h), logits)


def test_topk_coverage():
    """On real text the top-64 holds most of the mass; on random ids it need not."""
    from pathlib import Path
    from lwd.harvest.logits import topk_coverage, topk_pack
    from lwd.harvest.model import ResidentModel
    p = Path("out/slice/anchor_rows.npy")
    if not p.exists():
        pytest.skip("slice not fetched")
    ids = torch.from_numpy(np.load(p)[:2, :128].astype(np.int64))
    _, logits = ResidentModel(MODEL, 3, torch.float32).forward(ids)
    i, logp, lse = topk_pack(logits, 64)
    cov = topk_coverage(logp)
    assert cov.max() <= 1.0 + 1e-3 and cov.mean() > 0.8, (cov.min(), cov.mean(), cov.max())
    assert i.dtype == torch.uint16
    # the stored log-probs reproduce the top-1 log-prob to fp16 precision
    exact = torch.log_softmax(logits.float(), -1).max(-1).values
    assert (logp[..., 0].float() - exact).abs().max() < 2e-3
