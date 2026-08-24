"""The heal loss must be zero exactly when the student matches the teacher."""
import math

import numpy as np
import pytest
import torch

from lwd.heal.train import kd_and_ce

pytestmark = pytest.mark.slow


def test_kd_is_zero_iff_the_student_matches_the_teacher():
    g = torch.Generator().manual_seed(0)
    b, L, V, k = 2, 16, 500, 64
    teacher_logits = torch.randn(b, L, V, generator=g) * 2
    tlogp = teacher_logits.log_softmax(-1)
    vals, ids = tlogp.topk(k, dim=-1)
    tokens = torch.randint(0, V, (b, L), generator=g)
    # the teacher against itself: KL is zero
    kd, ce = kd_and_ce(teacher_logits, ids, vals, tokens)
    assert abs(float(kd)) < 1e-5, float(kd)
    # a different student: KL is positive
    kd2, _ = kd_and_ce(torch.randn(b, L, V, generator=g), ids, vals, tokens)
    assert float(kd2) > 0.5, float(kd2)
    # KL is invariant to a constant shift of the student's logits (softmax invariance)
    kd3, _ = kd_and_ce(teacher_logits + 7.0, ids, vals, tokens)
    assert abs(float(kd3) - float(kd)) < 1e-5
    # and the CE matches torch's own on the same inputs
    import torch.nn.functional as F
    ref = F.cross_entropy(teacher_logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
    assert abs(float(ce) - float(ref)) < 1e-5


def test_store_yields_aligned_batches():
    from pathlib import Path
    from lwd.heal.train import TopKStore
    p = Path("out/harvest-1.4b/topk")
    if not p.exists():
        pytest.skip("no top-k store")
    s = TopKStore(str(p))
    got = 0
    for tok, ids, logp in s.batches(batch=2, max_tokens=8192, seed=0):
        assert tok.shape[0] == ids.shape[0] == logp.shape[0]
        assert tok.shape[1] == ids.shape[1] and ids.shape[2] == 64
        # the stored log-probs are a real distribution's top-k: mass under 1, sorted
        p_top = logp.float().exp().sum(-1)
        assert (p_top <= 1.0 + 1e-3).all() and (p_top > 0.2).all()
        assert (logp[..., :-1] >= logp[..., 1:] - 1e-3).all()
        got += tok.numel()
    assert got >= 8192
