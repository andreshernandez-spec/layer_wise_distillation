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


def test_the_composed_model_passes_gradients_to_its_stages():
    """The heal back-propagates through the frozen head into the stages. An
    @torch.no_grad() on the teacher's edges made that silently impossible."""
    import numpy as np
    from lwd.compose.chain import Wrapped
    from lwd.compose.model import StudentLM
    from lwd.contract.whiten import Affine, Contract
    from lwd.harvest.model import Edges
    from lwd.heal.train import kd_and_ce
    from lwd.stage.student import StudentStage, student_config
    MODEL, d = "EleutherAI/pythia-70m", 512
    eye = Contract(None, Affine(torch.zeros(d), torch.eye(d), torch.eye(d)))
    stu = StudentStage(student_config(MODEL, d, 1, 8), seed=0)
    edges = Edges(MODEL, torch.float32)
    for p in edges.parameters():
        p.requires_grad_(False)
    lm = StudentLM(edges, [Wrapped(stu, eye, eye)])
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:1, :32].astype(np.int64))
    logits = lm(input_ids=ids).logits
    assert logits.requires_grad, "no graph: the heal would have no gradients"
    tl = logits.detach().log_softmax(-1)
    vals, tid = tl.topk(16, dim=-1)
    kd, ce = kd_and_ce(logits, tid, vals, ids)
    (kd + ce).backward()
    grads = [p.grad for p in stu.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), "stages got no gradient"
    assert all(p.grad is None for p in edges.parameters()), "frozen edges got gradients"


def test_an_empty_topk_store_fails_loudly(tmp_path):
    """Silence here is indistinguishable from success: heal() would run zero steps and
    report loss_before == loss_after, which looks like a converged model."""
    from lwd.heal.train import TopKStore
    with pytest.raises(FileNotFoundError, match="no top-k chunks"):
        TopKStore(str(tmp_path / "empty"))
