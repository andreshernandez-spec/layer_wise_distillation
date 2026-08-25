"""The heal loss must be zero exactly when the student matches the teacher."""
import argparse
import math
from pathlib import Path

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


def test_a_diverged_heal_aborts_instead_of_burning_its_budget():
    """The guard stops non-finite weights getting worse; it cannot repair them. Without
    an abort a diverged run skips the rest of its budget and reports a NaN loss after
    spending every token."""
    import torch

    from lwd.heal.train import HealConfig, heal

    class Poison:
        def batches(self, batch, max_tokens, seed=0):
            for _ in range(500):
                yield (torch.zeros(1, 8, dtype=torch.long),
                       torch.zeros(1, 8, 4, dtype=torch.long),
                       torch.full((1, 8, 4), float("nan")))

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(4, 7)

        def forward(self, input_ids=None, **kw):
            return Out(self.w(torch.zeros(*input_ids.shape, 4)))

    with pytest.raises(RuntimeError, match="diverged"):
        heal(M(), Poison(), HealConfig(tokens=4000, batch=1, abort_after_skips=10, amp=False),
             device="cpu", log=lambda r: None)


def test_the_store_loops_and_a_short_run_is_refused(tmp_path):
    """A single pass over the files capped delivery at one epoch, so a run asking for
    17 epochs silently trained on one and reported success."""
    import numpy as np
    from lwd.heal.train import TopKStore
    for i in range(3):
        np.savez(tmp_path / f"c{i}.npz", tokens=np.zeros((2, 8), dtype=np.int64),
                 ids=np.zeros((2, 8, 4), dtype=np.uint16),
                 logp=np.full((2, 8, 4), -1.0, dtype=np.float16), lse=np.zeros((2, 8), np.float32))
    s = TopKStore(str(tmp_path))
    assert s.epoch_tokens() == 3 * 16
    got = sum(t.numel() for t, _, _ in s.batches(batch=1, max_tokens=500))
    assert got >= 500, f"store stopped early at {got}"          # loops, not one pass
    got2 = sum(t.numel() for t, _, _ in s.batches(batch=1, max_tokens=16))
    assert 16 <= got2 <= 24                                      # and still stops on time


def test_a_result_file_is_never_silently_overwritten(tmp_path, monkeypatch):
    """dagger_stack.py promotes stage checkpoints in place, so the same heal command
    loads a different model afterwards. It overwrote two cells of the C2.4 curve."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "phase2"))
    import heal as heal_cli

    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"out: {tmp_path}\nn_stages: 1\nmodel: m\ndevice: cpu\ndtype: float32\n"
                   f"harvest: {tmp_path}\nheldout: {tmp_path}/h.npy\nstudent_layers: 1\n"
                   f"student_heads: 1\ntrain: {{seq_len: 4}}\n")
    (tmp_path / "heal_random_C_t1e06_s0.json").write_text("{}")

    a = argparse.Namespace(config=str(cfg), init="random", measure="C", q=1e8, tokens=1e6,
                           seed=0, heal_seed=None, eval_rows=1, lr=0.0, tag="", overwrite=False,
                           stack_suffix="")
    with pytest.raises(SystemExit, match="exists"):
        heal_cli.main(a)
    a.tag = "_dagger"                       # a tag is the way past it, not a flag to add
    with pytest.raises(Exception) as e:     # gets past the guard, then fails on the model
        heal_cli.main(a)
    assert "exists" not in str(e.value)
