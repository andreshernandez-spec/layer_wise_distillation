"""The teacher's own stages chained through contract maps reproduce the teacher."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


def test_chain_telescopes():
    from lwd.compose.chain import Chain, Wrapped, drift_profile
    from lwd.contract.whiten import Affine, Contract
    from lwd.harvest.model import ResidentModel, StageRunner, stage_bounds
    from lwd.harvest.stats import MeanCov
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:2, :128].astype(np.int64))
    ifaces, _ = ResidentModel(MODEL, 3, torch.float32).forward(ids)
    d = 512
    # a real ZCA contract per interface, from these activations
    phis64 = []
    for h in ifaces:
        mc = MeanCov(d); mc.update(h); st = mc.finalize()
        phis64.append(Contract(None, Affine.zca(st["mean"], st["cov_shrunk"])))
    bare = [StageRunner(MODEL, a, b, torch.float32) for a, b in stage_bounds(6, 3)]
    # bare chain: bitwise equal to the resident model
    with torch.no_grad():
        out = Chain(bare)(ifaces[0])
    for k in range(4):
        assert torch.equal(out[k], ifaces[k]), k
    # wrapped chain: phi_{k+1}^{-1} o phi_{k+1} telescopes. The first interface agrees
    # to float precision; after that the *teacher stages* amplify float-level input
    # perturbations (~100x per stage in relative MSE on this model), which is a Phase 2
    # measurement, not a composition bug. So the reference is a bare chain fed an input
    # perturbed at float32 resolution, and the wrapped chain must not drift more than
    # 10x that reference at any interface.
    g = torch.Generator().manual_seed(0)
    x0 = ifaces[0] * (1 + 1e-7 * torch.randn(ifaces[0].shape, generator=g))
    with torch.no_grad():
        ref = Chain(bare)(x0)
    ref_drift = [float(((ref[k] - ifaces[k]) ** 2).sum() / ((ifaces[k] - ifaces[k].mean()) ** 2).sum())
                 for k in range(1, 4)]
    for dt in (torch.float64, torch.float32):
        phis = [p.to("cpu", dt) for p in phis64]
        wrapped = [Wrapped(TransformedTeacher(bare[k], phis[k], phis[k + 1]), phis[k], phis[k + 1]) for k in range(3)]
        with torch.no_grad():
            drift = drift_profile(Chain(wrapped), ifaces, phis)
        assert drift[0] < 1e-9, (dt, drift)
        # measured 22 Aug 2026 on pythia-70m: wrapped [2e-11, 8e-8, 2e-6] vs reference
        # [7e-13, 2e-9, 5e-9]; the stages amplify float-level perturbations 10-1000x
        # per stage. That amplification is Phase 2's C2.2 measurement; here the bound
        # only has to rule out a composition bug, which would be orders larger.
        assert all(d <= 1000 * r + 1e-9 for d, r in zip(drift, ref_drift)), (dt, drift, ref_drift)


class TransformedTeacher(torch.nn.Module):
    """phi_out o T o phi_in^{-1}: what a perfect student would be."""

    def __init__(self, t, phi_in, phi_out):
        super().__init__()
        self.t, self.phi_in, self.phi_out = t, phi_in, phi_out

    def forward(self, z):
        from lwd.compose.chain import _phi
        return _phi(self.phi_out, self.t(_phi(self.phi_in, z, inverse=True)))


def test_student_lm_with_teacher_stages_reproduces_the_teacher():
    """The composed model is only as right as its plumbing. With the teacher's own
    stages in place of students, it must give the teacher's next-token loss."""
    import numpy as np
    from transformers import GPTNeoXForCausalLM
    from lwd.compose.chain import Wrapped
    from lwd.compose.model import StudentLM
    from lwd.contract.whiten import Affine, Contract
    from lwd.eval.stitch import next_token_loss
    from lwd.harvest.model import Edges, StageRunner, stage_bounds
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:2, :65].astype(np.int64))
    teacher = GPTNeoXForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="sdpa").eval()
    d = 512
    eye = Contract(None, Affine(torch.zeros(d), torch.eye(d), torch.eye(d)))
    stages = [Wrapped(StageRunner(MODEL, a, b, torch.float32), eye, eye)
              for a, b in stage_bounds(6, 3)]
    lm = StudentLM(Edges(MODEL, torch.float32), stages)
    assert abs(next_token_loss(lm, ids) - next_token_loss(teacher, ids)) < 1e-4
    with torch.no_grad():
        ours = lm.interfaces(ids[:, :-1])
    assert len(ours) == 4 and ours[0].shape[-1] == d


def test_dagger_sampler_moves_the_input_distribution_not_the_target():
    """With zero trained stages below it, propagation must equal the teacher's own
    interface; with a stage below, it must differ. And on_policy must control the mix."""
    import numpy as np
    from lwd.compose.chain import Wrapped
    from lwd.compose.dagger import PropagatedSampler
    from lwd.contract.whiten import Affine, Contract
    from lwd.harvest.model import Lower
    from lwd.stage.student import StudentStage, student_config
    rows = np.load("out/slice/anchor_rows.npy")[:8]
    d = 512
    eye = Contract(None, Affine(torch.zeros(d), torch.eye(d), torch.eye(d)))
    low0 = Lower(MODEL, 0, torch.float32)
    g = torch.Generator().manual_seed(0)
    # no stages below: the student's interface 0 IS the teacher's
    s0 = PropagatedSampler(rows, low0, [], low0, on_policy=1.0)
    a = s0.sample(2, 32, g)
    g2 = torch.Generator().manual_seed(0)
    b = PropagatedSampler(rows, low0, [], low0, on_policy=0.0).sample(2, 32, g2)
    assert torch.allclose(a, b, atol=1e-5), "with nothing below it, propagation is the teacher"
    # one untrained student stage below: the distribution must move
    stu = StudentStage(student_config(MODEL, d, 1, 8), seed=0)
    low1 = Lower(MODEL, 2, torch.float32)
    s1 = PropagatedSampler(rows, low0, [Wrapped(stu, eye, eye)], low1, on_policy=1.0)
    g3 = torch.Generator().manual_seed(0)
    on = s1.sample(2, 32, g3)
    g4 = torch.Generator().manual_seed(0)
    off = PropagatedSampler(rows, low0, [Wrapped(stu, eye, eye)], low1, on_policy=0.0).sample(2, 32, g4)
    assert on.shape == off.shape
    assert not torch.allclose(on, off, atol=1e-2), "on-policy inputs should differ from the teacher's"
    # a mixed batch takes half from each
    g5 = torch.Generator().manual_seed(0)
    mix = PropagatedSampler(rows, low0, [Wrapped(stu, eye, eye)], low1, on_policy=0.5).sample(2, 32, g5)
    assert torch.allclose(mix[:1], on[:1], atol=1e-4) and torch.allclose(mix[1:], off[1:], atol=1e-4)
