"""Stitching the teacher's own stage back in must reproduce the teacher's loss exactly;
stitching an untrained student must not."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


def test_stitch_identity_and_untrained():
    from transformers import GPTNeoXForCausalLM
    from lwd.contract.whiten import Affine, Contract
    from lwd.eval.stitch import next_token_loss, stitching_delta
    from lwd.harvest.model import StageRunner
    from lwd.stage.student import StudentStage, student_config
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:4, :129].astype(np.int64))
    m = GPTNeoXForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    d = 512
    eye = Contract(None, Affine(torch.zeros(d), torch.eye(d), torch.eye(d)))
    # the teacher's own stage 1 (blocks 2-4) as the "student": loss must match bitwise
    own = StageRunner(MODEL, 2, 4, torch.float32, "sdpa")
    r = stitching_delta(m, 2, 4, own, eye, eye, ids)
    assert r["delta"] == 0.0, r
    # a non-trivial contract pair around the same stage still telescopes (to float error)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(d, d, generator=g, dtype=torch.float64); W = A @ A.T / d + torch.eye(d, dtype=torch.float64)
    phi = Contract(None, Affine(torch.randn(d, generator=g, dtype=torch.float64), W, torch.linalg.inv(W))).to("cpu", torch.float32)

    class Wrapped(torch.nn.Module):  # phi_out o T o phi_in^-1, the transformed teacher stage
        def __init__(s): super().__init__(); s.t = own
        def forward(s, z):
            x = phi.inverse(z.reshape(-1, d)).reshape(z.shape)
            y = s.t(x)
            return phi.forward(y.reshape(-1, d)).reshape(y.shape)
    r2 = stitching_delta(m, 2, 4, Wrapped(), phi, phi, ids)
    assert abs(r2["delta"]) < 1e-3, r2
    # untrained student: clearly worse
    st = StudentStage(student_config(MODEL, d, 1, 8), seed=0)
    r3 = stitching_delta(m, 2, 4, st, eye, eye, ids)
    assert r3["delta"] > 0.5, r3
