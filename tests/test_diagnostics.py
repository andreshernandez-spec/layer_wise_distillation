import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


def test_attention_entropy_real_vs_isotropic_noise():
    from lwd.eval.diagnostics import attention_entropy_under
    from lwd.harvest.model import ResidentModel, StageRunner
    from lwd.harvest.stats import MeanCov
    from lwd.noise.samplers import Isotropic
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:4, :256].astype(np.int64))
    ifaces, _ = ResidentModel(MODEL, 3, torch.float32).forward(ids)
    X = ifaces[1]
    stage = StageRunner(MODEL, 2, 4, torch.float32, attn="eager")
    e_real = attention_entropy_under(stage, X)
    mc = MeanCov(512); mc.update(X); st = mc.finalize()
    g = torch.Generator().manual_seed(0)
    noise = Isotropic(st["mean"], st["cov"]).sample(4, 256, g)
    e_noise = attention_entropy_under(stage, noise)
    assert e_real.shape == (2, 8)
    # real text: far below uniform; isotropic noise: closer to it (the Exp 1b mechanism)
    assert e_real.mean() < 0.7 * math.log(256)
    assert e_noise.mean() > e_real.mean()


def test_jacobian_agreement_of_teacher_with_itself():
    from lwd.contract.whiten import Affine, Contract
    from lwd.eval.diagnostics import jacobian_agreement
    from lwd.harvest.model import ResidentModel, StageRunner
    from lwd.stage.student import StudentStage, student_config
    ids = torch.from_numpy(np.load("out/slice/anchor_rows.npy")[:2, :64].astype(np.int64))
    ifaces, _ = ResidentModel(MODEL, 3, torch.float32).forward(ids)
    X = ifaces[1]
    teacher = StageRunner(MODEL, 2, 4, torch.float32)
    d = 512
    g = torch.Generator().manual_seed(1)
    A = torch.randn(d, d, generator=g, dtype=torch.float64); W = A @ A.T / d + torch.eye(d, dtype=torch.float64)
    phi = Contract(None, Affine(torch.randn(d, generator=g, dtype=torch.float64), W, torch.linalg.inv(W))).to("cpu", torch.float32)

    class Wrapped(torch.nn.Module):
        def __init__(s): super().__init__(); s.t = teacher
        def forward(s, z):
            x = phi.inverse(z.reshape(-1, d)).reshape(z.shape)
            y = s.t(x)
            return phi.forward(y.reshape(-1, d)).reshape(y.shape)
    r = jacobian_agreement(Wrapped(), teacher, phi, phi, X, n_dirs=4)
    assert r["cos_mean"] > 0.999 and abs(r["norm_ratio"] - 1) < 1e-2, r
    st = StudentStage(student_config(MODEL, d, 1, 8), seed=0)
    r2 = jacobian_agreement(st, teacher, phi, phi, X, n_dirs=4)
    assert r2["cos_mean"] < 0.9, r2
