"""Student stage trains on real activations of pythia-70m stage 1 (CPU, slow tier)."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow
MODEL = "EleutherAI/pythia-70m"


class RealSampler:
    def __init__(self, X):
        self.X = X

    def sample(self, b, L, g, device):
        idx = torch.randint(0, self.X.shape[0], (b,), generator=g)
        return self.X[idx, :L].to(device)


def test_student_learns_teacher_stage_on_real_activations():
    from lwd.harvest.model import ResidentModel, StageRunner, stage_bounds
    from lwd.stage.student import StudentStage, student_config
    from lwd.stage.train import TrainConfig, evaluate, train_stage
    from lwd.contract.whiten import Affine
    from lwd.harvest.stats import MeanCov
    p = "out/slice/anchor_rows.npy"
    rows = np.load(p)[:24, :64].astype(np.int64)
    ifaces, _ = ResidentModel(MODEL, 3, torch.float32).forward(torch.from_numpy(rows))
    X = ifaces[1]                                   # interface 1, (24, 64, 512)
    teacher = StageRunner(MODEL, 2, 4, torch.float32, "sdpa")
    mc = MeanCov(512); mc.update(X); st = mc.finalize()
    phi = Affine.zca(st["mean"], st["cov_shrunk"]).to("cpu", torch.float32)
    cfg = student_config(MODEL, d_s=512, n_layers=1, n_heads=8)
    student = StudentStage(cfg, seed=0)
    tc = TrainConfig(steps=60, batch=4, seq_len=64, lr=2e-3, warmup=5, eval_every=20, log_every=100, amp=False)
    e0 = evaluate(student, teacher, X[16:], phi, phi, 4, "cpu")
    hist = train_stage(student, teacher, RealSampler(X[:16]), tc, phi, phi, eval_X=X[16:], log=lambda r: None)
    e1 = hist[-1]["eval"]
    assert e1 < 0.7 * e0, (e0, e1)
    assert hist[-1]["positions"] == 60 * 4 * 64
