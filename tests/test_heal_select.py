"""The heal under a data budget: recycle the store, stop on validation, keep the best."""
import numpy as np
import torch

from lwd.heal.select import HealSelectConfig, heal_selected
from lwd.heal.train import TopKStore


class Out:
    def __init__(self, logits):
        self.logits = logits


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(11, 6)
        self.out = torch.nn.Linear(6, 11)
        self.frozen = torch.nn.Linear(2, 2)
        for p in self.frozen.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids=None, **kw):
        return Out(self.out(self.emb(input_ids)))


def store(tmp_path, n_files=3):
    g = np.random.default_rng(0)
    for i in range(n_files):
        np.savez(tmp_path / f"c{i}.npz", tokens=g.integers(0, 11, (2, 8)).astype(np.int64),
                 ids=g.integers(0, 11, (2, 8, 4)).astype(np.uint16),
                 logp=np.log(np.full((2, 8, 4), 0.2, dtype=np.float32)).astype(np.float16),
                 lse=np.zeros((2, 8), np.float32))
    return TopKStore(str(tmp_path))


CFG = dict(cap_tokens=200 * 8, batch=1, seq_len=8, lr=1e-2, warmup=5, val_every=10, patience=2,
           cooldown_frac=0.5, min_cooldown=3, amp=False, log_every=10**9)


def test_recycles_the_store_stops_on_patience_and_reports_epochs(tmp_path):
    s = store(tmp_path)                                   # 6 sequences of 8 tokens: 48 per epoch
    vals = iter([1.0, 0.8, 0.6, 0.7, 0.9, 0.5])
    _, r = heal_selected(Tiny(), s, HealSelectConfig(**CFG), lambda m: next(vals), log=lambda x: None)
    assert r["stop"] == "patience" and r["best_step"] == 30 and r["selected"] == "cooled"
    assert r["stable_steps"] == 50 and r["cooldown_steps"] == 15
    assert r["tokens_seen"] == 65 * 8 and abs(r["epochs"] - 65 * 8 / 48) < 1e-9
    assert r["tokens_to_best"] == 30 * 8                  # more than six epochs to the best:
    assert r["epochs_to_best"] == 5.0                     # the store really was recycled


def test_restores_exactly_the_best_weights_and_leaves_frozen_ones_alone(tmp_path):
    s, model, seen = store(tmp_path), Tiny(), []
    frozen = model.frozen.weight.clone()

    def val(m):
        seen.append({k: v.clone() for k, v in m.state_dict().items()})
        return [1.0, 0.6, 0.7, 0.9, 0.95][len(seen) - 1]

    _, r = heal_selected(model, s, HealSelectConfig(**CFG), val, log=lambda x: None)
    assert r["selected"] == "stable-best" and r["best_step"] == 20
    for k, v in model.state_dict().items():
        assert torch.equal(v, seen[1][k]), k
    assert torch.equal(model.frozen.weight, frozen)
