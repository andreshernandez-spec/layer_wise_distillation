"""The heal under a data budget: recycle the store, choose the length on validation,
rewind, anneal to it. The rule itself is tested in test_select.py; this is the wiring."""
import numpy as np
import torch

from lwd.heal.select import HealSelectConfig, _pack, _unpack, heal_selected
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


CFG = dict(cap_tokens=400 * 8, batch=1, seq_len=8, lr=1e-2, warmup=5, val_every=10, patience=3,
           cooldown_frac=0.5, min_cooldown=10, amp=False, log_every=10**9)
CURVE = [1.0, 0.8, 0.6, 0.62, 0.61, 1.5, 0.3, 1.6, 0.55]      # eight stable, then the annealed one


def test_recycles_the_store_chooses_the_plateau_and_reports_epochs(tmp_path):
    s = store(tmp_path)                                   # 6 sequences of 8 tokens: 48 per epoch
    vals = iter(CURVE)
    _, r = heal_selected(Tiny(), s, HealSelectConfig(**CFG), lambda m: next(vals), log=lambda x: None)
    assert (r["stop"], r["best_step"], r["rewound_to"], r["cooldown_steps"]) == ("patience", 40, 20, 20)
    assert r["selected_val"] == 0.55 and r["stable_steps"] == 80
    assert r["tokens_seen"] == 100 * 8 and abs(r["epochs"] - 800 / 48) < 1e-9
    assert r["tokens_to_best"] == 320 and abs(r["epochs_to_best"] - 320 / 48) < 1e-9   # recycled


def test_frozen_weights_are_left_alone_and_the_rewind_is_real(tmp_path):
    s, model, seen = store(tmp_path), Tiny(), []
    frozen = model.frozen.weight.clone()

    def val(m):
        seen.append(m.out.weight.detach().clone())
        return CURVE[len(seen) - 1]

    heal_selected(model, s, HealSelectConfig(**CFG), val, log=lambda x: None)
    assert torch.equal(model.frozen.weight, frozen)
    # the annealed model descends from the step-20 weights, so it is neither those nor the
    # step-80 ones the stable phase ended on
    assert not torch.equal(seen[-1], seen[1]) and not torch.equal(seen[-1], seen[7])


def test_a_packed_optimizer_state_restores_the_moments_to_three_digits():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(); lin(torch.randn(8, 16)).pow(2).mean().backward(); opt.step()
    sd = opt.state_dict()
    back = _unpack(_pack(sd))
    for k, st in sd["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            a, b = st[name], back["state"][k][name]
            assert b.dtype == torch.float32
            assert torch.allclose(a, b, rtol=2e-2, atol=1e-12)
        assert back["state"][k]["step"] == st["step"]                 # the scalar is untouched
    opt.load_state_dict(back)                                          # and it loads
