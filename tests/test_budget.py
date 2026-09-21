"""The data budget: nested, split once, and nothing outside the training rows leaks in."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lwd.harvest.budget import split_budget

ROOT = Path(__file__).resolve().parents[1]


def rows(n, L=33, seed=0):
    return np.random.default_rng(seed).integers(0, 50000, (n, L)).astype(np.uint16)


def test_budgets_are_nested_prefixes_and_the_split_is_disjoint():
    r = rows(500)
    t1, v1, m1 = split_budget(r, 49)
    t2, v2, m2 = split_budget(r, 488)
    assert len(v1) == 5 and len(t1) == 44 and len(v2) == 49 and len(t2) == 439
    assert np.array_equal(t1, r[:44]) and np.array_equal(v1, r[44:49])
    assert np.array_equal(t2[:44], t1)                       # the small budget's training rows
    assert m1["budget_tokens"] == 49 * 32 and m1["sha256_train"] != m1["sha256_val"]
    assert m1["sha256_budget"] != m2["sha256_budget"]


def test_a_tiny_budget_still_gets_a_usable_validation_split():
    t, v, _ = split_budget(rows(100), 20)
    assert len(v) == 4 and len(t) == 16                      # 10% would be 2: too few to select on
    with pytest.raises(AssertionError):
        split_budget(rows(100), 4)


@pytest.mark.slow
def test_the_harvest_takes_statistics_from_the_training_rows_only(tmp_path):
    """pythia-70m on CPU. 12 rows of budget: 8 training, 4 validation."""
    np.save(tmp_path / "rows.npy", rows(40, L=33))
    cfg = tmp_path / "h.yaml"
    cfg.write_text(f"""model: EleutherAI/pythia-70m
n_stages: 3
mode: resident
attn: sdpa
dtype: float32
device: cpu
slice: {tmp_path}/rows.npy
batch: 4
seq_len: 32
anchor_seqs: 2
ref_ifaces: [1]
topk: 8
seed: 0
budget_rows: 12
out: {tmp_path}/out
""")
    r = subprocess.run([sys.executable, "experiments/phase0/run.py", str(cfg)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / "out"
    meta = json.load(open(out / "run.json"))["budget"]
    assert (meta["train_rows"], meta["val_rows"]) == (8, 4)
    st = np.load(out / "stats_iface1.npz")
    assert int(st["mc_n"]) == 8 * 32                         # validation positions are not in it
    refs = np.concatenate([np.load(f) for f in sorted((out / "anchors" / "iface1").glob("ref_*.npy"))])
    val = np.concatenate([np.load(f) for f in sorted((out / "anchors_val" / "iface1").glob("ref_*.npy"))])
    assert refs.shape[:2] == (8, 32) and val.shape[:2] == (4, 32)   # every training row is an anchor
    assert not (out / "anchors" / "iface0").exists() or not list((out / "anchors" / "iface0").glob("*.npy"))
    tk = sum(np.load(f)["tokens"].shape[0] for f in (out / "topk").glob("*.npz"))
    assert tk == 8                                           # the heal store is training rows only
    assert np.array_equal(np.load(out / "val_rows.npy"), np.load(tmp_path / "rows.npy")[8:12])
