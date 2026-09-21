"""Gate S1 must choose on validation and judge on held-out (docs/08)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(d, m, lam, wd, seed, val, held):
    r = {"cell": f"b49_st2_m{m:g}_cf{lam:g}_wd{wd:g}_s{seed}", "m": m, "cf_lambda": lam, "weight_decay": wd,
         "budget": {"budget_rows": 49, "budget_tokens": 100352, "train_tokens": 90112},
         "held_stitch_delta": held, "val_stitch_delta": val, "noise_per_distinct_real_position": 0.0,
         "real_passes": 1.0, "dispersion_student_over_teacher": 0.6,
         "select": {"stop": "cap", "best_step": 100, "positions": 1}}
    json.dump(r, open(d / f"{r['cell']}.json", "w"))


def gate(d):
    return subprocess.run([sys.executable, "experiments/phase2b/gate.py", str(d)], cwd=ROOT,
                          capture_output=True, text=True).stdout


def test_the_treatment_is_chosen_on_validation_not_on_its_held_out_score(tmp_path):
    for s in (0, 1):
        cell(tmp_path, 0, 0, 0.1, s, val=2.0, held=2.0 + 0.01 * s)          # control
        cell(tmp_path, 8, 0, 0.1, s, val=1.0, held=1.5 + 0.01 * s)          # best on validation
        cell(tmp_path, 32, 0, 0.1, s, val=1.4, held=0.9 + 0.01 * s)         # lucky on held-out
    out = gate(tmp_path)
    assert "treatment chosen on validation (1.0000): m=8" in out, out
    assert "held-out 1.5050" in out and "m=32" not in out.split("treatment chosen")[1].split("\n")[0]
    assert "S1 at this budget: PASS" in out


def test_overlapping_seeds_fail_the_gate_even_with_a_better_mean(tmp_path):
    cell(tmp_path, 0, 0, 0.1, 0, val=2.0, held=2.00); cell(tmp_path, 0, 0, 0.1, 1, val=2.0, held=1.40)
    cell(tmp_path, 8, 0, 0.1, 0, val=1.0, held=1.50); cell(tmp_path, 8, 0, 0.1, 1, val=1.0, held=1.20)
    out = gate(tmp_path)
    assert "ranges separated: False" in out and "S1 at this budget: fail" in out


def test_the_control_is_the_better_of_its_weight_decays_on_validation(tmp_path):
    for s in (0, 1):
        cell(tmp_path, 0, 0, 0.1, s, val=2.0, held=2.0 + 0.01 * s)
        cell(tmp_path, 0, 0, 1.0, s, val=1.6, held=1.7 + 0.01 * s)          # the stronger control
        cell(tmp_path, 8, 0, 0.1, s, val=1.0, held=1.65 + 0.01 * s)
    out = gate(tmp_path)
    assert "control chosen on validation (1.6000): wd=1" in out, out
    assert "gap +0.0500" in out
