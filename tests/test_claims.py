"""The paper's numbers must come from the result files, not from a doc that has drifted.

Skipped when the result trees are absent: out/ is gitignored and 31 GB.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "out" / "phase1-1.4b-a100", ROOT / "out" / "phase2-1.4b-a100"]

pytestmark = pytest.mark.skipif(
    not all(p.is_dir() for p in NEEDED),
    reason="needs out/phase1-1.4b-a100 and out/phase2-1.4b-a100")


def test_every_published_number_still_matches_its_result_file():
    r = subprocess.run([sys.executable, "experiments/paper/claims.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"a published number drifted from the data:\n{r.stdout}\n{r.stderr}"
    assert "0 disagreeing" in r.stdout, r.stdout
