"""Cross-check the numbers written in paper/paper.md against the claims registry.

Not a parser for arbitrary prose: it holds an explicit map from a string that appears in
the paper to the claim that must back it, so a number cannot be edited in one place and
not the other.

    python experiments/paper/check_paper.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "paper.md"

# text in the paper -> (claim key, how many digits the paper rounds to)
BOUND = [
    ("3.5520", "heal.random_eqflops.mean", 4),
    ("3.7283", "heal.stagewise.mean", 4),
    ("3.6375", "heal.dagger.mean", 4),
    ("3.4931", "heal.oracle.mean", 4),
    ("5.2693", "heal.random@1e7", 4),
    ("0.176", "kill.gap_vs_mean", 3),
    ("0.0135", "repro.cross_machine", 4),
    ("0.085", "kill.gap_vs_dagger", 3),
    ("0.235", "criterion5.gap", 3),
    ("0.091", "dagger.healed_effect", 3),
    ("3.338", "dagger.unhealed_effect", 3),
    ("1.8367e8", "flops.equal_tokens", None),
    ("6.6671e17", "flops.stagewise_end_to_end", None),
    ("0.029", "drift.C.pure_propagation_at_output", 3),
    ("2.335", "drift.C.output", 3),
    ("1.387", "drift.C_dagger.output", 3),
    ("1.473", "drift.R.output", 3),
    ("3.1%", "lipschitz.contractive_max_disagreement", None),
    ("0.038", "Rstack.stage0.jacobian_cos", 3),
    ("0.524", "Rstack.stage0.eps", 3),
    ("0.326", "beta[L_iid]", 3),
    ("0.336", "beta[R_iid]", 3),
    ("0.313", "beta[C_mix]", 3),
    ("0.334", "beta[G_iid]", 3),
    ("0.206", "beta[I_iid]", 3),
    ("113", "phase1_cells", 0),
    ("0.059", "kill.gap_vs_oracle", 3),
    ("0.353", "iface.L.eps", 3),
    ("0.452", "iface.L.stitch", 3),
    ("0.412", "iface.R.eps", 3),
    ("0.450", "iface.R.stitch", 3),
    ("0.463", "iface.C_mix.eps", 3),
    ("0.527", "iface.C_mix.stitch", 3),
    ("11.07", "iface.G_iid.eps", 2),
    ("7.92", "iface.G_iid.stitch", 2),
    ("0.773", "crossover.noise_worth@94208", 3),
    ("0.142", "crossover.noise_worth@188416", 3),
]


def check_tables(text):
    """Each table in the paper must be byte-identical to what tables.py generates."""
    import subprocess as sp
    bad = []
    for name in ("interface", "heal", "equalflops", "drift"):
        out = sp.run([sys.executable, "experiments/paper/tables.py", "--name", name],
                     cwd=ROOT, capture_output=True, text=True).stdout
        body = out.split("\n", 1)[1].rsplit("<!-- /TABLE", 1)[0].strip()
        if body and body not in text:
            bad.append(f"table {name!r} in the paper differs from tables.py output")
    return bad


def main():
    reg_path = ROOT / "paper" / "claims.json"
    subprocess.run([sys.executable, "experiments/paper/claims.py", "--json", str(reg_path)],
                   cwd=ROOT, check=False, capture_output=True)
    reg = json.load(open(reg_path))
    text = PAPER.read_text()
    bad = []
    for literal, key, digits in BOUND:
        if literal not in text:
            bad.append(f"{literal!r} ({key}) is not in the paper any more")
            continue
        if key not in reg:
            bad.append(f"{literal!r} cites {key}, which claims.py does not produce")
            continue
        v = reg[key]["value"]
        if digits is None:
            got = f"{abs(v):.4g}"
            want = re.sub(r"[%e].*$", "", literal.replace("e8", "").replace("e17", ""))
            ok = literal.rstrip("%") in (f"{abs(v):.4g}", f"{abs(v)*100:.1f}", f"{abs(v):.4e}") \
                 or abs(abs(v) - float(literal.rstrip("%")) / (100 if "%" in literal else 1)) < 5e-3 * max(1, abs(v)) \
                 or f"{abs(v):.4e}".replace("e+0", "e").replace("e+", "e") == literal
        else:
            ok = f"{abs(v):.{digits}f}" == literal
        if not ok:
            bad.append(f"{literal!r} does not round from {key} = {v:.6g}")
    bad += check_tables(text)
    for b in bad:
        print("MISMATCH:", b)
    print(f"\n{len(BOUND)} paper numbers and 4 tables checked, {len(bad)} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
