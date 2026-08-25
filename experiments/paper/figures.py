"""The paper's figures, generated from the run records.

Same rule as claims.py: nothing is drawn from a number typed into this file. Every
series is read from out/, so a figure cannot drift from the data behind it.

    python experiments/paper/figures.py [--out paper/figures]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

P2 = "out/phase2-1.4b-a100"
INK, GRID = "#1a1a1a", "#d8d8d8"


def heals():
    d = {}
    for f in sorted(glob.glob(f"{P2}/heal_*.json")):
        r = json.load(open(f))
        hs = r.get("heal_seed", r["seed"])
        key = r["init"] + r.get("tag", "") + (f"_h{hs}" if hs != r["seed"] else "")
        d[(key, r["tokens"])] = r
    return d


def fig_heal_curve(H, out):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    budgets = [1e5, 1e6, 3e6, 1e7]
    series = [("random", "random init", "o", "#b03a2e"),
              ("stagewise", "noise recipe", "s", "#1f618d"),
              ("oracle", "real activations", "^", "#196f3d")]
    for key, label, mk, col in series:
        xs = [t for t in budgets if (key, t) in H]
        ys = [H[(key, t)]["loss_after"] for t in xs]
        ax.plot(xs, ys, marker=mk, color=col, label=label, lw=1.6, ms=5)
        no_heal = H[(key, budgets[0])]["loss_before"]
        ax.scatter([budgets[0] / 3], [no_heal], marker=mk, color=col, alpha=0.45, s=28)
    ax.set_xscale("log")
    ax.set_xlabel("heal tokens (leftmost point, faded: no heal)")
    ax.set_ylabel("held-out next-token loss (nats)")
    ax.set_title("Healing closes most of what separates the recipes", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "heal_curve.pdf"); fig.savefig(out / "heal_curve.png", dpi=180)
    plt.close(fig)


def fig_drift(out):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for m, label, col in [("R", "real activations", "#196f3d"),
                          ("C", "noise recipe", "#1f618d"),
                          ("C_dagger", "noise + on-policy retraining", "#7d3c98")]:
        p = Path(f"{P2}/drift_{m}.json")
        if not p.exists():
            continue
        d = json.load(open(p))
        # interface 0 is zero by construction (the student uses the teacher's embedding)
        # and would run off a log axis
        ax.plot(range(1, len(d["realized_drift"])), d["realized_drift"][1:],
                marker="o", ms=4, lw=1.6, color=col, label=label)
    d = json.load(open(f"{P2}/drift_C.json"))
    pp = d["predicted_from_stage1_drift"]
    ax.plot(range(1, len(pp) + 1), pp, ls="--", lw=1.4, color="#b03a2e",
            label="pure propagation, no fresh error (noise recipe)")
    ax.set_yscale("log")
    ax.set_xlabel("interface"); ax.set_ylabel("realized drift (whitened, log scale)")
    ax.set_title("Drift should decay if it were only propagated; it grows", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6, which="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "drift.pdf"); fig.savefig(out / "drift.png", dpi=180)
    plt.close(fig)


def fig_collapse(H, out):
    """The paper's spine. Every row is computed; the first draft of this figure carried
    a 2.0 taken from G1's pre-registered pass margin, which is a threshold and not a
    measured effect."""
    import numpy as np
    sw = float(np.mean([H[k]["loss_after"] for k in H
                        if k[0].startswith("stagewise") and "dagger" not in k[0] and k[1] == 1e7]))
    dg = float(np.mean([H[k]["loss_after"] for k in H if "dagger" in k[0] and k[1] == 1e7]))
    orc = H[("oracle", 1e7)]
    plain, rnd_eq = H[("stagewise", 1e7)], H[("random_eqflops", 183670000.0)]
    rows = [
        ("real activations vs\nthe noise recipe",
         plain["loss_before"] - orc["loss_before"], sw - orc["loss_after"]),
        ("on-policy\nretraining",
         plain["loss_before"] - H[("stagewise_dagger", 1e7)]["loss_before"], sw - dg),
        ("the stagewise init\nitself, vs random",
         H[("random", 1e7)]["loss_before"] - plain["loss_before"],
         rnd_eq["loss_after"] - sw),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = range(len(rows))
    ax.barh([i + 0.18 for i in y], [r[1] for r in rows], height=0.34,
            color="#c0c8d0", label="measured before healing")
    ax.barh([i - 0.18 for i in y], [r[2] for r in rows], height=0.34,
            color=["#1f618d", "#1f618d", "#b03a2e"], label="survives an equal-FLOPs heal")
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.06, i + 0.18, f"{r[1]:.2f}", va="center", fontsize=8.5, color=INK)
        c = "#b03a2e" if r[2] < 0 else "#1f618d"
        ax.text(r[2] + (0.06 if r[2] >= 0 else -0.06), i - 0.18, f"{r[2]:+.2f}",
                va="center", ha="left" if r[2] >= 0 else "right", fontsize=8.5, color=c)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("nats of held-out next-token loss")
    ax.set_title("Every interface-level gap shrinks by about 10x, and the last one flips",
                 fontsize=10.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "collapse.pdf"); fig.savefig(out / "collapse.png", dpi=180)
    plt.close(fig)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = heals()
    fig_heal_curve(H, out); fig_drift(out); fig_collapse(H, out)
    for f in sorted(out.glob("*.pdf")):
        print(f, f.stat().st_size, "bytes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="paper/figures")
    main(p.parse_args())
