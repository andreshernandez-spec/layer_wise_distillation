"""Step 1 of docs/08 as a table and a figure, generated from the cell records.

    python experiments/phase2b/report.py out/phase2b/cells-pod [--fig paper/figures/dose.png]

The table answers the question the protocol was opened with: how much noise does it take,
and what does it cost. Columns: the dose m, the CF weight, held-out stitching delta over
seeds, the step validation chose and why the run stopped, noise positions per distinct
real token, passes over the real data, FLOPs as a multiple of the control's, and the
student's output dispersion relative to the teacher's.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict


def load(d):
    cells = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f))
        cells[(r["budget"]["budget_rows"], r["m"], r["cf_lambda"], r["weight_decay"])].append(r)
    return cells


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def table(cells):
    out = []
    for B in sorted({k[0] for k in cells}):
        rows = {k[1:]: v for k, v in cells.items() if k[0] == B}
        tok = next(iter(rows.values()))[0]["budget"]["budget_tokens"]
        ctrl = [v for k, v in rows.items() if k[0] == 0 and k[1] == 0]
        ctrl_flops = min(mean(r["flops"]["total"] for r in v) for v in ctrl) if ctrl else float("nan")
        out.append(f"\n**Budget {B} rows ({tok:,} real tokens).** Held-out stitching delta, nats.\n")
        out.append("| m | lambda | wd | seeds | held-out | range | chosen step | stopped | noise pos / real token "
                   "| passes over D | FLOPs x control | dispersion |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(rows):
            rs = rows[key]
            h = [r["held_stitch_delta"] for r in rs]
            stops = "/".join(sorted({r["select"]["stop"] for r in rs}))
            out.append(f"| {key[0]:g} | {key[1]:g} | {key[2]:g} | {len(rs)} | {mean(h):.3f} | "
                       f"{min(h):.3f} to {max(h):.3f} | {mean(r['select']['best_step'] for r in rs):.0f} | {stops} | "
                       f"{mean(r['noise_per_distinct_real_position'] for r in rs):.0f} | "
                       f"{mean(r['real_passes'] for r in rs):.0f} | "
                       f"{mean(r['flops']['total'] for r in rs) / ctrl_flops:.1f} | "
                       f"{mean(r['dispersion_student_over_teacher'] for r in rs):.2f} |")
    return "\n".join(out)


def figure(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = sorted({k[0] for k in cells})
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 3.9), squeeze=False)
    for ax, B in zip(axes[0], budgets):
        rows = {k[1:]: v for k, v in cells.items() if k[0] == B}
        tok = next(iter(rows.values()))[0]["budget"]["budget_tokens"]
        for lam, col, lab in ((0, "#1f618d", "no CF term"), (3, "#b9770e", "CF, lambda 3"), (30, "#b03a2e", "CF, lambda 30")):
            pts = sorted((k[0], [r["held_stitch_delta"] for r in v]) for k, v in rows.items()
                         if k[1] == lam and k[2] == 0.1)
            if not pts:
                continue
            xs = [p[0] if p[0] > 0 else 0.5 for p in pts]          # m = 0 drawn at 0.5 on a log axis
            ax.plot(xs, [mean(p[1]) for p in pts], marker="o", ms=4, lw=1.5, color=col, label=lab)
            for x, p in zip(xs, pts):
                ax.plot([x] * len(p[1]), p[1], ls="none", marker="_", ms=9, color=col, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 2, 8, 32]); ax.set_xticklabels(["0\n(control)", "2", "8", "32"])
        ax.set_xlabel("noise positions per real position, m")
        ax.set_ylabel("held-out stitching delta (nats)")
        ax.set_title(f"{tok:,} real tokens", fontsize=10.5)
        ax.grid(True, color="#d8d8d8", lw=0.6)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Dose-response at a fixed real-data budget, every arm validation-selected", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--fig", default=None)
    a = p.parse_args()
    cells = load(a.dir)
    if not cells:
        sys.exit("no cells")
    print(table(cells))
    if a.fig:
        figure(cells, a.fig); print(f"\nfigure: {a.fig}")
