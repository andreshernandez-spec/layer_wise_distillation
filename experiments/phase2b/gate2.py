"""Gate S2 of the data-limited protocol (docs/08): composed and healed on the same real
data, does the treatment beat BOTH plain KD and the same pipeline without it?

For each arm the learning rate is chosen on validation loss, then the held-out loss of the
chosen rate is read, over heal seeds.

    python experiments/phase2b/gate2.py out/phase2b/heals
"""
import argparse
import glob
import json
import sys
from collections import defaultdict

MARGIN = 0.10          # pre-registered in docs/08: five times the 0.02 run-to-run noise


def main(d):
    runs = defaultdict(lambda: defaultdict(list))            # budget -> (arm, lr) -> records
    for f in sorted(glob.glob(f"{d}/heal_*.json")):
        r = json.load(open(f))
        arm = "B0 plain KD" if r["init"] == "random" else (
            "control" if (r["m"] == 0 and r["cf_lambda"] == 0) else
            f"treatment m={r['m']:g} cf={r['cf_lambda']:g}")
        runs[r["budget"]["budget_rows"]][(arm, r["lr"])].append(r)
    passed = []
    for B in sorted(runs):
        tokens = next(iter(runs[B].values()))[0]["budget"]["budget_tokens"]
        print(f"\n=== budget {B} rows ({tokens:,} real tokens): held-out next-token loss (nats)\n")
        chosen = {}
        for arm in sorted({k[0] for k in runs[B]}):
            by_lr = {lr: rs for (a, lr), rs in runs[B].items() if a == arm}
            lr = min(by_lr, key=lambda x: sum(r["val_loss"] for r in by_lr[x]) / len(by_lr[x]))
            rs = by_lr[lr]
            h = [r["held_loss"] for r in rs]
            chosen[arm] = {"lr": lr, "vals": h, "mean": sum(h) / len(h), "rs": rs}
        b0 = chosen.get("B0 plain KD")
        for arm, c in chosen.items():
            r0 = c["rs"][0]
            mult = (sum(r["flops"]["total"] for r in c["rs"]) / len(c["rs"])) / \
                   (sum(r["flops"]["total"] for r in b0["rs"]) / len(b0["rs"])) if b0 else float("nan")
            print(f"  {arm:28s} lr {c['lr']:g}  n={len(c['vals'])}  held {c['mean']:.4f}  "
                  f"[{', '.join('%.4f' % v for v in c['vals'])}]  epochs to best "
                  f"{sum(r['select']['epochs_to_best'] for r in c['rs']) / len(c['rs']):.1f}  "
                  f"noise pos/real tok {r0['noise_positions'] / max(1, r0['budget']['train_tokens']):.0f}  "
                  f"FLOPs x{mult:.1f} of B0")
        treats = {a: c for a, c in chosen.items() if a.startswith("treatment")}
        if not treats or "control" not in chosen or b0 is None:
            print("  an arm is missing, S2 not evaluable at this budget"); continue
        ta, t = min(treats.items(), key=lambda kv: sum(r["val_loss"] for r in kv[1]["rs"]) / len(kv[1]["rs"]))
        ok = True
        for name in ("control", "B0 plain KD"):
            o = chosen[name]
            gap = o["mean"] - t["mean"]
            sep = max(t["vals"]) < min(o["vals"])
            print(f"  {ta} vs {name}: gap {gap:+.4f} (needs >= {MARGIN}), ranges separated: {sep}")
            ok = ok and gap >= MARGIN and sep
        print(f"  S2 at this budget: {'PASS' if ok else 'fail'}")
        if ok:
            passed.append(B)
    print("\nS2:", f"PASS at budgets {passed}" if passed else "FAIL at every budget")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dir")
    sys.exit(main(p.parse_args().dir))
