"""Collect lm-eval results under out/eval/*/ into one markdown table.

    python experiments/eval/summarize.py out/eval
"""
import glob
import json
import sys

METRIC = {"lambada_openai": "acc", "piqa": "acc_norm", "winogrande": "acc", "arc_easy": "acc_norm",
          "arc_challenge": "acc_norm", "sciq": "acc_norm", "hellaswag": "acc_norm"}


def main(root):
    rows = {}
    for f in glob.glob(f"{root}/*/**/results_*.json", recursive=True):
        r = json.load(open(f))
        ma = r.get("config", {}).get("model_args", f)
        name = ma.get("pretrained", f) if isinstance(ma, dict) else str(ma).split("pretrained=")[-1].split(",")[0]
        rows[name] = {t: r["results"][t].get(f"{m},none", r["results"][t].get(m)) for t, m in METRIC.items() if t in r["results"]}
    tasks = list(METRIC)
    print("| model | " + " | ".join(tasks) + " | mean |")
    print("|---|" + "---|" * (len(tasks) + 1))
    for name, res in sorted(rows.items()):
        vals = [res.get(t) for t in tasks]
        mean = sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals))
        print(f"| {name} | " + " | ".join("" if v is None else f"{100*v:.1f}" for v in vals) + f" | {100*mean:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/eval")
