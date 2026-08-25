"""C0.1 decontamination: 13-gram overlap between the slice and the pre-declared eval
suite. Reports and writes the set of slice rows to drop.

    python experiments/phase0/decontam.py out/slice/anchor_rows.npy out/slice/decontam.json
"""
import json
import sys
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

N = 13
# task -> (hf dataset, config, split, text fields). Pythia's reported suite plus HellaSwag.
SUITE = {
    "lambada_openai": ("EleutherAI/lambada_openai", "default", "test", ["text"]),
    "piqa": ("ybisk/piqa", None, "validation", ["goal", "sol1", "sol2"]),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", ["sentence"]),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", ["question"]),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", ["question"]),
    "sciq": ("allenai/sciq", None, "test", ["question", "support"]),
    "logiqa": ("EleutherAI/logiqa", None, "test", ["context", "question"]),
    "hellaswag": ("Rowan/hellaswag", None, "validation", ["ctx"]),
}


def ngrams(ids, n=N):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def main(slice_path, out):
    tok = Tokenizer.from_file(hf_hub_download("EleutherAI/pythia-70m", "tokenizer.json"))
    eval_grams = defaultdict(set)
    for task, (ds, cfg, split, fields) in SUITE.items():
        try:
            d = load_dataset(ds, cfg, split=split)
        except RuntimeError:  # script-based dataset: use HF's auto-converted parquet branch
            d = load_dataset(ds, cfg, split=split, revision="refs/convert/parquet")
        for ex in d:
            for f in fields:
                if ex.get(f):
                    eval_grams[task] |= ngrams(tok.encode(ex[f]).ids)
        print(task, len(d), "examples,", len(eval_grams[task]), f"{N}-grams", flush=True)
    rows = np.load(slice_path)
    hits = defaultdict(list)
    for r, row in enumerate(rows):
        g = ngrams(row.tolist())
        for task, eg in eval_grams.items():
            if g & eg:
                hits[task].append(r)
    drop = sorted({r for rs in hits.values() for r in rs})
    rep = {"n": N, "rows": int(len(rows)), "drop": drop, "per_task": {k: len(v) for k, v in hits.items()}}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "drop"}), "dropped", len(drop))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
