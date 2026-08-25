"""Materialize held-out rows from the Pile test split (docs/01 C0.1): documents are
tokenized with the Pythia tokenizer, joined with EOS, and cut into rows of 2049 tokens
exactly like the training stream. Never seen by any Pythia model.

    python experiments/phase0/heldout.py out/heldout/test.jsonl.zst out/heldout 2000000
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROW = 2049
EOS = 0  # <|endoftext|> in the Pythia/GPT-NeoX tokenizer


def main(src, out_dir, n_tokens):
    tok = Tokenizer.from_file(hf_hub_download("EleutherAI/pythia-70m", "tokenizer.json"))
    n_rows = int(n_tokens) // ROW
    buf, rows, docs = [], [], 0
    with open(src, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            ids = tok.encode(json.loads(line)["text"]).ids + [EOS]
            buf.extend(ids); docs += 1
            while len(buf) >= ROW and len(rows) < n_rows:
                rows.append(buf[:ROW]); buf = buf[ROW:]
            if len(rows) >= n_rows:
                break
    arr = np.array(rows, dtype=np.uint16)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "heldout_rows.npy", arr)
    meta = {"source": str(src), "rows": int(arr.shape[0]), "tokens": int(arr.size), "docs": docs,
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    json.dump(meta, open(out / "heldout_rows.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
