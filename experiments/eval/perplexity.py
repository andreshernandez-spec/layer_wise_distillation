"""Held-out next-token loss of a GPTNeoX checkpoint on the declared held-out rows.

    python experiments/eval/perplexity.py EleutherAI/pythia-410m out/heldout/heldout_rows.npy
"""
import argparse
import json
import math

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from lwd.eval.stitch import next_token_loss


def main(model, rows, n_rows, batch, device):
    ids = torch.from_numpy(np.load(rows)[:n_rows].astype(np.int64))
    m = GPTNeoXForCausalLM.from_pretrained(model, dtype=torch.float16 if device == "cuda" else torch.float32,
                                           attn_implementation="sdpa").to(device).eval()
    loss = next_token_loss(m, ids, batch)
    r = {"model": model, "rows": int(ids.shape[0]), "tokens": int(ids[:, 1:].numel()), "loss": loss, "ppl": math.exp(loss)}
    print(json.dumps(r))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("rows")
    p.add_argument("--n-rows", type=int, default=976); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.model, a.rows, a.n_rows, a.batch, a.device)
