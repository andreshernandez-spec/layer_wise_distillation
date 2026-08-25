"""Attention entropy per head per block on real inputs (docs/01 C0.2, the Phase 1
diagnostic baseline). Eager attention so weights are returned.

    python experiments/phase0/attn_entropy.py EleutherAI/pythia-1.4b out/slice/anchor_rows.npy out/attn_entropy_1.4b.npz --rows 64
"""
import argparse

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from lwd.harvest.stats import attention_entropy


def main(model_id, slice_path, out, rows, seq_len, device, batch):
    # float32: fp16 eager attention overflows on massive activations (NaN from block 13)
    m = GPTNeoXForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                           attn_implementation="eager").to(device).eval()
    ids = torch.from_numpy(np.load(slice_path)[:rows, :seq_len].astype(np.int64))
    layers = m.gpt_neox.layers
    acc = torch.zeros(len(layers), m.config.num_attention_heads, dtype=torch.float64)
    # reduce each layer's attention map inside a hook: holding all of them costs
    # n_layers x heads x L^2 (12 GB at 24 layers, L=2048, fp16)
    def hook(i):
        def f(mod, args, output):
            w = output[1]
            if w is not None:
                acc[i] += attention_entropy(w.float()).cpu()
        return f
    hs = [l.attention.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    n = 0
    with torch.no_grad():
        for i in range(0, rows, batch):
            m(input_ids=ids[i:i + batch].to(device))
            n += 1
    for h in hs:
        h.remove()
    acc = (acc / n).numpy()
    np.savez(out, entropy=acc, seq_len=seq_len, rows=rows, model=model_id)
    print("mean entropy per layer (nats), uniform =", np.log(seq_len).round(2))
    print(acc.mean(1).round(3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("slice"); p.add_argument("out")
    p.add_argument("--rows", type=int, default=64); p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    main(a.model, a.slice, a.out, a.rows, a.seq_len, a.device, a.batch)
