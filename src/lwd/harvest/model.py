"""GPTNeoX stage access: resident (hooks on the full model) and streaming (only the
blocks of one stage resident). Both must produce bitwise-identical interface tensors
(docs/01 G0.1); the test is tests/test_harvest.py.

Interface k is the residual stream entering block stage_bounds[k][0]; interface S is
the input to final_layer_norm.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


def stage_bounds(n_layers: int, n_stages: int) -> list[tuple[int, int]]:
    assert n_layers % n_stages == 0, (n_layers, n_stages)
    s = n_layers // n_stages
    return [(k * s, (k + 1) * s) for k in range(n_stages)]


def snapshot_dir(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))


def _weight_index(snap: Path) -> dict[str, Path]:
    """param name -> safetensors file."""
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        m = json.load(open(idx))["weight_map"]
        return {k: snap / v for k, v in m.items()}
    from safetensors import safe_open
    out = {}
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                out[k] = f
    return out


def _load(prefix_filter, snap: Path, dtype) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    wi = _weight_index(snap)
    keys = [k for k in wi if prefix_filter(k)]
    out = {}
    by_file: dict[Path, list[str]] = {}
    for k in keys:
        by_file.setdefault(wi[k], []).append(k)
    for f, ks in by_file.items():
        with safe_open(f, framework="pt") as sf:
            for k in ks:
                out[k] = sf.get_tensor(k).to(dtype)
    return out


class StageRunner(nn.Module):
    """Blocks [a, b) of a GPTNeoX model, loaded alone. forward(h) -> h'."""

    def __init__(self, model_id: str, a: int, b: int, dtype=torch.float16,
                 attn: str = "sdpa", device="cpu"):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            GPTNeoXLayer, GPTNeoXRotaryEmbedding)
        cfg = AutoConfig.from_pretrained(model_id)
        cfg._attn_implementation = attn
        self.config, self.a, self.b = cfg, a, b
        snap = snapshot_dir(model_id)
        with torch.device("meta"):
            layers = nn.ModuleList([GPTNeoXLayer(cfg, i) for i in range(a, b)])
        sd = _load(lambda k: any(k.startswith(f"gpt_neox.layers.{i}.") for i in range(a, b)),
                   snap, dtype)
        sd = {k[len("gpt_neox.layers."):]: v for k, v in sd.items()}
        sd = {str(int(k.split(".")[0]) - a) + k[k.index("."):]: v for k, v in sd.items()}
        missing, unexpected = layers.load_state_dict(sd, strict=False, assign=True)
        assert not missing, missing
        # old Pythia checkpoints carry attention.bias / masked_bias / rotary inv_freq
        assert all(u.endswith(("attention.bias", "attention.masked_bias",
                               "rotary_emb.inv_freq")) for u in unexpected), unexpected
        self.layers = layers.to(device)
        self.rotary = GPTNeoXRotaryEmbedding(config=cfg).to(device)
        self.eval()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Differentiable; wrap in torch.no_grad() at call sites that do not need it."""
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        mask = create_causal_mask(config=self.config, inputs_embeds=h, attention_mask=None,
                                  past_key_values=None, position_ids=pos)
        pe = self.rotary(h, position_ids=pos)
        for layer in self.layers:
            h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        return h


class Edges(nn.Module):
    """Embedding (interface 0 from tokens) and head (logits from interface S)."""

    def __init__(self, model_id: str, dtype=torch.float16, device="cpu"):
        super().__init__()
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        snap = snapshot_dir(model_id)
        sd = _load(lambda k: k.startswith(("gpt_neox.embed_in.", "gpt_neox.final_layer_norm.",
                                           "embed_out.")), snap, dtype)
        self.embed_in = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.embed_out = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.embed_in.weight = nn.Parameter(sd["gpt_neox.embed_in.weight"])
        self.final_layer_norm.weight = nn.Parameter(sd["gpt_neox.final_layer_norm.weight"])
        self.final_layer_norm.bias = nn.Parameter(sd["gpt_neox.final_layer_norm.bias"])
        self.embed_out.weight = nn.Parameter(sd["embed_out.weight"])
        self.to(device).eval()

    @torch.no_grad()
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_in(ids)

    @torch.no_grad()
    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.embed_out(self.final_layer_norm(h))


class ResidentModel:
    """Full model with hooks capturing every interface in one forward."""

    def __init__(self, model_id: str, n_stages: int, dtype=torch.float16,
                 attn: str = "sdpa", device="cpu"):
        from transformers import GPTNeoXForCausalLM
        self.model = GPTNeoXForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn).to(device).eval()
        cfg = self.model.config
        self.bounds = stage_bounds(cfg.num_hidden_layers, n_stages)
        self._cap: list[torch.Tensor | None] = [None] * (n_stages + 1)
        layers = self.model.gpt_neox.layers
        for k, (a, _) in enumerate(self.bounds):
            layers[a].register_forward_pre_hook(self._hook(k))
        self.model.gpt_neox.final_layer_norm.register_forward_pre_hook(self._hook(n_stages))

    def _hook(self, k):
        def f(mod, args):
            self._cap[k] = args[0]
        return f

    @torch.no_grad()
    def forward(self, ids: torch.Tensor):
        """-> (interfaces [I0..IS], logits)."""
        out = self.model(input_ids=ids)
        ifaces = list(self._cap)
        assert all(t is not None for t in ifaces)
        return ifaces, out.logits


class Lower(torch.nn.Module):
    """Embedding plus blocks [0, a): token ids -> interface a. For live real arms."""

    def __init__(self, model_id: str, a: int, dtype=torch.float16, attn: str = "sdpa", device="cpu"):
        super().__init__()
        self.edges = Edges(model_id, dtype, device)
        self.stage = StageRunner(model_id, 0, a, dtype, attn, device) if a > 0 else None

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.edges.embed(ids)
        return self.stage(h) if self.stage is not None else h
