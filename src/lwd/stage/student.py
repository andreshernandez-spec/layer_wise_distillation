"""A student stage: n fresh GPTNeoX blocks at width d_s, with an optional fixed
input/output bridge (the contract maps), trained to match a teacher stage in
transformed coordinates (docs/02).

    target  T~(x) = phi_out(T(x))
    student S~(z)  with z = phi_in(x)
    loss    || S~(z) - T~(x) ||^2  (relative, see rel_mse)

phi_in / phi_out are Contract objects (lwd.contract.whiten) or None.
"""
from __future__ import annotations

import torch
from torch import nn


def student_config(teacher_model_id: str, d_s: int, n_layers: int, n_heads: int,
                   intermediate: int | None = None):
    """A GPTNeoX config for the student blocks, inheriting everything else (rotary
    fraction, layernorm eps, parallel residual) from the teacher."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(teacher_model_id)
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads = d_s, n_layers, n_heads
    cfg.intermediate_size = intermediate or 4 * d_s
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    return cfg


class StudentStage(nn.Module):
    def __init__(self, cfg, seed: int = 0):
        super().__init__()
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            GPTNeoXLayer, GPTNeoXRotaryEmbedding)
        torch.manual_seed(seed)
        self.config = cfg
        self.layers = nn.ModuleList([GPTNeoXLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.rotary = GPTNeoXRotaryEmbedding(config=cfg)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(z.shape[1], device=z.device).unsqueeze(0)
        mask = create_causal_mask(config=self.config, inputs_embeds=z, attention_mask=None,
                                  past_key_values=None, position_ids=pos)
        pe = self.rotary(z, position_ids=pos)
        for layer in self.layers:
            z = layer(z, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        return z

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """sum||pred - target||^2 / sum||target - mean(target)||^2 over the batch."""
    t = target.reshape(-1, target.shape[-1])
    p = pred.reshape(-1, pred.shape[-1])
    return ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum().clamp_min(1e-12)
