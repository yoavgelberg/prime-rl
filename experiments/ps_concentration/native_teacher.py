"""Recompute the effective BF16 policy used by the fused native vLLM worker."""

from dataclasses import replace

import torch
from torch.utils.checkpoint import checkpoint

from prime_rl.trainer.ps_backward.attention import ps_attention
from prime_rl.trainer.ps_backward.model import rotate_half

from .linear import ConcentrationQwen


def effective_bf16(value):
    rounded = value.bfloat16().float()
    return value + (rounded - value).detach() if value.requires_grad else rounded


def native_rms(inputs, module):
    values = inputs.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
    return effective_bf16(normalized * effective_bf16(module.weight))


class NativeTeacher(ConcentrationQwen):
    def linear(self, inputs, weight, bias=None):
        return effective_bf16(super().linear(inputs, weight, bias))

    def attention(self, q, k, v, indices, prefix_length, plan):
        return ps_attention(
            q.bfloat16(), k.bfloat16(), v.bfloat16(), indices, indices, prefix_length, self.window, self.backend, plan
        )

    def _layer(self, layer, hidden, cos, sin, indices, prefix_length, plan):
        normalized = native_rms(hidden, layer.input_layernorm)
        residual = effective_bf16(hidden)
        attn, count = layer.self_attn, hidden.shape[1]
        q, k, v = [
            self.project(proj, normalized).view(1, count, -1, attn.head_dim)
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
        ]
        q, k = native_rms(q, attn.q_norm), native_rms(k, attn.k_norm)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = effective_bf16(q * cos + rotate_half(q) * sin)
        k = effective_bf16(k * cos + rotate_half(k) * sin)
        with torch.autocast(hidden.device.type, enabled=False):
            output = self.attention(q, k, v, indices, prefix_length, plan)
        mid = residual + self.project(attn.o_proj, output.transpose(1, 2).reshape(1, count, -1))
        normalized = native_rms(mid, layer.post_attention_layernorm)
        gate, up = self.project(layer.mlp.gate_proj, normalized), self.project(layer.mlp.up_proj, normalized)
        gated = effective_bf16(torch.nn.functional.silu(gate) * up)
        return effective_bf16(mid) + self.project(layer.mlp.down_proj, gated)

    def segment(self, ids, positions, indices, prefix_length, *, coordinate_gap=None):
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("A segment requires nonempty one-dimensional token IDs")
        hidden = effective_bf16(self.base.model.embed_tokens(ids.unsqueeze(0)))
        cos, sin = self.base.model.rotary_emb(hidden, positions.unsqueeze(0))
        cos, sin = effective_bf16(cos).unsqueeze(1), effective_bf16(sin).unsqueeze(1)
        plan = None
        if self.backend == "fa4":
            from prime_rl.trainer.ps_backward.fa4 import sparse_ps_plan

            block = 128 if torch.cuda.get_device_capability(hidden.device)[0] == 9 else 256
            plan = sparse_ps_plan(indices, indices, prefix_length, self.window, (block, 128))
            if coordinate_gap is not None:
                plan = replace(plan, coordinate_gap=coordinate_gap)
        for index, layer in enumerate(self.base.model.layers):

            def run(value, c, s, layer=layer):
                return self._layer(layer, value, c, s, indices, prefix_length, plan)

            if self.checkpoint_layers and index < self.depth - self.retain_last_layers and torch.is_grad_enabled():
                hidden = checkpoint(run, hidden, cos, sin, use_reentrant=False)
            else:
                hidden = run(hidden, cos, sin)
        return native_rms(hidden, self.base.model.norm).squeeze(0)
