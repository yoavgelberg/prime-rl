"""Qwen weight adapter for full-sequence and final-window PS baselines."""

from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import ps_attention


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class QwenPS(nn.Module):
    def __init__(
        self, base, window, backend="fa4", *, checkpoint_layers=False, retain_last_layers=0, fp32_linear_gradients=False
    ):
        super().__init__()
        if base.config.model_type not in ("qwen2", "qwen3"):
            raise ValueError("This model adapter supports dense Qwen2 and Qwen3")
        if getattr(base.config, "attention_dropout", 0) != 0:
            raise ValueError("Recorded backward requires dropout-free inference")
        if window < 1 or backend not in ("fa4", "dense"):
            raise ValueError("Choose a positive window and the fa4 or dense reference backend")
        self.base, self.window, self.backend = base, window, backend
        self.checkpoint_layers = checkpoint_layers
        self.retain_last_layers = retain_last_layers
        self.fp32_linear_gradients = fp32_linear_gradients
        self._weight_cast_cache = {}
        if not 0 <= retain_last_layers <= self.depth:
            raise ValueError("Invalid number of layers to retain without checkpointing")

    @property
    def device(self):
        return self.base.model.embed_tokens.weight.device

    @property
    def depth(self):
        return len(self.base.model.layers)

    def linear(self, inputs, weight, bias=None):
        if self.fp32_linear_gradients and torch.is_autocast_enabled(inputs.device.type):
            from .linear import fp32grad_linear

            if torch.get_autocast_dtype(inputs.device.type) != torch.bfloat16:
                raise ValueError("FP32 linear gradients require BF16 compute")
            stamp = weight._version, weight.data_ptr(), getattr(self, "_ps_update_version", 0)
            cached = self._weight_cast_cache.get(id(weight))
            if cached is None or cached[0] != stamp:
                cached = stamp, weight.detach().to(torch.bfloat16)
                self._weight_cast_cache[id(weight)] = cached
            return fp32grad_linear(inputs, weight, bias, cached[1])
        return torch.nn.functional.linear(inputs, weight, bias)

    def project(self, module, inputs):
        return self.linear(inputs, module.weight, module.bias)

    def _layer(self, layer, hidden, cos, sin, indices, prefix_length, plan):
        normalized = layer.input_layernorm(hidden)
        attn, count = layer.self_attn, hidden.shape[1]
        q, k, v = [
            self.project(proj, normalized).view(1, count, -1, attn.head_dim)
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
        ]
        if hasattr(attn, "q_norm"):
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if torch.is_autocast_enabled(hidden.device.type):
            dtype = torch.get_autocast_dtype(hidden.device.type)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        with torch.autocast(hidden.device.type, enabled=False):
            output = ps_attention(q, k, v, indices, indices, prefix_length, self.window, self.backend, plan)
        hidden = hidden + self.project(attn.o_proj, output.transpose(1, 2).reshape(1, count, -1))
        normalized = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        if self.fp32_linear_gradients:
            gated = mlp.act_fn(self.project(mlp.gate_proj, normalized)) * self.project(mlp.up_proj, normalized)
            return hidden + self.project(mlp.down_proj, gated)
        return hidden + mlp(normalized)

    def segment(self, ids, positions, indices, prefix_length, *, coordinate_gap=None):
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("A segment requires nonempty one-dimensional token IDs")
        hidden = self.base.model.embed_tokens(ids.unsqueeze(0))
        cos, sin = self.base.model.rotary_emb(hidden, positions.unsqueeze(0))
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        plan = None
        if self.backend == "fa4":
            from .fa4 import sparse_ps_plan

            block = 128 if torch.cuda.get_device_capability(hidden.device)[0] == 9 else 256
            plan = sparse_ps_plan(indices, indices, prefix_length, self.window, (block, 128))
            if coordinate_gap is not None:
                plan = replace(plan, coordinate_gap=coordinate_gap)
        for index, layer in enumerate(self.base.model.layers):
            # Checkpoint recomputation happens after this loop; bind each layer.
            def run(x, c, s, layer=layer):
                return self._layer(layer, x, c, s, indices, prefix_length, plan)

            if self.checkpoint_layers and index < self.depth - self.retain_last_layers and torch.is_grad_enabled():
                hidden = checkpoint(run, hidden, cos, sin, use_reentrant=False)
            else:
                hidden = run(hidden, cos, sin)
        return self.base.model.norm(hidden).squeeze(0)

    def logps(self, hidden, labels, block_size=256, temperatures=None):
        def head(h, weight, targets, temperature):
            logits = self.linear(h, weight)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            if temperature is not None:
                logits = logits / temperature.unsqueeze(-1)
            return -torch.nn.functional.cross_entropy(logits, targets, reduction="none")

        pieces = []
        for start in range(0, labels.numel(), block_size):
            h, targets = hidden[start : start + block_size], labels[start : start + block_size]
            temperature = temperatures[start : start + block_size] if temperatures is not None else None
            args = h, self.base.lm_head.weight, targets, temperature
            pieces.append(checkpoint(head, *args, use_reentrant=False) if torch.is_grad_enabled() else head(*args))
        return torch.cat(pieces)
