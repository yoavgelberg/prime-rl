"""Cold PS rebuilds at original positions, with independent continuation caches."""

import torch
import torch.nn.functional as F

from prime_rl.trainer.ps_backward.model import rotate_half
from prime_rl.trainer.ps_backward.attention import ps_attention

from .native_teacher import NativeTeacher, effective_bf16, native_rms


class CachedTeacher(NativeTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys, self.values = [], []

    def attention(self, q, k, v, indices, prefix_length, plan):
        self.keys.append(k.bfloat16())
        self.values.append(v.bfloat16())
        return super().attention(q, k, v, indices, prefix_length, plan)


class NativeCold:
    def __init__(self, base, prefix, window, copies=1, *, backend='fa4'):
        if min(prefix, window, copies) < 1:
            raise ValueError('Cold rebuilds require a nonempty pinned prefix, suffix window, and batch')
        self.base, self.prefix, self.window, self.copies = base, prefix, window, copies
        self.backend = backend
        self.adapter = CachedTeacher(base, window, backend=backend)
        self.keys, self.values = [], []
        self.indices = None

    @torch.inference_mode()
    def prefill(self, tokens, positions):
        if self.indices is not None:
            raise ValueError('Each branch must start with a fresh cold cache')
        if not self.prefix <= len(tokens) <= self.prefix + self.window or len(tokens) != len(positions):
            raise ValueError('The cold input is exactly the pinned prefix plus retained suffix')
        if positions[:self.prefix] != list(range(self.prefix)) or any(b <= a for a, b in zip(positions, positions[1:])):
            raise ValueError('Keep original increasing position IDs and the immutable initial prefix')
        ids = torch.tensor(tokens, device=self.base.device)
        indices = torch.tensor(positions, device=self.base.device)
        hidden = self.adapter.segment(ids, indices, indices, self.prefix)
        for key, value in zip(self.adapter.keys, self.adapter.values, strict=True):
            shape = self.copies, key.shape[1], self.prefix + self.window, key.shape[-1]
            keys = torch.zeros(shape, dtype=torch.bfloat16, device=self.base.device)
            values = torch.zeros_like(keys)
            keys[:, :, :len(tokens)].copy_(key.expand(self.copies, -1, -1, -1))
            values[:, :, :len(tokens)].copy_(value.expand(self.copies, -1, -1, -1))
            self.keys.append(keys)
            self.values.append(values)
        self.adapter.keys.clear()
        self.adapter.values.clear()
        self.indices = torch.full((self.prefix + self.window,), -1, device=self.base.device, dtype=torch.long)
        self.indices[:len(tokens)] = indices
        self.next_slot = self.prefix + (len(tokens) - self.prefix) % self.window
        self.next_position = positions[-1] + 1
        logits = self.adapter.linear(hidden[-1:], self.base.lm_head.weight)
        return logits.expand(self.copies, -1).clone()

    @torch.inference_mode()
    def step(self, tokens):
        if self.indices is None or tokens.shape != (self.copies,):
            raise ValueError('Prefill this branch before continuing one token per independent cache')
        position, slot = self.next_position, self.next_slot
        hidden = effective_bf16(self.base.model.embed_tokens(tokens[:, None]))
        positions = torch.full((self.copies, 1), position, device=tokens.device, dtype=torch.long)
        cos, sin = self.base.model.rotary_emb(hidden, positions)
        cos, sin = effective_bf16(cos)[:, None], effective_bf16(sin)[:, None]
        self.indices[slot] = position
        allowed = (self.indices >= 0) & (self.indices <= position) & (
            (self.indices < self.prefix) | (self.indices >= position - self.window + 1))
        for layer, keys, values in zip(self.base.model.layers, self.keys, self.values, strict=True):
            normalized, residual = native_rms(hidden, layer.input_layernorm), effective_bf16(hidden)
            attn = layer.self_attn
            q, k, v = [self.adapter.project(proj, normalized).view(self.copies, 1, -1, attn.head_dim)
                       for proj in (attn.q_proj, attn.k_proj, attn.v_proj)]
            q, k = native_rms(q, attn.q_norm).transpose(1, 2), native_rms(k, attn.k_norm).transpose(1, 2)
            q = effective_bf16(q * cos + rotate_half(q) * sin)
            k = effective_bf16(k * cos + rotate_half(k) * sin)
            keys[:, :, slot:slot + 1] = k.bfloat16()
            values[:, :, slot:slot + 1] = v.transpose(1, 2).bfloat16()
            if self.backend == 'dense':
                output = F.scaled_dot_product_attention(q.bfloat16(), keys, values,
                    attn_mask=allowed, dropout_p=0., enable_gqa=True)
            else:
                output = ps_attention(q.bfloat16(), keys, values, self.indices.new_tensor([position]),
                    self.indices, self.prefix, self.window, 'fa4')
            mid = residual + self.adapter.project(attn.o_proj, output.transpose(1, 2).reshape(self.copies, 1, -1))
            normalized = native_rms(mid, layer.post_attention_layernorm)
            gate, up = [self.adapter.project(proj, normalized) for proj in (layer.mlp.gate_proj, layer.mlp.up_proj)]
            gated = effective_bf16(F.silu(gate) * up)
            hidden = effective_bf16(mid) + self.adapter.project(layer.mlp.down_proj, gated)
        hidden = native_rms(hidden, self.base.model.norm)
        self.next_position += 1
        self.next_slot = self.prefix + (slot + 1 - self.prefix) % self.window
        return self.adapter.linear(hidden[:, 0], self.base.lm_head.weight)

    @property
    def cache_bytes(self):
        return sum(value.numel() * value.element_size() for value in [*self.keys, *self.values])
