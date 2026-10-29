"""Incremental Qwen decoding with a persistent, bounded KV cache."""
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from prime_rl.trainer.ps_backward.model import rotate_half
from .linear import ProjectionCache


class StreamingQwen:
    def __init__(self, base, batch_size, prefix_length, window, attention, cap):
        self.base = base
        self.prefix = prefix_length
        self.window = window
        self.attention = attention
        self.capacity = cap if attention == 'full' else prefix_length + window
        self.batch_size = batch_size
        self.position = 0
        self.consumed = 0
        self.projections = ProjectionCache()
        self.graphs = {}
        self.use_graphs = os.environ.get('PS_CONCENTRATION_GRAPHS', '0') == '1'
        self.graph_pool = None
        self.keys, self.values = [], []
        for layer in base.model.layers:
            attn = layer.self_attn
            kv_heads = attn.k_proj.weight.shape[0] // attn.head_dim
            shape = (batch_size, self.capacity, kv_heads, attn.head_dim)
            self.keys.append(torch.zeros(shape, device=base.device, dtype=torch.bfloat16).transpose(1,2))
            self.values.append(torch.zeros(shape, device=base.device, dtype=torch.bfloat16).transpose(1,2))

    @property
    def cache_bytes(self):
        used = min(self.consumed, self.capacity)
        return sum(k.shape[0] * k.shape[1] * used * k.shape[-1] * k.element_size() * 2 for k in self.keys)

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def step(self, tokens):
        i = self.position
        if self.attention == 'full' and i >= self.capacity:
            raise RuntimeError('Transcript capacity exceeded; this is an experiment error, not task failure')
        n = self.consumed
        slot = n if self.attention == 'full' or n < self.prefix else self.prefix + (n - self.prefix) % self.window
        positions = torch.full((len(tokens), 1), i, device=tokens.device, dtype=torch.long)
        slot_tensor = torch.tensor([slot], device=tokens.device)
        if self.use_graphs and (self.attention == 'full' or n >= self.capacity):
            used = self.capacity if self.attention == 'ps' else min(self.capacity, ((n+1+2047)//2048)*2048)
            if used not in self.graphs:
                self._capture(tokens, positions, slot_tensor, used)
            graph, static_tokens, static_positions, static_slot, output = self.graphs[used]
            static_tokens.copy_(tokens)
            static_positions.copy_(positions)
            static_slot.copy_(slot_tensor)
            graph.replay()
            logits = output.clone()
        else:
            used = min(n+1, self.capacity) if self.attention == 'ps' else min(self.capacity, ((n+1+2047)//2048)*2048)
            logits = self._forward(tokens, positions, slot_tensor, used)
        self.position += 1
        self.consumed += 1
        return logits

    def _capture(self, tokens, positions, slot, used):
        static_tokens, static_positions, static_slot = tokens.clone(), positions.clone(), slot.clone()
        warm = torch.cuda.Stream(device=tokens.device)
        warm.wait_stream(torch.cuda.current_stream(tokens.device))
        with torch.cuda.stream(warm):
            for _ in range(2):
                self._forward(static_tokens, static_positions, static_slot, used)
        torch.cuda.current_stream(tokens.device).wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=warm):
            output = self._forward(static_tokens, static_positions, static_slot, used)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[used] = graph, static_tokens, static_positions, static_slot, output

    def _forward(self, tokens, positions, slot, used):
        hidden = self.base.model.embed_tokens(tokens[:, None])
        cos, sin = self.base.model.rotary_emb(hidden, positions)
        cos, sin = cos[:, None], sin[:, None]
        for layer, keys, values in zip(self.base.model.layers, self.keys, self.values, strict=True):
            normalized = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q, k, v = [self.projections.project(proj, normalized).view(len(tokens), 1, -1, attn.head_dim).transpose(1, 2)
                       for proj in (attn.q_proj, attn.k_proj, attn.v_proj)]
            if hasattr(attn, 'q_norm'):
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            keys.index_copy_(2, slot, k)
            values.index_copy_(2, slot, v)
            # Only causal visible keys are present. Ring order does not affect attention.
            if self.attention == 'full':
                from .decode_fa4 import full_decode
                output = full_decode(q, keys[:, :, :used], values[:, :, :used], positions)
            else:
                output = F.scaled_dot_product_attention(q, keys[:, :, :used], values[:, :, :used],
                                                        enable_gqa=True, dropout_p=0.)
            hidden = hidden + self.projections.project(attn.o_proj, output.transpose(1, 2).reshape(len(tokens), 1, -1))
            x = layer.post_attention_layernorm(hidden)
            gated = layer.mlp.act_fn(self.projections.project(layer.mlp.gate_proj, x)) * self.projections.project(layer.mlp.up_proj, x)
            hidden = hidden + self.projections.project(layer.mlp.down_proj, gated)
        return self.projections.project(self.base.lm_head, self.base.model.norm(hidden)[:, 0])

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def prefill(self, tokens, positions=None):
        if self.consumed:
            raise ValueError('Cold prefill requires an empty cache')
        count = tokens.shape[1]
        if count > self.capacity:
            raise ValueError('Cold prefill exceeds cache capacity')
        if positions is None:
            positions = torch.arange(count, device=tokens.device)
        assert len(positions) == count and bool((positions[1:] > positions[:-1]).all())
        hidden = self.base.model.embed_tokens(tokens)
        cos, sin = self.base.model.rotary_emb(hidden, positions[None].expand(len(tokens), -1))
        cos, sin = cos[:, None], sin[:, None]
        allowed = positions[None] <= positions[:, None]
        if self.attention == 'ps':
            allowed &= (positions[None] < self.prefix) | (positions[None] >= positions[:, None] - self.window + 1)
        for layer, keys, values in zip(self.base.model.layers, self.keys, self.values, strict=True):
            normalized = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q, k, v = [self.projections.project(proj, normalized).view(len(tokens), count, -1, attn.head_dim).transpose(1, 2)
                       for proj in (attn.q_proj, attn.k_proj, attn.v_proj)]
            if hasattr(attn, 'q_norm'):
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            keys[:, :, :count].copy_(k)
            values[:, :, :count].copy_(v)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True, dropout_p=0.)
            hidden = hidden + self.projections.project(attn.o_proj, output.transpose(1, 2).reshape(len(tokens), count, -1))
            x = layer.post_attention_layernorm(hidden)
            gated = layer.mlp.act_fn(self.projections.project(layer.mlp.gate_proj, x)) * self.projections.project(layer.mlp.up_proj, x)
            hidden = hidden + self.projections.project(layer.mlp.down_proj, gated)
        self.consumed, self.position = count, int(positions[-1]) + 1
        return self.projections.project(self.base.lm_head, self.base.model.norm(hidden)[:, -1])


def load_model(path, device='cuda', train=False):
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if (Path(path)/'weights-retired.json').exists():
        raise ValueError(f'Unselected checkpoint weights retired; inspect {path}/weights-retired.json for retained checkpoints')
    base = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                              attn_implementation='sdpa').to(device)
    base.eval()
    if not train:
        base.requires_grad_(False)
    return base
