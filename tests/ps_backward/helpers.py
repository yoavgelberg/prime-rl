import types
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from prime_rl.trainer.ps_backward.model import QwenPS, rotate_half
from prime_rl.trainer.ps_backward.types import Trace


def tiny_model(kind="qwen3", depth=3, window=3, dtype=torch.float64, checkpoint_layers=False):
    torch.manual_seed(310)
    config_type, model_type = (Qwen3Config, Qwen3ForCausalLM) if kind == "qwen3" else (Qwen2Config, Qwen2ForCausalLM)
    config = config_type(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        attention_dropout=0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    base = model_type(config).to(dtype)
    if dtype == torch.float64:
        # HF RMSNorm unconditionally casts to FP32, even on a double model.
        # Keep the algebra reference entirely FP64 instead of relaxing its tolerance.
        def double_norm(self, hidden):
            return self.weight * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.variance_epsilon)

        for module in base.modules():
            if hasattr(module, "variance_epsilon"):
                module.forward = types.MethodType(double_norm, module)
    return QwenPS(base, window, backend="dense", checkpoint_layers=checkpoint_layers)


def make_trace(prefix=4, length=19, *, clipped=False, masked=False):
    generator = torch.Generator().manual_seed(170)
    ids = torch.randint(0, 41, (prefix + length,), generator=generator)
    advantages = torch.randn(length, generator=generator, dtype=torch.float64)
    mask = torch.arange(length) % 3 != 1 if masked else None
    old = torch.full((length,), -3.8, dtype=torch.float64) if clipped else None
    # An offset verifies that replay/cropping does not rebase RoPE positions.
    return Trace(ids, prefix, torch.arange(prefix + length) + 123, advantages, old, mask)


def gradients(model):
    return {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}


def assert_grads(actual, expected, rtol=1e-8, atol=1e-10):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=rtol, atol=atol, msg=lambda msg: name + ": " + msg
        )


@torch.no_grad()
def dense_tape(model, trace):
    n, hidden = trace.input_ids.numel() - 1, model.base.config.hidden_size
    x = model.base.model.embed_tokens(trace.input_ids[:-1])
    cos, sin = model.base.model.rotary_emb(x[None], trace.positions[None, :-1])
    ropes = [torch.cat((cos[0, :, : cos.shape[-1] // 2], sin[0, :, : sin.shape[-1] // 2]), -1)] * model.depth
    cos, sin = cos[:, None], sin[:, None]
    recs = []
    indices = torch.arange(n)
    allowed = (indices[None] <= indices[:, None]) & (
        (indices[None] < trace.prefix_length) | (indices[None] >= indices[:, None] - model.window + 1)
    )
    for layer in model.base.model.layers:
        attn, head = layer.self_attn, layer.self_attn.head_dim
        normalized = layer.input_layernorm(x)
        raw_q, raw_k, raw_v = [model.project(proj, normalized) for proj in (attn.q_proj, attn.k_proj, attn.v_proj)]
        q, k = raw_q.reshape(1, n, -1, head), raw_k.reshape(1, n, -1, head)
        q, k = attn.q_norm(q).transpose(1, 2), attn.k_norm(k).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        v = raw_v.reshape(1, n, -1, head).transpose(1, 2)
        groups = q.shape[1] // k.shape[1]
        scores = ((q @ k.repeat_interleave(groups, 1).transpose(-2, -1)) * head**-0.5).masked_fill(~allowed, -torch.inf)
        lse = scores.logsumexp(-1)
        attended = scores.softmax(-1) @ v.repeat_interleave(groups, 1)
        attention = attended.transpose(1, 2).reshape(n, -1)
        mid = x + model.project(attn.o_proj, attention)
        norm2 = layer.post_attention_layernorm(mid)
        gate, up = model.project(layer.mlp.gate_proj, norm2), model.project(layer.mlp.up_proj, norm2)
        gated = torch.nn.functional.silu(gate) * up
        recs.append(
            dict(
                input=x,
                norm1=normalized,
                qkv=torch.cat((raw_q, raw_k, raw_v), -1),
                query=q.transpose(1, 2).reshape(n, -1),
                key=k.transpose(1, 2).reshape(n, -1),
                attention=attention,
                lse=lse[0].T,
                mid=mid,
                norm2=norm2,
                gate_up=torch.cat((gate, up), -1),
                gated=gated,
            )
        )
        x = mid + model.project(layer.mlp.down_proj, gated)
    normalized = model.base.model.norm(x)
    tape = SimpleNamespace(
        requests={"request": 0},
        capacity=n,
        hidden=hidden,
        qwidth=raw_q.shape[-1],
        kwidth=raw_k.shape[-1],
        final=torch.cat((x, normalized), -1),
        logits=model.project(model.base.lm_head, normalized),
        layer=lambda request, index, start, end: {k: v[start:end] for k, v in recs[index].items()},
    )
    return tape, ropes
