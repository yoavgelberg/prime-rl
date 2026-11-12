import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.ps_concentration.native_teacher import NativeTeacher, effective_bf16, native_rms


def test_effective_precision_keeps_fp32_master_derivatives():
    x = torch.tensor([.7317, 1.2751, .9913], requires_grad=True)
    actual = effective_bf16(x)
    torch.testing.assert_close(actual, x.detach().bfloat16().float(), rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_norm_uses_unrounded_sum_and_effective_native_weight():
    norm = torch.nn.RMSNorm(4, eps=1e-6)
    norm.variance_epsilon = 1e-6
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([.7317, 1.2751, .9913, .3459]))
    x = torch.tensor([[1.003, 2.007, 3.011, 4.015]], requires_grad=True)
    expected = (x.detach() * torch.rsqrt(x.detach().square().mean(-1, keepdim=True) + 1e-6)
                * norm.weight.detach().bfloat16().float()).bfloat16().float()
    actual = native_rms(x, norm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad.dtype == norm.weight.grad.dtype == torch.float32
    assert bool(torch.isfinite(x.grad).all())


def test_teacher_matches_independent_two_residual_bf16_reference_across_evictions(monkeypatch):
    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=73, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
        attention_dropout=0.)
    base = Qwen3ForCausalLM(config).float().eval()
    adapter = NativeTeacher(base, 16, backend='dense', checkpoint_layers=True)
    def linear_cpu(inputs, weight, bias=None):
        return F.linear(inputs.bfloat16(), weight.bfloat16(), None if bias is None else bias.bfloat16()).float()
    monkeypatch.setattr(adapter.projections, 'linear', linear_cpu)
    ids, indices = torch.randint(0, 73, (128,)), torch.arange(128)
    actual = adapter.segment(ids, indices, indices, 4)

    def norm(values, module):
        values = values.float()
        inv = torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
        return (values * inv * module.weight.detach().bfloat16().float()).bfloat16()

    def linear(module, values):
        return F.linear(values.bfloat16(), module.weight.detach().bfloat16(),
                        None if module.bias is None else module.bias.detach().bfloat16())

    with torch.no_grad():
        hidden, residual = base.model.embed_tokens(ids[None]).bfloat16(), None
        cos, sin = base.model.rotary_emb(hidden, indices[None])
        cos, sin = cos.bfloat16().float()[:, None], sin.bfloat16().float()[:, None]
        allowed = (indices[None] <= indices[:, None]) & ((indices[None] < 4) | (indices[None] >= indices[:, None] - 15))
        for layer in base.model.layers:
            raw = hidden.float() if residual is None else hidden.float() + residual.float()
            normalized, residual = norm(raw, layer.input_layernorm), raw.bfloat16()
            attn = layer.self_attn
            q, k, v = [linear(getattr(attn, name), normalized).view(1, 128, -1, 16)
                       for name in ('q_proj', 'k_proj', 'v_proj')]
            q, k = norm(q, attn.q_norm).transpose(1, 2).float(), norm(k, attn.k_norm).transpose(1, 2).float()
            def rotate(value):
                a, b = value.chunk(2, dim=-1)
                return torch.cat([-b, a], dim=-1)
            q, k = (q * cos + rotate(q) * sin).bfloat16(), (k * cos + rotate(k) * sin).bfloat16()
            attended = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                v.transpose(1, 2).repeat_interleave(2, 1), attn_mask=allowed, dropout_p=0.)
            projected = linear(attn.o_proj, attended.transpose(1, 2).reshape(1, 128, 64))
            raw = projected.float() + residual.float()
            normalized, residual = norm(raw, layer.post_attention_layernorm), raw.bfloat16()
            gate, up = linear(layer.mlp.gate_proj, normalized), linear(layer.mlp.up_proj, normalized)
            gated = (F.silu(gate.float()) * up.float()).bfloat16()
            hidden = linear(layer.mlp.down_proj, gated)
        expected = norm(hidden.float() + residual.float(), base.model.norm).squeeze(0).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    labels = ids.roll(-1)
    loss = -adapter.logps(actual, labels, block_size=32).mean()
    loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in base.parameters())
