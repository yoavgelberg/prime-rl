import pytest
import torch

from prime_rl.trainer.ps_backward.recorded import (
    recorded_add,
    recorded_attention,
    recorded_linear,
    recorded_rms,
    recorded_rope,
    recorded_silu,
)


def test_recorded_norm_rope_and_silu_vjps_match_double_precision():
    from types import SimpleNamespace

    from prime_rl.trainer.ps_backward.model import rotate_half

    torch.manual_seed(17)
    x = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float64, requires_grad=True)
    up = torch.randn_like(x, requires_grad=True)
    angles = torch.randn(3, 5, 4, dtype=torch.float64).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    norm = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight
    rope = norm * cos + rotate_half(norm) * sin
    result = torch.nn.functional.silu(rope) * up
    seed = torch.randn_like(result)
    expected = torch.autograd.grad(result, (x, weight, up), seed)
    normalized = recorded_rms(x, SimpleNamespace(weight=weight, variance_epsilon=1e-6), norm.detach())
    rotated = recorded_rope(normalized, cos, sin, rope.detach())
    actual = torch.autograd.grad(recorded_silu(rotated, up, result.detach()), (x, weight, up), seed)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def test_recorded_residual_mlp_gradient_matches_original_without_forward_gemms(monkeypatch):
    torch.manual_seed(39)
    x = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(13, 7, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(7, 13, dtype=torch.float64, requires_grad=True)
    b = torch.randn(13, dtype=torch.float64, requires_grad=True)
    z = torch.nn.functional.linear(x, w1, b)
    hidden = torch.nn.functional.silu(z)
    y = torch.nn.functional.linear(hidden, w2)
    out = x + y
    seed = torch.randn_like(out)
    expected = torch.autograd.grad(out, (x, w1, w2, b), seed)
    tape = z.detach().clone(), hidden.detach().clone(), y.detach().clone(), out.detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Recorded forward must not issue a forward linear operation")

    monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
    rz = recorded_linear(x, w1, tape[0], bias=b)
    rh = torch.nn.functional.silu(rz)
    ry = recorded_linear(rh, w2, tape[2])
    result = recorded_add(x, ry, tape[3])
    actual = torch.autograd.grad(result, (x, w1, w2, b), seed)
    torch.testing.assert_close(result, tape[3])
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp32_adjoint", [False, True])
def test_recorded_fa4_vjp_uses_saved_normalizers_and_skips_attention_forward(monkeypatch, fp32_adjoint):
    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    from prime_rl.trainer.ps_backward.fa4 import _ps_mask, sparse_ps_plan

    if fp32_adjoint and torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Experimental FP32 accumulator conversion requires Blackwell")
    torch.manual_seed(19)
    q = torch.randn(1, 4, 257, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 385, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qi, ki = torch.arange(128, 385, device="cuda"), torch.arange(385, device="cuda")
    plan = sparse_ps_plan(qi, ki, 37, 129, (256 if torch.cuda.get_device_capability()[0] >= 10 else 128, 128))
    out, lse = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        mask_mod=_ps_mask(37, 129),
        aux_tensors=[qi, ki],
        return_lse=True,
        block_sparse_tensors=BlockSparseTensorsTorch(*plan.forward, block_size=plan.block_size),
        block_sparse_tensors_bwd=BlockSparseTensorsTorch(*plan.backward, block_size=plan.block_size),
    )
    seed = torch.randn_like(out)
    expected = torch.autograd.grad(out, (q, k, v), seed)
    if fp32_adjoint:
        q, k, v = (value.detach().float().requires_grad_() for value in (q, k, v))

    def forbidden(*args, **kwargs):
        raise AssertionError("Recorded attention must not rerun forward")

    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_fwd", forbidden)
    actual_out = recorded_attention(
        q, k, v, out.detach().transpose(1, 2), lse.detach(), plan, 37, 129, fp32_adjoint=fp32_adjoint
    )
    actual = torch.autograd.grad(actual_out, (q, k, v), seed.transpose(1, 2))
    torch.testing.assert_close(actual_out, out.transpose(1, 2).to(actual_out.dtype), atol=0, rtol=0)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.to(e.dtype), e, atol=5e-3, rtol=5e-3)
