from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.ps_backward.recorded import (
    recorded_rms,
    recorded_rope,
    recorded_silu,
    unfused_recorded_pointwise,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def check_gradients(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == reference.dtype
        relative = (value.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-20)
        assert relative < (5e-4 if value.dtype == torch.bfloat16 else 2e-5), relative.item()


@pytest.mark.parametrize("shape", [(3, 37), (1024, 2048), (1, 257, 16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_rms_matches_unfused_and_double_precision(monkeypatch, shape, dtype):
    torch.manual_seed(172)
    x = torch.randn((*shape[:-1], 2 * shape[-1]), device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    weight = torch.randn(shape[-1], device="cuda", requires_grad=True)
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    output = torch.empty(shape, device="cuda")
    seed = torch.randn_like(output)
    xd, wd = (value.detach().double().requires_grad_() for value in (x, weight))
    reference = torch.autograd.grad(
        xd * (xd.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * wd, (xd, wd), seed.double()
    )
    expected = tuple(value.to(target.dtype) for value, target in zip(reference, (x, weight), strict=True))
    with unfused_recorded_pointwise():
        plain = torch.autograd.grad(recorded_rms(x, norm, output), (x, weight), seed)
    actual = torch.autograd.grad(recorded_rms(x, norm, output), (x, weight), seed)
    check_gradients(actual, expected)
    check_gradients(actual, plain)


@pytest.mark.parametrize("shape", [(3, 17), (1024, 6144)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_silu_matches_unfused_and_double_precision(monkeypatch, shape, dtype):
    torch.manual_seed(93)
    pair = torch.randn((shape[0], 2 * shape[1]), device="cuda", dtype=dtype)
    gate, up = (value.detach().requires_grad_() for value in pair.chunk(2, -1))
    output = torch.empty(shape, device="cuda", dtype=dtype)
    seed = torch.randn_like(output)
    expected = torch.autograd.grad(torch.nn.functional.silu(gate.double()) * up.double(), (gate, up), seed.double())
    with unfused_recorded_pointwise():
        plain = torch.autograd.grad(recorded_silu(gate, up, output), (gate, up), seed)
    actual = torch.autograd.grad(recorded_silu(gate, up, output), (gate, up), seed)
    check_gradients(actual, expected)
    check_gradients(actual, plain)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
def test_fused_rope_handles_strides_broadcast_and_unequal_halves(monkeypatch, dtype, broadcast):
    from prime_rl.trainer.ps_backward.model import rotate_half

    torch.manual_seed(74)
    x = torch.randn((2, 257, 8, 128), device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    shape = (1, 1, 257, 128) if broadcast else x.shape
    cos, sin = (torch.randn(shape, device="cuda") for _ in range(2))
    seed = torch.randn((2, 257, 8, 128), device="cuda", dtype=dtype).transpose(1, 2)
    output = torch.empty_like(x)
    xd = x.detach().double().requires_grad_()
    expected = tuple(
        value.to(x.dtype)
        for value in torch.autograd.grad(xd * cos.double() + rotate_half(xd) * sin.double(), (xd,), seed.double())
    )
    with unfused_recorded_pointwise():
        plain = torch.autograd.grad(recorded_rope(x, cos, sin, output), (x,), seed)
    actual = torch.autograd.grad(recorded_rope(x, cos, sin, output), (x,), seed)
    check_gradients(actual, expected)
    check_gradients(actual, plain)


def test_fused_vjps_replay_cuda_graph_with_changed_inputs(monkeypatch):
    torch.manual_seed(7)
    x, up = (torch.randn(1, 4, 257, 128, device="cuda", requires_grad=True) for _ in range(2))
    weight = torch.randn(128, device="cuda", requires_grad=True)
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    cos, sin = (torch.randn(1, 1, 257, 128, device="cuda") for _ in range(2))
    seed = torch.randn_like(x)
    recorded = torch.empty_like(x)

    def operation():
        a = recorded_rms(x, norm, recorded)
        b = recorded_rope(a, cos, sin, recorded)
        return torch.autograd.grad(recorded_silu(b, up, recorded), (x, weight, up), seed)

    recorded.normal_()
    operation()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = operation()
    for _ in range(2):
        with unfused_recorded_pointwise():
            expected = operation()
        graph.replay()
        check_gradients(actual, expected)
        with torch.no_grad():
            x.mul_(0.7)
            up.add_(0.1)
            seed.neg_()
