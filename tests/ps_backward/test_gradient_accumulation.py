from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.ps_backward.gradient_accumulation import direct_weight_gradients
from prime_rl.trainer.ps_backward.recorded import recorded_linear
from prime_rl.trainer.ps_backward.recorded_qkv import recorded_qkv


@pytest.mark.parametrize("packed", [False, True])
def test_direct_weight_gradients_add_to_existing_buffer_without_autograd_weight_outputs(packed):
    torch.manual_seed(85)
    x = torch.randn(17, 7, dtype=torch.float64, requires_grad=True)
    widths = (5, 3, 3) if packed else (11,)
    weights = [torch.nn.Parameter(torch.randn(width, 7, dtype=torch.float64)) for width in widths]
    cast = torch.cat([w.detach() for w in weights])
    output = x @ cast.T
    seed = torch.randn_like(output)
    initial = [torch.randn_like(w) for w in weights]
    targets = {id(w): value.clone() for w, value in zip(weights, initial, strict=True)}
    indices = {id(w): i for i, w in enumerate(weights)}
    with direct_weight_gradients(None, indices, targets):
        if packed:
            projections = [SimpleNamespace(weight=w, bias=None) for w in weights]
            attn = SimpleNamespace(q_proj=projections[0], k_proj=projections[1], v_proj=projections[2])
            value = recorded_qkv(x, attn, output.detach(), cast)
        else:
            value = recorded_linear(x, weights[0], output.detach(), cast_weight=cast)
        dx, *returned = torch.autograd.grad(value, (x, *weights), seed, allow_unused=True)
    assert all(value is None for value in returned)
    torch.testing.assert_close(dx, seed @ cast)
    offset = 0
    for w, old in zip(weights, initial, strict=True):
        torch.testing.assert_close(targets[id(w)], old + seed[:, offset : offset + w.shape[0]].T @ x)
        offset += w.shape[0]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("widths", [(77,), (65, 31, 31)])
@pytest.mark.parametrize("backend", ["triton", "cublas"])
def test_direct_cuda_accumulation_edges_and_graph_destination_switch(widths, backend):
    if backend == "cublas":
        from prime_rl.trainer.ps_backward.gradient_accumulation_cublas import accumulate
    else:
        from prime_rl.trainer.ps_backward.gradient_accumulation_kernel import accumulate

    torch.manual_seed(91)
    x = torch.randn(191, 79, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(191, sum(widths), device="cuda", dtype=torch.bfloat16)
    banks = [[torch.zeros(width, 79, device="cuda") for width in widths] for _ in range(2)]
    pointers = torch.tensor([v.data_ptr() for v in banks[0]], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        accumulate(dy, x, pointers, tuple(range(len(widths))), widths)
    stream.synchronize()
    for value in banks[0]:
        value.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        accumulate(dy, x, pointers, tuple(range(len(widths))), widths)
    torch.cuda.synchronize()
    for value in banks[0]:
        value.zero_()
    expected = torch.mm(dy.T, x, out_dtype=torch.float32).split(widths)
    for bank in banks:
        pointers.copy_(torch.tensor([v.data_ptr() for v in bank], device="cuda", dtype=torch.int64))
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(bank, expected, strict=True):
            torch.testing.assert_close(actual, 2 * reference, atol=2e-3, rtol=2e-3)
    for a, b in zip(banks[0], banks[1], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
