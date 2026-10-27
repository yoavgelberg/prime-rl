import pytest
import torch

from prime_rl.trainer.ps_backward.fa4_adjoint import backward_fp32


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10, 11),
    reason="FP32 accumulator conversion requires Hopper or Blackwell",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [259, 2049])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fp32_fa4_postprocess_matches_dense_and_supports_cuda_graphs(length, head_dim, causal):
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    torch.manual_seed(444)
    q = torch.randn((1, 4, length, head_dim), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k, v = [
        torch.randn((1, 2, length, head_dim), device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(2)
    ]
    out, lse = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True)[:2]
    do = torch.randn_like(out)
    standard = _flash_attn_bwd(q, k, v, out, do, lse, causal=causal)
    actual = backward_fp32(q, k, v, out, do, lse, causal=causal)
    assert all(value.dtype == torch.float32 for value in actual)
    assert all(not torch.equal(value, value.to(torch.bfloat16).float()) for value in actual)
    for value, reference in zip(actual, standard, strict=True):
        torch.testing.assert_close(value.to(torch.bfloat16), reference, rtol=0.02, atol=0.02)
    query, key, value = [tensor.float().detach().requires_grad_() for tensor in (q, k, v)]
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key.repeat_interleave(2, 2)) * head_dim**-0.5
    if causal:
        mask = torch.arange(length, device="cuda")[:, None] >= torch.arange(length, device="cuda")[None]
        logits = logits.masked_fill(~mask, -torch.inf)
    probabilities = logits.softmax(-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, value.repeat_interleave(2, 2))
    expected = torch.autograd.grad(output, (query, key, value), do.float())
    for value, reference in zip(actual, expected, strict=True):
        relative = ((value - reference).square().sum() / reference.square().sum()).sqrt()
        assert relative.item() < 0.015
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = backward_fp32(q, k, v, out, do, lse, causal=causal)
    graph.replay()
    torch.cuda.synchronize()
    for value, reference in zip(replayed, actual, strict=True):
        torch.testing.assert_close(value, reference, rtol=0.02, atol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="WGMMA accumulator layouts require Hopper",
)
@pytest.mark.parametrize(
    "head_dim,tile,threads,atom,swap,alignment",
    [
        (64, 128, 256, 2, False, 32),
        (96, 64, 128, 1, False, 32),
        (128, 64, 256, 1, False, 32),
        (128, 80, 256, 1, True, 32),
        (160, 96, 256, 2, True, 64),
        (192, 64, 256, 1, False, 64),
        (256, 64, 128, 1, False, 32),
    ],
)
def test_hopper_layout_conversion_matches_upstream_postprocessor(head_dim, tile, threads, atom, swap, alignment):
    import cutlass
    from flash_attn.cute.interface import _bwd_postprocess_convert

    from prime_rl.trainer.ps_backward.fa4_fp32_postprocess import convert_fp32

    torch.manual_seed(732)
    length, heads, batch = 259, 3, 2
    tile_dim = ((head_dim + alignment - 1) // alignment) * alignment
    accum = torch.randn((batch, heads, ((length + tile - 1) // tile) * tile * tile_dim), device="cuda")
    expected = torch.empty((batch, heads, length, head_dim), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    actual = torch.empty_like(expected, dtype=torch.float32)
    options = dict(hdim_multiple_of=alignment, fake_mode=False)
    _bwd_postprocess_convert(
        accum, expected, 0.125, None, None, 90, cutlass.BFloat16, head_dim, tile, threads, atom, swap, **options
    )
    convert_fp32(accum, actual, 0.125, None, None, 90, cutlass.Float32, head_dim, tile, threads, atom, swap, **options)
    torch.testing.assert_close(actual.to(torch.bfloat16), expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="Synthetic SM100 accumulator layouts require Blackwell",
)
@pytest.mark.parametrize("order", ["row_major", "tmem_frag_1cta", "tmem_frag_2cta"])
@pytest.mark.parametrize("head_dim", [96, 128])
def test_fp32_layout_conversion_matches_upstream_postprocessor(order, head_dim):
    import cutlass
    from flash_attn.cute.bwd_accum_tile import AccumTileOrder
    from flash_attn.cute.interface import _bwd_postprocess_convert

    from prime_rl.trainer.ps_backward.fa4_fp32_postprocess import convert_fp32

    torch.manual_seed(732)
    length, tile, heads, batch = 259, 128, 3, 2
    alignment = 64 if order == "row_major" else 32
    tile_dim = ((head_dim + alignment - 1) // alignment) * alignment
    accum = torch.randn((batch, heads, 3 * tile * tile_dim), device="cuda")
    expected = torch.empty((batch, heads, length, head_dim), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    actual = torch.empty_like(expected, dtype=torch.float32)
    options = dict(
        use_2cta_instrs=order == "tmem_frag_2cta",
        accum_order=AccumTileOrder(order),
        hdim_multiple_of=alignment,
        fake_mode=False,
    )
    _bwd_postprocess_convert(
        accum, expected, 0.125, None, None, 100, cutlass.BFloat16, head_dim, tile, 128, 1, False, **options
    )
    convert_fp32(accum, actual, 0.125, None, None, 100, cutlass.Float32, head_dim, tile, 128, 1, False, **options)
    torch.testing.assert_close(actual.to(torch.bfloat16), expected, rtol=0, atol=0)
