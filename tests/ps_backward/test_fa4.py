import pytest
import torch

from prime_rl.trainer.ps_backward.attention import dense_ps_attention, ps_attention
from prime_rl.trainer.ps_backward.fa4 import _ReplicatePrefix, aligned_chunk_plan, sparse_ps_plan


def _unpack(counts, indices, ncolumns):
    result = torch.zeros((counts.shape[-1], ncolumns), dtype=torch.bool)
    for row, count in enumerate(counts[0, 0].tolist()):
        result[row, indices[0, 0, row, :count]] = True
    return result


@pytest.mark.parametrize("prefix,window", [(0, 1), (7, 9), (15, 3)])
def test_translating_complete_nonprefix_interval_preserves_ps_edges(prefix, window):
    positions = torch.arange(prefix + 31)

    def allowed(ix):
        return (ix[None, :] <= ix[:, None]) & ((ix[None, :] < prefix) | (ix[None, :] >= ix[:, None] - window + 1))

    for gap in (0, 1, 1000, 262144):
        original = positions.clone()
        original[prefix:] += gap
        torch.testing.assert_close(allowed(original), allowed(positions))


@pytest.mark.parametrize("start", [59, 1083])
def test_aligned_chunk_padding_preserves_all_attention_gradients(start):
    prefix, window, end = 59, 257, start + 73
    history = max(prefix, start - window + 1)
    qi = torch.arange(start, end)
    ki = torch.cat((torch.arange(prefix), torch.arange(history, end)))
    plan = aligned_chunk_plan(prefix, start, end, history, window, "cpu", (64, 32))
    torch.manual_seed(321)
    q = torch.randn(1, 4, qi.numel(), 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, ki.numel(), 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qp = torch.nn.functional.pad(q, (0, 0, plan.q_padding, plan.q_tail_padding))
    kp, vp = (
        (k, v)
        if plan.source_indices is None
        else (_ReplicatePrefix.apply(k, plan.source_indices), _ReplicatePrefix.apply(v, plan.source_indices))
    )
    actual = dense_ps_attention(qp, kp, vp, plan.q_indices, plan.k_indices, prefix, window)[
        :, :, plan.q_padding : plan.q_padding + qi.numel()
    ]
    seed = torch.randn_like(actual)
    actual.backward(seed)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    expected = dense_ps_attention(qr, kr, vr, qi, ki, prefix, window)
    expected.backward(seed)
    for got, want in zip((actual, q.grad, k.grad, v.grad), (expected, qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-11)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fa4_aligned_chunk_padding_gradient():
    from prime_rl.trainer.ps_backward.fa4 import fa4_ps_attention

    prefix, start, end, window = 59, 4155, 4668, 257
    history = start - window + 1
    qi = torch.arange(start, end, device="cuda")
    ki = torch.cat((torch.arange(prefix, device="cuda"), torch.arange(history, end, device="cuda")))
    tile = 128 if torch.cuda.get_device_capability()[0] == 9 else 256
    plan = aligned_chunk_plan(prefix, start, end, history, window, "cuda", (tile, 128))
    torch.manual_seed(322)
    q = torch.randn(1, 8, qi.numel(), 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, ki.numel(), 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = fa4_ps_attention(q, k, v, qi, ki, prefix, window, plan)
    seed = torch.randn_like(actual)
    actual.backward(seed)
    qr, kr, vr = [t.detach().float().requires_grad_() for t in (q, k, v)]
    expected = dense_ps_attention(qr, kr, vr, qi, ki, prefix, window)
    expected.backward(seed.float())
    for got, want in zip((actual, q.grad, k.grad, v.grad), (expected, qr.grad, kr.grad, vr.grad), strict=True):
        assert ((got.float() - want).norm() / want.norm().clamp_min(1e-12)).item() < 1e-2


@pytest.mark.parametrize(
    "prefix,window,nq,nk", [(0, 1, 93, 120), (35, 17, 97, 167), (65, 129, 193, 270), (190, 32, 190, 190)]
)
@pytest.mark.parametrize("block", [(32, 32), (128, 128), (256, 128)])
def test_fa4_sparse_metadata_covers_exact_mask(prefix, window, nq, nk, block):
    ki = torch.arange(nk)
    ki[prefix:] += 1000
    qi = ki[-nq:]
    plan = sparse_ps_plan(qi, ki, prefix, window, block)
    assert all(t.stride(-1) == 1 for t in (*plan.forward, *plan.backward))
    nbq, nbk = (nq + block[0] - 1) // block[0], (nk + block[1] - 1) // block[1]
    partial = _unpack(*plan.forward[:2], nbk)
    full = _unpack(*plan.forward[2:], nbk)
    assert not (partial & full).any()
    assert torch.equal(partial.T, _unpack(*plan.backward[:2], nbq))
    assert torch.equal(full.T, _unpack(*plan.backward[2:], nbq))
    expected = (ki[None, :] <= qi[:, None]) & ((ki[None, :] < prefix) | (ki[None, :] >= qi[:, None] - window + 1))
    for m in range(nbq):
        for n in range(nbk):
            tile = expected[m * block[0] : (m + 1) * block[0], n * block[1] : (n + 1) * block[1]]
            assert not tile.any() or partial[m, n] or full[m, n]
            assert not full[m, n] or tile.all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "prefix,window,nq,nk,hq,hk,dim",
    [
        (0, 1, 93, 120, 4, 2, 32),
        (35, 17, 97, 167, 4, 2, 32),
        (65, 129, 193, 270, 16, 8, 128),
        (190, 32, 190, 190, 16, 8, 128),
        (59, 2048, 2112, 2304, 16, 8, 128),
    ],
)
def test_fa4_operator_gradient(prefix, window, nq, nk, hq, hk, dim):
    pytest.importorskip("flash_attn.cute")
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("Hopper/Blackwell test")
    torch.manual_seed(155)
    ki = torch.arange(nk, device="cuda")
    ki[prefix:] += 1000
    qi = ki[-nq:]
    q = torch.randn(1, hq, nq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, hk, nk, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = ps_attention(q, k, v, qi, ki, prefix, window, "fa4")
    seed = torch.randn_like(output)
    output.backward(seed)
    qr, kr, vr = [t.detach().float().requires_grad_() for t in (q, k, v)]
    expected = dense_ps_attention(qr, kr, vr, qi, ki, prefix, window)
    expected.backward(seed.float())
    for i, (got, want) in enumerate(
        zip((output, q.grad, k.grad, v.grad), (expected, qr.grad, kr.grad, vr.grad), strict=True)
    ):
        if prefix == 0 and window == 1 and i in (1, 2):
            assert got.abs().max() < 1e-5
            continue
        assert ((got.float() - want).norm() / want.norm().clamp_min(1e-12)).item() < 1e-2
