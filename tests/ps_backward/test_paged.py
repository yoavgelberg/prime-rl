import pytest
import torch

from prime_rl.trainer.ps_backward.paged import compact_pages, paged_ps_attention


@pytest.mark.parametrize("page,window", [(4, 7), (8, 8), (16, 19)])
def test_compaction_preserves_exact_ps_keys_across_prefill_and_decode(page, window):
    prompts = torch.tensor([1, 5, 17, 17, 5, 1], dtype=torch.int32)
    lengths = torch.tensor([1, 3, 17, 80, 97, 99], dtype=torch.int32)
    query_lengths = torch.tensor([1, 3, 9, 7, 1, 5], dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), query_lengths.cumsum(0).int()))
    columns = (100 + page - 1) // page
    table = torch.arange(6 * columns, dtype=torch.int32).reshape(6, columns)
    output, fields = compact_pages(table, lengths, starts, prompts, page, window)
    for row in range(6):
        prefix, boundary, gap, qstart, count = fields[:, row].tolist()
        compact = torch.arange(count)
        original = compact + (compact >= boundary) * gap
        assert torch.unique(original).numel() == count
        actual_pages = output[row, compact // page]
        assert torch.equal(actual_pages, table[row, original // page])
        for q in range(qstart, int(lengths[row])):
            allowed = (original <= q) & ((original < prefix) | (original >= q - window + 1))
            all_keys = torch.arange(lengths[row])
            expected = all_keys[(all_keys <= q) & ((all_keys < prefix) | (all_keys >= q - window + 1))]
            assert torch.equal(original[allowed], expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("query_lengths", [[1, 1, 1], [7, 19, 3]])
def test_paged_fa4_matches_dense_ps_with_evicted_pages_and_cuda_graph(query_lengths):
    torch.manual_seed(71)
    device, page, window = "cuda", 128, 193
    lengths = torch.tensor([97, 613, 1101], dtype=torch.int32, device=device)
    prompts = torch.tensor([61, 131, 17], dtype=torch.int32, device=device)
    starts = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    table = torch.randperm(27, device=device).int().reshape(3, 9)
    k = torch.randn(27, page, 2, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(sum(query_lengths), 4, 64, device=device, dtype=torch.bfloat16)
    reference_pages, reference_fields = compact_pages(
        table.cpu(), lengths.cpu(), starts.cpu(), prompts.cpu(), page, window
    )
    pages, fields = compact_pages(table, lengths, starts, prompts, page, window)
    assert torch.equal(pages.cpu(), reference_pages)
    assert torch.equal(fields.cpu(), reference_fields)
    references = []
    for row, count in enumerate(query_lengths):
        length, prefix = int(lengths[row]), int(prompts[row])
        keys = k[table[row].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1).float()
        values = v[table[row].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1).float()
        queries = q[int(starts[row]) : int(starts[row + 1])].float()
        qp, kp = torch.arange(length - count, length, device=device), torch.arange(length, device=device)
        mask = (kp[None] <= qp[:, None]) & ((kp[None] < prefix) | (kp[None] >= qp[:, None] - window + 1))
        scores = torch.einsum("qhd,khd->hqk", queries, keys) / 8
        references.append(torch.einsum("hqk,khd->qhd", scores.masked_fill(~mask, -torch.inf).softmax(-1), values))
    expected = torch.cat(references)
    for row in range(3):
        boundary, gap = int(fields[1, row]) // page, int(fields[2, row]) // page
        evicted = table[row, boundary : boundary + gap].long()
        k[evicted] = torch.nan
        v[evicted] = torch.nan
    for _ in range(2):
        actual = paged_ps_attention(q, k, v, starts, pages, fields, window, max(query_lengths))
    torch.testing.assert_close(actual.float(), expected, atol=4e-3, rtol=4e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = paged_ps_attention(q, k, v, starts, pages, fields, window, max(query_lengths))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, atol=4e-3, rtol=4e-3)
    q.mul_(0.5)
    changed = paged_ps_attention(q, k, v, starts, pages, fields, window, max(query_lengths))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, changed)
