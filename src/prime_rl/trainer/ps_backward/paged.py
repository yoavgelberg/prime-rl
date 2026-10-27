"""Compact paged PS attention without copying KV tensors or visiting evicted pages."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_pages(
    table,
    lengths,
    starts,
    prompts,
    output,
    fields,
    table_stride: tl.constexpr,
    output_stride: tl.constexpr,
    field_stride: tl.constexpr,
    columns: tl.constexpr,
    page_size: tl.constexpr,
    window: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    query_length = tl.load(starts + row + 1) - tl.load(starts + row)
    prompt = tl.load(prompts + row)
    prefix_pages = tl.minimum(tl.cdiv(prompt, page_size), tl.cdiv(length, page_size))
    local_page = tl.maximum(prefix_pages, (length - query_length - window + 1) // page_size)
    gap_pages = local_page - prefix_pages
    compact_length = length - gap_pages * page_size
    col = tl.arange(0, BLOCK)
    source = tl.where(col < prefix_pages, col, col + gap_pages)
    valid = (col < columns) & (col * page_size < compact_length)
    pages = tl.load(table + row * table_stride + source, valid, 0)
    tl.store(output + row * output_stride + col, pages, col < columns)
    tl.store(fields + row, prompt)
    tl.store(fields + field_stride + row, prefix_pages * page_size)
    tl.store(fields + 2 * field_stride + row, gap_pages * page_size)
    tl.store(fields + 3 * field_stride + row, length - query_length)
    tl.store(fields + 4 * field_stride + row, compact_length)


def compact_pages(table, lengths, starts, prompts, page_size, window, *, output=None, fields=None):
    """Return remapped page IDs and [prompt, prefix boundary, gap, query start, KV length]."""
    if page_size < 1 or window < 1:
        raise ValueError("Page and window sizes must be positive")
    rows, columns = lengths.numel(), table.shape[1]
    if starts.numel() != rows + 1 or prompts.numel() != rows:
        raise ValueError("Request metadata has inconsistent lengths")
    if output is None:
        output = torch.empty((rows, columns), dtype=torch.int32, device=table.device)
    if fields is None:
        fields = torch.empty((5, rows), dtype=torch.int32, device=table.device)
    if table.is_cuda:
        _compact_pages[(rows,)](
            table,
            lengths,
            starts,
            prompts,
            output,
            fields,
            table.stride(0),
            output.stride(0),
            fields.stride(0),
            columns,
            page_size,
            window,
            triton.next_power_of_2(columns),
        )
    else:
        query_lengths = starts[1:] - starts[:-1]
        prefix_pages = torch.minimum((prompts + page_size - 1) // page_size, (lengths + page_size - 1) // page_size)
        local_pages = torch.maximum(prefix_pages, (lengths - query_lengths - window + 1) // page_size)
        gaps = (local_pages - prefix_pages) * page_size
        fields.copy_(torch.stack((prompts, prefix_pages * page_size, gaps, lengths - query_lengths, lengths - gaps)))
        for row in range(rows):
            col = torch.arange(columns)
            source = torch.where(col < prefix_pages[row], col, col + local_pages[row] - prefix_pages[row])
            valid = col * page_size < fields[4, row]
            output[row].zero_()
            output[row, valid] = table[row, source[valid]]
    return output, fields


@lru_cache(maxsize=16)
def paged_ps_mask(window):
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def read(tensor, batch):
        index = cute.make_rmem_tensor(1, cutlass.Int32)
        index.store(batch)
        value = cute.make_rmem_tensor(1, cutlass.Int32)
        value[0] = tensor[index[0]]
        return value.load()

    @cute.jit
    def mask(batch, head, q_idx, k_idx, seqlen_info, aux_tensors, aux_scalars=None):
        prefix = read(aux_tensors[0], batch)
        prefix_boundary = read(aux_tensors[1], batch)
        gap = read(aux_tensors[2], batch)
        qpos = q_idx + read(aux_tensors[3], batch)
        kpos = k_idx + (k_idx >= prefix_boundary).to(cutlass.Int32) * gap
        return (kpos <= qpos) & ((kpos < prefix) | (kpos >= qpos - window + 1))

    return mask


def paged_ps_attention(
    query,
    key_cache,
    value_cache,
    query_starts,
    page_table,
    fields,
    window,
    max_query_length,
    scale=None,
    *,
    output=None,
    return_lse=False,
):
    from flash_attn.cute.interface import _flash_attn_fwd

    if query.dtype not in (torch.bfloat16, torch.float16) or not query.is_cuda:
        raise ValueError("Paged FA4 PS requires CUDA BF16/FP16")
    result = _flash_attn_fwd(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=query_starts,
        seqused_k=fields[4],
        page_table=page_table,
        max_seqlen_q=max_query_length,
        softmax_scale=scale,
        mask_mod=paged_ps_mask(window),
        aux_tensors=[fields[i] for i in range(4)],
        num_splits=1,
        out=output,
        return_lse=return_lse,
    )
    return (result[0], result[1]) if return_lse else result[0]
