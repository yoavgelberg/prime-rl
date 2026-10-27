"""FP32 conversion of the pinned FA4 Hopper and Blackwell accumulator layouts.

Layout definitions derive from flash-attn-4 4.0.0b36 bwd_accum_tile.py.
Copyright (c) 2025, Tri Dao. See FA4_LICENSE.txt.
"""

import triton
import triton.language as tl


@triton.jit
def _convert(
    accum,
    output,
    scale,
    ab,
    ah,
    af: tl.constexpr,
    ob,
    os,
    oh,
    od: tl.constexpr,
    length,
    dim: tl.constexpr,
    tile_m: tl.constexpr,
    tile_d: tl.constexpr,
    order: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // dim, index % dim
    local_row = row % tile_m
    if order == "row_major":
        local = local_row * tile_d + col
    elif order == "tmem_frag_1cta":
        local = local_row * 4 + col % 4 + (col // 4) * (4 * tile_m)
    else:
        local = (local_row % 64) * 4 + (local_row // 64) * (64 * tile_d) + col % 4
        local += ((col // 4) % (tile_d // 8)) * 512 + (col // (tile_d // 2)) * 256
    source = (row // tile_m) * (tile_m * tile_d) + local
    head, batch = tl.program_id(1), tl.program_id(2)
    value = tl.load(accum + batch * ab + head * ah + source * af, row < length, 0)
    tl.store(output + batch * ob + row * os + head * oh + col * od, value * scale, row < length)


def convert_fp32(
    accum,
    output,
    scale,
    cu_seqlens,
    seqused,
    arch,
    dtype,
    hdim,
    block_size,
    threads,
    atom_layout,
    swap_ab,
    use_2cta_instrs=False,
    cluster_size=1,
    cu_total_m_blocks=None,
    sink_tensors=None,
    hdim_multiple_of=32,
    accum_order=None,
    *,
    fake_mode,
):
    import cutlass
    import torch
    from flash_attn.cute.bwd_accum_tile import AccumTileOrder

    if cu_seqlens is not None or seqused is not None or cu_total_m_blocks is not None or sink_tensors is not None:
        raise ValueError("FP32 PS conversion supports fixed batches without learnable sinks")
    if dtype != cutlass.Float32 or arch // 10 not in (9, 10, 11):
        raise ValueError("FP32 PS conversion requires Hopper or Blackwell accumulator layouts")
    if output.dtype != torch.float32 or accum.dtype != torch.float32:
        raise TypeError("FP32 PS conversion requires Float32 accumulators and output")
    if output.ndim != 4 or accum.ndim != 3 or output.shape[-1] != hdim:
        raise ValueError("Unexpected FA4 gradient or accumulator dimensions")
    if arch // 10 != 9 and not isinstance(accum_order, AccumTileOrder):
        raise ValueError("FA4 must specify its accumulator storage order")
    tile_d = triton.cdiv(hdim, hdim_multiple_of) * hdim_multiple_of
    batch, length, heads, _ = output.shape
    required = triton.cdiv(length, block_size) * block_size * tile_d
    if accum.shape[:2] != (batch, heads) or accum.shape[2] < required:
        raise ValueError("FA4 accumulator does not cover the output tiles")
    if arch // 10 == 9:
        if accum_order is not None or use_2cta_instrs or cluster_size != 1:
            raise ValueError("Unexpected Hopper accumulator configuration")
        from .fa4_hopper_postprocess import convert_hopper

        convert_hopper(
            accum, output, scale, hdim, block_size, threads, atom_layout, swap_ab, hdim_multiple_of,
            fake_mode=fake_mode,
        )
        return
    if accum_order == AccumTileOrder.TMEM_FRAG_2CTA and (block_size != 128 or tile_d % 8):
        raise ValueError("Invalid FA4 two-CTA tile geometry")
    if not fake_mode:
        _convert[(triton.cdiv(length * hdim, 512), heads, batch)](
            accum,
            output,
            scale,
            *accum.stride(),
            *output.stride(),
            length,
            hdim,
            block_size,
            tile_d,
            accum_order.value,
            512,
        )
