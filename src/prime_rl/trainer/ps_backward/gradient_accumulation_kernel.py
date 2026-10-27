"""Tensor-core weight-gradient GEMM with direct FP32 accumulation."""

import triton
import triton.language as tl


@triton.jit
def _accumulate(
    DY,
    X,
    POINTERS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    DY0: tl.constexpr,
    DY1: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    IDS: tl.constexpr,
    WIDTHS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    result = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + reduction
        a = tl.load(
            DY + rows[:, None] * DY1 + k[None, :] * DY0,
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        )
        b = tl.load(
            X + k[:, None] * X0 + cols[None, :] * X1,
            (k[:, None] < K) & (cols[None, :] < N),
            0,
        )
        result += tl.dot(a, b, input_precision="ieee")
    address = tl.full((BM,), 0, tl.int64)
    local_rows = rows
    offset: tl.constexpr = 0
    for i in tl.static_range(len(IDS)):
        pointer = tl.load(POINTERS + IDS[i])
        selected = (rows >= offset) & (rows < offset + WIDTHS[i])
        address = tl.where(selected, pointer, address)
        local_rows = tl.where(selected, rows - offset, local_rows)
        offset += WIDTHS[i]
    destination = address.to(tl.pointer_type(tl.float32))[:, None] + local_rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(destination, tl.load(destination, mask, 0) + result, mask)


def accumulate(dy, x, pointers, indices, widths):
    _accumulate[(triton.cdiv(dy.shape[1], 32), triton.cdiv(x.shape[1], 64))](
        dy,
        x,
        pointers,
        dy.shape[1],
        x.shape[1],
        x.shape[0],
        *dy.stride(),
        *x.stride(),
        indices,
        widths,
        32,
        64,
        32,
        num_warps=4,
    )
