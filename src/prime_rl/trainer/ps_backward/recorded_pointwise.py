"""Fused pointwise derivatives of the recorded inference graph."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_backward(
    x,
    dy,
    weight,
    dx,
    partial_dw,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    valid = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    values = tl.load(x + rows[:, None] * WIDTH + cols[None, :], valid, 0).to(tl.float32)
    seed = tl.load(dy + rows[:, None] * WIDTH + cols[None, :], valid, 0).to(tl.float32)
    w = tl.load(weight + cols, cols < WIDTH, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(values * values, 1) / WIDTH + EPS)
    normalized = values * rstd[:, None]
    weighted = seed * w[None, :]
    dot = tl.sum(weighted * normalized, 1) / WIDTH
    gradient = rstd[:, None] * (weighted - normalized * dot[:, None])
    tl.store(dx + rows[:, None] * WIDTH + cols[None, :], gradient, valid)
    dw = tl.sum(seed * normalized, 0)
    tl.store(partial_dw + group * WIDTH + cols, dw, cols < WIDTH)


def rms_backward(inputs, gradient, weight, epsilon):
    width = inputs.shape[-1]
    x = inputs.contiguous().view(-1, width)
    dy = gradient.contiguous().view(-1, width)
    rows = x.shape[0]
    block_rows = 16 if width <= 256 else 4
    groups = triton.cdiv(rows, block_rows)
    dx = torch.empty_like(x)
    partials = torch.empty((groups, width), device=x.device, dtype=torch.float32)
    _rms_backward[(groups,)](
        x,
        dy,
        weight,
        dx,
        partials,
        rows,
        width,
        epsilon,
        block_rows,
        triton.next_power_of_2(width),
        num_warps=4 if width <= 256 else 8,
        enable_fp_fusion=False,
    )
    return dx.view_as(inputs), partials.sum(0).to(weight.dtype)


@triton.jit
def _silu_backward(
    gate,
    up,
    seed,
    dgate,
    dup,
    ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROW: tl.constexpr,
    GCOL: tl.constexpr,
    UROW: tl.constexpr,
    UCOL: tl.constexpr,
    DROW: tl.constexpr,
    DCOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // WIDTH, index % WIDTH
    g = tl.load(gate + row * GROW + col * GCOL, index < ELEMENTS, 0).to(tl.float32)
    u = tl.load(up + row * UROW + col * UCOL, index < ELEMENTS, 0).to(tl.float32)
    dy = tl.load(seed + row * DROW + col * DCOL, index < ELEMENTS, 0).to(tl.float32)
    s = tl.sigmoid(g)
    dg = dy * u * s * (1.0 + g * (1.0 - s))
    du = dy * g * s
    tl.store(dgate + index, dg, index < ELEMENTS)
    tl.store(dup + index, du, index < ELEMENTS)


def silu_backward(gate, up, gradient):
    width = gate.shape[-1]
    g, u, dy = (value.reshape(-1, width) for value in (gate, up, gradient))
    dg = torch.empty(g.shape, device=g.device, dtype=gate.dtype)
    du = torch.empty(u.shape, device=u.device, dtype=up.dtype)
    _silu_backward[(triton.cdiv(g.numel(), 512),)](
        g, u, dy, dg, du, g.numel(), width, *g.stride(), *u.stride(), *dy.stride(), 512, enable_fp_fusion=False
    )
    return dg.view_as(gate), du.view_as(up)


@triton.jit
def _rope_backward(
    seed,
    cos,
    sin,
    output,
    ELEMENTS: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    SEED_STRIDES: tl.constexpr,
    COS_STRIDES: tl.constexpr,
    SIN_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = index % WIDTH
    token = index // WIDTH % TOKENS
    head = index // (WIDTH * TOKENS) % HEADS
    batch = index // (WIDTH * TOKENS * HEADS)
    partner = (col + WIDTH // 2) % WIDTH
    base = batch * SEED_STRIDES[0] + head * SEED_STRIDES[1] + token * SEED_STRIDES[2]
    g = tl.load(seed + base + col * SEED_STRIDES[3], index < ELEMENTS, 0).to(tl.float32)
    other = tl.load(seed + base + partner * SEED_STRIDES[3], index < ELEMENTS, 0).to(tl.float32)
    c = tl.load(
        cos + batch * COS_STRIDES[0] + head * COS_STRIDES[1] + token * COS_STRIDES[2] + col * COS_STRIDES[3],
        index < ELEMENTS,
        0,
    ).to(tl.float32)
    s = tl.load(
        sin + batch * SIN_STRIDES[0] + head * SIN_STRIDES[1] + token * SIN_STRIDES[2] + partner * SIN_STRIDES[3],
        index < ELEMENTS,
        0,
    ).to(tl.float32)
    value = g * c + tl.where(col < WIDTH // 2, other * s, -other * s)
    tl.store(output + index, value, index < ELEMENTS)


def rope_backward(gradient, cos, sin, dtype):
    if gradient.ndim != 4 or gradient.shape[-1] % 2:
        raise ValueError("Fused RoPE derivatives require [batch, heads, tokens, even head width]")
    cos, sin = (value.expand_as(gradient) for value in (cos, sin))
    output = torch.empty(gradient.shape, device=gradient.device, dtype=dtype)
    _rope_backward[(triton.cdiv(gradient.numel(), 512),)](
        gradient,
        cos,
        sin,
        output,
        gradient.numel(),
        *gradient.shape[1:],
        gradient.stride(),
        cos.stride(),
        sin.stride(),
        512,
        enable_fp_fusion=False,
    )
    return output
