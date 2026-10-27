"""Shared lossless SwiGLU value reconstruction and activation recording."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_record(
    inputs,
    output,
    destination,
    slots,
    input_stride: tl.constexpr,
    WIDTH: tl.constexpr,
    RECORD: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    GATE_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    SAVE_OUTPUT: tl.constexpr = True,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(inputs + token * input_stride + cols, cols < WIDTH, 0)
    up = tl.load(inputs + token * input_stride + WIDTH + cols, cols < WIDTH, 0)
    gated = (gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32)) * up.to(tl.float32)).to(output.dtype.element_ty)
    tl.store(output + token * WIDTH + cols, gated, cols < WIDTH)
    if RECORD:
        slot = tl.load(slots + token)
        valid = (slot >= 0) & (cols < WIDTH)
        tl.store(destination + slot * ROW_WIDTH + GATE_OFFSET + cols, gate, valid)
        tl.store(destination + slot * ROW_WIDTH + GATE_OFFSET + WIDTH + cols, up, valid)
        if SAVE_OUTPUT:
            tl.store(destination + slot * ROW_WIDTH + GATE_OFFSET + 2 * WIDTH + cols, gated, valid)


def silu_value(inputs):
    width = inputs.shape[-1] // 2
    output = torch.empty((inputs.shape[0], width), device=inputs.device, dtype=inputs.dtype)
    _silu_record[(inputs.shape[0], triton.cdiv(width, 512))](
        inputs, output, output, output, inputs.stride(0), width, False, width, 0, 512
    )
    return output
