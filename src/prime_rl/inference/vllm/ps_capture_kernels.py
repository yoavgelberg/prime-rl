"""Record activations in RMSNorm and SwiGLU launches already needed for inference."""

import torch
import triton
import triton.language as tl

from prime_rl.trainer.ps_backward.recording_kernels import _silu_record


@triton.jit
def _rms_record(
    inputs,
    residual,
    weight,
    output,
    residual_output,
    destination,
    raw_destination,
    slots,
    extras,
    extra_strides: tl.constexpr,
    extra_widths: tl.constexpr,
    lse,
    lse_destination,
    input_stride: tl.constexpr,
    residual_stride: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    RECORD: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    DATA_OFFSET: tl.constexpr,
    NORM_OFFSET: tl.constexpr,
    EXTRA_OFFSET: tl.constexpr,
    LSE_HEADS: tl.constexpr,
    LSE_TOKENS: tl.constexpr,
    LSE_H_STRIDE: tl.constexpr,
    LSE_T_STRIDE: tl.constexpr,
    NORM_BLOCK: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    INLINE_EXTRAS: tl.constexpr = False,
):
    token, part = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + token) if RECORD else -1
    if part == 0:
        cols = tl.arange(0, NORM_BLOCK)
        value = tl.load(inputs + token * input_stride + cols, cols < HIDDEN, 0).to(tl.float32)
        if HAS_RESIDUAL:
            value += tl.load(residual + token * residual_stride + cols, cols < HIDDEN, 0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, 0) / HIDDEN + EPS)
        w = tl.load(weight + cols, cols < HIDDEN, 0).to(tl.float32)
        normalized = (value * rstd * w).to(output.dtype.element_ty)
        rounded = value.to(residual_output.dtype.element_ty)
        tl.store(output + token * HIDDEN + cols, normalized, cols < HIDDEN)
        tl.store(residual_output + token * HIDDEN + cols, rounded, cols < HIDDEN)
        if RECORD:
            valid = (cols < HIDDEN) & (slot >= 0)
            if DATA_OFFSET >= 0:
                tl.store(destination + slot * ROW_WIDTH + DATA_OFFSET + cols, rounded, valid)
            tl.store(destination + slot * ROW_WIDTH + NORM_OFFSET + cols, normalized, valid)
            tl.store(raw_destination + slot * HIDDEN + cols, value, valid)
            if LSE_HEADS > 0:
                valid_lse = (cols < LSE_HEADS) & (slot >= 0) & (token < LSE_TOKENS)
                normalizer = tl.load(lse + cols * LSE_H_STRIDE + token * LSE_T_STRIDE, valid_lse, 0)
                tl.store(lse_destination + slot * LSE_HEADS + cols, normalizer, valid_lse)
    if INLINE_EXTRAS or part > 0:
        copy_cols = tl.arange(0, COPY_BLOCK) if INLINE_EXTRAS else (part - 1) * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
        base = 0
        for field in tl.static_range(len(extra_widths)):
            local = copy_cols - base
            copy_valid = (slot >= 0) & (local >= 0) & (local < extra_widths[field])
            values = tl.load(extras[field] + token * extra_strides[field] + local, copy_valid, 0)
            tl.store(destination + slot * ROW_WIDTH + EXTRA_OFFSET + copy_cols, values, copy_valid)
            base += extra_widths[field]


@torch.library.custom_op("prime_ps::fused_norm_record", mutates_args=())
def fused_norm_record(
    layer: int,
    kind: int,
    inputs: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    qkv: torch.Tensor | None = None,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .ps_capture import _store

    output = torch.empty(inputs.shape, dtype=inputs.dtype, device=inputs.device)
    residual_output = torch.empty_like(output)
    hidden = inputs.shape[-1]
    destination, raw_destination, slots = output, output, output
    lse, lse_destination = output, output
    data_offset, norm_offset, extra_offset, lse_heads, lse_tokens, lse_hs, lse_ts = 0, 0, 0, 0, 0, 0, 0
    extras = ()
    if _store is not None:
        slots = _store.token_slots
        if kind == 2:
            destination, raw_destination = _store.final, _store.norm_inputs[-1]
            norm_offset = hidden
        else:
            destination, raw_destination = _store.layers[layer], _store.norm_inputs[2 * layer + kind]
            if kind == 0:
                norm_offset = _store.field_offsets["norm1"]
                data_offset = _store.field_offsets.get("input", -1)
            elif kind == 1:
                extra_offset = _store.field_offsets["qkv"]
                extras = (qkv, query, key, attention)
                norm_offset = _store.field_offsets["norm2"]
                data_offset = _store.field_offsets.get("mid", -1)
                saved_lse = _store.pending_lse.get(layer)
                if saved_lse is not None:
                    lse, lse_destination = saved_lse, _store.lse[layer]
                    lse_heads, lse_tokens = lse.shape
                    lse_hs, lse_ts = lse.stride()
            else:
                raise ValueError("Unknown normalization stage")
    widths = tuple(t.shape[-1] for t in extras)
    inline = False
    copy_block = triton.next_power_of_2(max(1, sum(widths))) if inline else 512
    parts = 1 if inline else 1 + triton.cdiv(sum(widths), 512)
    _rms_record[(inputs.shape[0], parts)](
        inputs,
        inputs if residual is None else residual,
        weight,
        output,
        residual_output,
        destination,
        raw_destination,
        slots,
        extras,
        tuple(t.stride(0) for t in extras),
        widths,
        lse,
        lse_destination,
        inputs.stride(0),
        0 if residual is None else residual.stride(0),
        hidden,
        epsilon,
        residual is not None,
        _store is not None,
        destination.shape[-1],
        data_offset,
        norm_offset,
        extra_offset,
        lse_heads,
        lse_tokens,
        lse_hs,
        lse_ts,
        triton.next_power_of_2(hidden),
        copy_block,
        inline,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output, residual_output


@fused_norm_record.register_fake
def _(layer, kind, inputs, residual, weight, epsilon, qkv=None, query=None, key=None, attention=None):
    return torch.empty(inputs.shape, device=inputs.device, dtype=inputs.dtype), torch.empty(
        inputs.shape, device=inputs.device, dtype=inputs.dtype
    )


@torch.library.custom_op("prime_ps::fused_silu_record", mutates_args=())
def fused_silu_record(layer: int, inputs: torch.Tensor) -> torch.Tensor:
    from .ps_capture import _store

    width = inputs.shape[-1] // 2
    output = torch.empty((inputs.shape[0], width), device=inputs.device, dtype=inputs.dtype)
    destination, slots, offset = output, output, 0
    if _store is not None:
        destination, slots = _store.layers[layer], _store.token_slots
        offset = _store.field_offsets["gate_up"]
    row = False
    block = triton.next_power_of_2(width) if row else 512
    _silu_record[(inputs.shape[0], triton.cdiv(width, block))](
        inputs,
        output,
        destination,
        slots,
        inputs.stride(0),
        width,
        _store is not None,
        destination.shape[-1],
        offset,
        block,
        _store is None or not _store.compact,
        num_warps=8 if row else 4,
    )
    return output


@fused_silu_record.register_fake
def _(layer, inputs):
    return torch.empty((inputs.shape[0], inputs.shape[-1] // 2), device=inputs.device, dtype=inputs.dtype)
