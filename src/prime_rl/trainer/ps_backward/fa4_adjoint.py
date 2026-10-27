"""Expose FA4's FP32 accumulators through its existing postprocess kernel for GQA."""

from contextvars import ContextVar
from functools import wraps

import torch

_outputs = ContextVar("ps_fa4_fp32_outputs", default=None)


def _install():
    import cutlass
    from flash_attn.cute import interface

    original = interface._bwd_postprocess_convert
    if getattr(original, "_ps_fp32_outputs", False):
        return

    @wraps(original)
    def convert(accum, output, scale, cu_seqlens, seqused, arch, dtype, *args, **kwargs):
        captured = _outputs.get()
        if captured is None:
            return original(accum, output, scale, cu_seqlens, seqused, arch, dtype, *args, **kwargs)
        from .fa4_fp32_postprocess import convert_fp32

        result = torch.empty_like(output, dtype=torch.float32)
        convert_fp32(accum, result, scale, cu_seqlens, seqused, arch, cutlass.Float32, *args, **kwargs)
        captured[output.data_ptr()] = result

    convert._ps_fp32_outputs = True
    interface._bwd_postprocess_convert = convert


def backward_fp32(q, k, v, out, dout, lse, **options):
    from flash_attn.cute.interface import _flash_attn_bwd

    if q.shape[-2] <= k.shape[-2] or q.shape[-1] == 256:
        raise ValueError("FP32 accumulator exposure requires GQA and the ordinary FA4 backward kernel")
    _install()
    captured = {}
    token = _outputs.set(captured)
    try:
        result = _flash_attn_bwd(q, k, v, out, dout, lse, **options)
    finally:
        _outputs.reset(token)
    if len(result) != 3 or any(value.data_ptr() not in captured for value in result):
        raise RuntimeError("FA4 did not expose all three FP32 gradient accumulators")
    return tuple(captured[value.data_ptr()] for value in result)
