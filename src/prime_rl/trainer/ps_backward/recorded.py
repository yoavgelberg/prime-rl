"""VJPs that consume recorded inference values instead of repeating their forward GEMMs."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_pointwise_override = ContextVar("ps_recorded_pointwise_override", default=True)


def _fused_pointwise(tensor):
    enabled = _pointwise_override.get()
    return enabled and tensor.is_cuda and tensor.dtype != torch.float64


@contextmanager
def unfused_recorded_pointwise():
    token = _pointwise_override.set(False)
    try:
        yield
    finally:
        _pointwise_override.reset(token)


class _RecordedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, output, cast_weight, bias):
        if inputs.shape[:-1] != output.shape[:-1] or output.shape[-1] != weight.shape[0]:
            raise ValueError("Recorded linear shapes do not match")
        ctx.save_for_backward(inputs, cast_weight, weight)
        ctx.has_bias = bias is not None
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight, master = ctx.saved_tensors
        x = inputs.reshape(-1, inputs.shape[-1]).to(weight.dtype)
        dy = gradient.reshape(-1, weight.shape[0]).to(weight.dtype)
        with torch.autocast(inputs.device.type, enabled=False):
            options = {"out_dtype": torch.float32} if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) else {}
            dx = (
                torch.mm(dy, weight, **options).reshape_as(inputs).to(inputs.dtype) if ctx.needs_input_grad[0] else None
            )
            dw = torch.mm(dy.T, x, **options).to(master.dtype) if ctx.needs_input_grad[1] else None
            db = dy.to(master.dtype).sum(0) if ctx.has_bias and ctx.needs_input_grad[4] else None
        return dx, dw, None, None, db


def recorded_linear(inputs, weight, output, *, cast_weight=None, bias=None):
    """The recorded output must come from these inputs and this immutable weight version."""
    cast_weight = weight if cast_weight is None else cast_weight
    return _RecordedLinear.apply(inputs, weight, output, cast_weight, bias)


class _RecordedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, output):
        if left.shape != right.shape or left.shape != output.shape:
            raise ValueError("Recorded residual requires identical shapes")
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None


def recorded_add(left, right, output):
    return _RecordedAdd.apply(left, right, output)


class _RecordedRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, output, epsilon):
        ctx.save_for_backward(inputs, weight)
        ctx.epsilon = epsilon
        ctx.fused = _fused_pointwise(inputs)
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight = ctx.saved_tensors
        if ctx.fused:
            from .recorded_pointwise import rms_backward

            dx, dw = rms_backward(inputs, gradient, weight, ctx.epsilon)
            return dx, dw, None, None
        dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
        x, dy, w = inputs.to(dtype), gradient.to(dtype), weight.to(dtype)
        rstd = (x.square().mean(-1, keepdim=True) + ctx.epsilon).rsqrt()
        normalized, weighted = x * rstd, dy * w
        dx = rstd * (weighted - normalized * (weighted * normalized).mean(-1, keepdim=True))
        dw = (dy * normalized).sum(tuple(range(inputs.ndim - 1)))
        return dx.to(inputs.dtype), dw.to(weight.dtype), None, None


def recorded_rms(inputs, module, output):
    return _RecordedRMS.apply(inputs, module.weight, output, module.variance_epsilon)


class _RecordedRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, cos, sin, output):
        ctx.save_for_backward(cos, sin)
        ctx.fused = _fused_pointwise(inputs) and inputs.ndim == 4
        ctx.input_dtype = inputs.dtype
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        from .model import rotate_half

        cos, sin = ctx.saved_tensors
        if ctx.fused:
            from .recorded_pointwise import rope_backward

            return rope_backward(gradient, cos, sin, ctx.input_dtype), None, None, None
        return gradient * cos - rotate_half(gradient * sin), None, None, None


def recorded_rope(inputs, cos, sin, output):
    return _RecordedRope.apply(inputs, cos, sin, output)


class _RecordedSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, output):
        ctx.save_for_backward(gate, up)
        ctx.fused = _fused_pointwise(gate)
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        gate, up = ctx.saved_tensors
        if ctx.fused:
            from .recorded_pointwise import silu_backward

            dg, du = silu_backward(gate, up, gradient)
            return dg, du, None
        dtype = torch.float64 if gate.dtype == torch.float64 else torch.float32
        g, u, dy = gate.to(dtype), up.to(dtype), gradient.to(dtype)
        sigmoid = g.sigmoid()
        return (dy * u * sigmoid * (1 + g * (1 - sigmoid))).to(gate.dtype), (dy * g * sigmoid).to(up.dtype), None


def recorded_silu(gate, up, output):
    return _RecordedSilu.apply(gate, up, output)


class _RecordedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, output, lse, plan, prefix_length, window, fp32_adjoint):
        ctx.fp32_adjoint = fp32_adjoint
        if fp32_adjoint:
            if not q.is_cuda or any(t.dtype != torch.float32 for t in (q, k, v)):
                raise ValueError("FP32 attention adjoints require CUDA Float32 graph boundaries")
            q, k, v = (t.to(output.dtype) for t in (q, k, v))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.plan, ctx.prefix_length, ctx.window = plan, prefix_length, window
        return output.detach().float() if fp32_adjoint else output.detach()

    @staticmethod
    def backward(ctx, gradient):
        q, k, v, output, lse = ctx.saved_tensors
        if not q.is_cuda:
            q, k, v, output, gradient = (x.transpose(1, 2) for x in (q, k, v, output, gradient))
            groups = q.shape[1] // k.shape[1]
            expanded_k, expanded_v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
            qi, ki = ctx.plan.q_indices, ctx.plan.k_indices
            allowed = (
                (ki[None] <= qi[:, None])
                & (ki[None] >= 0)
                & ((ki[None] < ctx.prefix_length) | (ki[None] >= qi[:, None] - ctx.window + 1))
            )
            probabilities = ((q @ expanded_k.transpose(-2, -1)) * q.shape[-1] ** -0.5 - lse[..., None]).exp()
            probabilities = probabilities.masked_fill(~allowed, 0)
            dp = gradient @ expanded_v.transpose(-2, -1)
            ds = probabilities * (dp - (gradient * output).sum(-1, keepdim=True))
            dq = (ds @ expanded_k) * q.shape[-1] ** -0.5
            dk = (ds.transpose(-2, -1) @ q) * q.shape[-1] ** -0.5
            dv = probabilities.transpose(-2, -1) @ gradient
            dk = dk.reshape(k.shape[0], k.shape[1], groups, *k.shape[2:]).sum(2)
            dv = dv.reshape(v.shape[0], v.shape[1], groups, *v.shape[2:]).sum(2)
            return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None, None, None, None, None

        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import _flash_attn_bwd

        from .fa4 import _ps_mask

        backward = _flash_attn_bwd
        if ctx.fp32_adjoint:
            from .fa4_adjoint import backward_fp32

            backward = backward_fp32
            gradient = gradient.to(q.dtype)

        plan = ctx.plan
        metadata = BlockSparseTensorsTorch(*plan.backward, block_size=plan.block_size)
        dq, dk, dv = backward(
            q,
            k,
            v,
            output,
            gradient,
            lse,
            softmax_scale=q.shape[-1] ** -0.5,
            mask_mod=_ps_mask(ctx.prefix_length, ctx.window),
            aux_tensors=[plan.q_indices, plan.k_indices],
            block_sparse_tensors=metadata,
        )
        return dq, dk, dv, None, None, None, None, None, None


def recorded_attention(q, k, v, output, lse, plan, prefix_length, window, *, fp32_adjoint=False):
    """Use [B,H,T,D] tensors and [B,H,T] natural-log softmax normalizers."""
    from .fa4 import _ReplicatePrefix

    count = q.shape[2]
    if plan.q_padding or plan.q_tail_padding:
        pads = (0, 0, plan.q_padding, plan.q_tail_padding)
        q, output = torch.nn.functional.pad(q, pads), torch.nn.functional.pad(output, pads)
        lse = torch.nn.functional.pad(lse, (plan.q_padding, plan.q_tail_padding))
    if plan.source_indices is not None:
        k, v = _ReplicatePrefix.apply(k, plan.source_indices), _ReplicatePrefix.apply(v, plan.source_indices)
    result = _RecordedAttention.apply(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        output.transpose(1, 2),
        lse,
        plan,
        prefix_length,
        window,
        fp32_adjoint,
    )
    return result[:, plan.q_padding : plan.q_padding + count].transpose(1, 2)
