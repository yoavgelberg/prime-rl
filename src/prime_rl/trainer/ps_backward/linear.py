"""Native BF16 forward GEMMs with shared weight casts and FP32 dense gradient outputs."""

import torch


class _FP32GradLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, cast_weight):
        if not inputs.is_cuda or weight.dtype != torch.float32 or cast_weight.dtype != torch.bfloat16:
            raise ValueError("FP32-gradient linear requires CUDA, FP32 master weights and BF16 compute")
        with torch.autocast("cuda", enabled=False):
            x = inputs.reshape(-1, inputs.shape[-1]).to(torch.bfloat16)
            output = torch.nn.functional.linear(x, cast_weight, bias.to(torch.bfloat16) if bias is not None else None)
        ctx.save_for_backward(x, cast_weight, weight)
        ctx.input_shape, ctx.input_dtype, ctx.has_bias = inputs.shape, inputs.dtype, bias is not None
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, gradient):
        x, weight, _master = ctx.saved_tensors
        with torch.autocast("cuda", enabled=False):
            dy = gradient.reshape(-1, weight.shape[0]).to(torch.bfloat16)
            dx = dw = db = None
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, weight, out_dtype=torch.float32).reshape(ctx.input_shape).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                dw = torch.mm(dy.T, x, out_dtype=torch.float32)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                db = dy.float().sum(0)
        return dx, dw, db, None


def fp32grad_linear(inputs, weight, bias, cast_weight):
    return _FP32GradLinear.apply(inputs, weight, bias, cast_weight)
