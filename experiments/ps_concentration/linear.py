"""BF16 tensor-core products retaining FP32 projection outputs and gradients."""
import torch

from prime_rl.trainer.ps_backward.model import QwenPS


class AccumulatorLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, cast_weight):
        with torch.autocast('cuda', enabled=False):
            x = inputs.reshape(-1, inputs.shape[-1]).bfloat16()
            output = torch.mm(x, cast_weight.T, out_dtype=torch.float32)
            if bias is not None:
                output.add_(bias.float())
        ctx.save_for_backward(x, cast_weight)
        ctx.shape, ctx.dtype, ctx.bias = inputs.shape, inputs.dtype, bias is not None
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        with torch.autocast('cuda', enabled=False):
            dy = gradient.reshape(-1, weight.shape[0]).bfloat16()
            dx = torch.mm(dy, weight, out_dtype=torch.float32).reshape(ctx.shape).to(ctx.dtype) if ctx.needs_input_grad[0] else None
            dw = torch.mm(dy.T, x, out_dtype=torch.float32) if ctx.needs_input_grad[1] else None
            db = dy.float().sum(0) if ctx.bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


class ProjectionCache:
    def __init__(self):
        self.casts = {}

    def linear(self, inputs, weight, bias=None):
        stamp = weight._version, weight.data_ptr()
        cached = self.casts.get(id(weight))
        if cached is None or cached[0] != stamp:
            cached = stamp, weight.detach().bfloat16()
            self.casts[id(weight)] = cached
        return AccumulatorLinear.apply(inputs, weight, bias, cached[1])

    def project(self, module, inputs):
        return self.linear(inputs, module.weight, module.bias)


class ConcentrationQwen(QwenPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, fp32_linear_gradients=True)
        self.projections = ProjectionCache()

    def linear(self, inputs, weight, bias=None):
        return self.projections.linear(inputs, weight, bias)
