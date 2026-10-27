"""Differentiate recorded Q/K/V projections with two GEMMs and one shared input VJP."""

import torch

from .gradient_accumulation import accumulate_weight_gradient, weight_gradient_target


class _RecordedQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, wq, wk, wv, output, cast_weight, bq, bk, bv):
        widths = tuple(weight.shape[0] for weight in (wq, wk, wv))
        if (
            inputs.shape[:-1] != output.shape[:-1]
            or output.shape[-1] != sum(widths)
            or cast_weight.shape != (sum(widths), inputs.shape[-1])
        ):
            raise ValueError("Recorded QKV shapes do not match")
        ctx.save_for_backward(inputs, cast_weight, wq, wk, wv)
        ctx.widths = widths
        ctx.has_bias = tuple(bias is not None for bias in (bq, bk, bv))
        ctx.gradient_target = weight_gradient_target((wq, wk, wv))
        return output.detach()

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight, *masters = ctx.saved_tensors
        x = inputs.reshape(-1, inputs.shape[-1]).to(weight.dtype)
        dy = gradient.reshape(-1, weight.shape[0]).to(weight.dtype)
        with torch.autocast(inputs.device.type, enabled=False):
            options = {"out_dtype": torch.float32} if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) else {}
            dx = (
                torch.mm(dy, weight, **options).reshape_as(inputs).to(inputs.dtype) if ctx.needs_input_grad[0] else None
            )
            if ctx.gradient_target is not None and any(ctx.needs_input_grad[1:4]):
                if not all(ctx.needs_input_grad[1:4]):
                    raise ValueError("Packed direct gradients require all three projections to train")
                accumulate_weight_gradient(dy, x, ctx.gradient_target)
                dw = (None, None, None)
            else:
                dw = (
                    torch.mm(dy.T, x, **options).split(ctx.widths, 0)
                    if any(ctx.needs_input_grad[1:4])
                    else (None, None, None)
                )
                dw = tuple(
                    value.to(master.dtype) if value is not None and ctx.needs_input_grad[index + 1] else None
                    for index, (value, master) in enumerate(zip(dw, masters, strict=True))
                )
            db = dy.to(masters[0].dtype).sum(0).split(ctx.widths) if any(ctx.has_bias) else (None, None, None)
            db = tuple(
                value if present and ctx.needs_input_grad[index + 6] else None
                for index, (value, present) in enumerate(zip(db, ctx.has_bias, strict=True))
            )
        return dx, *dw, None, None, *db


def recorded_qkv(inputs, attention, output, cast_weight):
    projections = attention.q_proj, attention.k_proj, attention.v_proj
    return _RecordedQKV.apply(
        inputs,
        *(module.weight for module in projections),
        output,
        cast_weight,
        *(module.bias for module in projections),
    )


@torch.no_grad()
def prepare_qkv_weights(model, weights):
    if not hasattr(model, "_recorded_qkv_weights"):
        model._recorded_qkv_weights = {}
    for index, layer in enumerate(model.base.model.layers):
        attention = layer.self_attn
        parameters = tuple(module.weight for module in (attention.q_proj, attention.k_proj, attention.v_proj))
        stamp = tuple(
            (
                p._version,
                p.data_ptr(),
                weights[id(p)].data_ptr(),
                getattr(model, "_ps_update_version", 0),
            )
            for p in parameters
        )
        cached = model._recorded_qkv_weights.get(index)
        if cached is None or cached[0] != stamp:
            model._recorded_qkv_weights[index] = (
                stamp,
                torch.cat([weights[id(p)] for p in parameters]),
            )
