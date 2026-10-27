"""Full-sequence validation of VJPs built solely from the actual vLLM forward tape."""

import torch

from prime_rl.trainer.ps_backward.fa4 import sparse_ps_plan
from prime_rl.trainer.ps_backward.recorded import (
    recorded_add,
    recorded_attention,
    recorded_linear,
    recorded_rms,
    recorded_rope,
    recorded_silu,
)
from prime_rl.trainer.ps_backward.recorded_values import restore_values


def backward_recorded(
    model, trace, tape, request, rope_caches, *, fp32_boundaries=False, normalized_seed=False, head_block_tokens=None
):
    """No attention or dense forward; this validator has not yet applied the streaming schedule."""
    device = model.device
    count = trace.input_ids.numel() - 1
    offset = tape.requests[request] * tape.capacity
    dtype = tape.final.dtype
    boundary_dtype = torch.float32 if fp32_boundaries else dtype
    fp32_attention = getattr(model, "recorded_fp32_attention", False)
    weights = {}
    for p in model.parameters():
        if p.ndim == 2:
            stamp = p._version, p.data_ptr(), getattr(model, "_ps_update_version", 0)
            cached = model._weight_cast_cache.get(id(p))
            if cached is None or cached[0] != stamp:
                cached = stamp, p.detach().to(dtype)
                model._weight_cast_cache[id(p)] = cached
            weights[id(p)] = cached[1]

    def linear(module, x, y):
        return recorded_linear(x, module.weight, y, bias=module.bias, cast_weight=weights[id(module.weight)])

    gradient = torch.zeros((count, tape.hidden), device=device, dtype=boundary_dtype)
    loss = torch.zeros((), device=device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    for start in range(trace.prefix_length - 1, count, head_block_tokens or count):
        end = min(count, start + (head_block_tokens or count))
        final = tape.final[offset + start : offset + end]
        raw_final = tape.norm_inputs[-1][offset + start : offset + end] if fp32_boundaries else final[:, : tape.hidden]
        hidden = raw_final.to(device=device, dtype=boundary_dtype).detach().requires_grad_()
        normalized = recorded_rms(
            hidden, model.base.model.norm, final[:, tape.hidden :].to(device=device, dtype=boundary_dtype)
        )
        logits = recorded_linear(
            normalized,
            model.base.lm_head.weight,
            tape.logits[offset + start : offset + end].to(device),
            cast_weight=weights[id(model.base.lm_head.weight)],
        )
        block_loss = torch.nn.functional.cross_entropy(
            logits if logits.dtype == torch.float64 else logits.float(),
            trace.input_ids[start + 1 : end + 1].to(device),
            reduction="sum",
        )
        (block_loss / trace.completion_length if normalized_seed else block_loss).backward()
        gradient[start:end].copy_(hidden.grad)
        loss.add_(block_loss.detach())
    positions = torch.arange(count, device=device)
    plan = sparse_ps_plan(
        positions,
        positions,
        trace.prefix_length,
        model.window,
        (256 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10 else 128, 128),
    )
    for index in reversed(range(model.depth)):
        layer = model.base.model.layers[index]
        rec = restore_values({name: value.to(device) for name, value in tape.layer(request, index, 0, count).items()})
        attn, head = layer.self_attn, layer.self_attn.head_dim
        x = rec["norm_input" if fp32_boundaries else "input"].to(boundary_dtype).detach().requires_grad_()
        normalized = recorded_rms(x, layer.input_layernorm, rec["norm1"].to(boundary_dtype))
        raw_q, raw_k, raw_v = rec["qkv"].split([tape.qwidth, tape.kwidth, tape.kwidth], dim=-1)
        if fp32_attention:
            raw_q, raw_k, raw_v = (value.float() for value in (raw_q, raw_k, raw_v))
        q = linear(attn.q_proj, normalized, raw_q).view(1, count, -1, head)
        k = linear(attn.k_proj, normalized, raw_k).view(1, count, -1, head)
        v = linear(attn.v_proj, normalized, raw_v).view(1, count, -1, head).transpose(1, 2)
        q = recorded_rms(q, attn.q_norm, torch.empty_like(q, dtype=boundary_dtype)).transpose(1, 2)
        k = recorded_rms(k, attn.k_norm, torch.empty_like(k, dtype=boundary_dtype)).transpose(1, 2)
        cos, sin = rope_caches[index][:count].to(boundary_dtype).chunk(2, dim=-1)
        cos, sin = torch.cat((cos, cos), -1)[None, None], torch.cat((sin, sin), -1)[None, None]
        query_value = rec["query"].float() if fp32_attention else rec["query"]
        key_value = rec["key"].float() if fp32_attention else rec["key"]
        q = recorded_rope(q, cos, sin, query_value.view(1, count, -1, head).transpose(1, 2))
        k = recorded_rope(k, cos, sin, key_value.view(1, count, -1, head).transpose(1, 2))
        attended = recorded_attention(
            q,
            k,
            v,
            rec["attention"].view(1, count, -1, head).transpose(1, 2),
            rec["lse"].T[None].contiguous(),
            plan,
            trace.prefix_length,
            model.window,
            fp32_adjoint=fp32_attention,
        )
        projected = linear(attn.o_proj, attended.transpose(1, 2).reshape(count, -1), torch.zeros_like(x))
        mid = recorded_add(x, projected, rec["norm_mid" if fp32_boundaries else "mid"].to(boundary_dtype))
        normalized = recorded_rms(mid, layer.post_attention_layernorm, rec["norm2"].to(boundary_dtype))
        gate, up = rec["gate_up"].chunk(2, dim=-1)
        gate, up = linear(layer.mlp.gate_proj, normalized, gate), linear(layer.mlp.up_proj, normalized, up)
        gated = recorded_silu(gate, up, rec["gated"])
        output = mid + linear(layer.mlp.down_proj, gated, torch.zeros_like(mid))
        output.backward(gradient)
        gradient = x.grad
    embedding = model.base.model.embed_tokens.weight
    if embedding.grad is None:
        embedding.grad = torch.zeros_like(embedding)
    embedding.grad.index_add_(0, trace.input_ids[:-1].to(device), gradient.to(embedding.dtype))
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None and not normalized_seed:
                p.grad.div_(trace.completion_length)
    return loss.detach().item() / trace.completion_length
