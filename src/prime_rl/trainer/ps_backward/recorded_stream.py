"""Incremental reverse differentiation of immutable, recorded Qwen PS activations."""

from contextlib import contextmanager

import torch

from .fa4 import AlignedChunkPlans, sparse_ps_plan
from .recorded import (
    recorded_add,
    recorded_attention,
    recorded_linear,
    recorded_rms,
    recorded_rope,
    recorded_silu,
)
from .recorded_values import restore_values
from .retirement import RetirementSchedule


class RecordedStream:
    def __init__(
        self,
        model,
        tape,
        request,
        prefix,
        rope_caches,
        *,
        chunk=512,
        policy_version=0,
        graph_cache=None,
        tail_block_tokens=16384,
        tail_prefetch=None,
    ):
        self.model, self.tape, self.request, self.ropes = model, tape, request, rope_caches
        self.schedule = RetirementSchedule(model.depth, model.window, prefix, chunk, policy_version)
        self.graph_cache = graph_cache
        self.fp32_attention = getattr(model, "recorded_fp32_attention", False)
        self._tail_records = None
        self._finishing = False
        if tail_block_tokens < chunk:
            raise ValueError("Tail batches must hold at least one recorded block")
        self.tail_block_tokens = tail_block_tokens
        if tail_prefetch not in (None, "dma", "sm"):
            raise ValueError("Tail prefetch must use dma or sm")
        self.tail_prefetch = tail_prefetch
        if tail_prefetch is not None and not hasattr(model, "_ps_tail_copy_stream"):
            model._ps_tail_copy_stream = torch.cuda.Stream(device=model.device)
        self.plans = graph_cache.plans if graph_cache is not None else AlignedChunkPlans()
        self.parameters = dict(model.named_parameters())
        self.gradients = {name: None for name in self.parameters}
        self.dtype = tape.activation_dtype if hasattr(tape, "activation_dtype") else tape.final.dtype
        self.adjoint_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        self.weights = {}
        for p in self.parameters.values():
            if p.ndim == 2:
                stamp = p._version, p.data_ptr(), getattr(model, "_ps_update_version", 0)
                cached = model._weight_cast_cache.get(id(p))
                if cached is None or cached[0] != stamp:
                    cached = stamp, p.detach().to(self.dtype)
                    model._weight_cast_cache[id(p)] = cached
                self.weights[id(p)] = cached[1]
        self.versions = tuple(p._version for p in self.parameters.values())
        self.loss = torch.zeros((), device=model.device, dtype=self.adjoint_dtype)
        self.loss_tokens = 0

    @contextmanager
    def _active(self, policy_version):
        if policy_version != self.schedule.policy_version or self.versions != tuple(
            p._version for p in self.parameters.values()
        ):
            raise ValueError("Policy weights changed during recorded backward")
        previous = {name: p.grad for name, p in self.parameters.items()}
        for name, p in self.parameters.items():
            p.grad = self.gradients[name]
        try:
            yield
        finally:
            for name, p in self.parameters.items():
                self.gradients[name], p.grad = p.grad, previous[name]

    def _linear(self, module, x, output):
        return recorded_linear(x, module.weight, output, bias=module.bias, cast_weight=self.weights[id(module.weight)])

    def _rec(self, layer, start, end, fields=None):
        if self._tail_records is not None:
            return self._tail_records.read(layer, start, end, fields)
        return self._read_rec(layer, start, end, fields)

    def _read_rec(self, layer, start, end, fields=None, *, copy_mode=None):
        if hasattr(self.tape, "load_layer"):
            options = {} if copy_mode is None else {"copy_mode": copy_mode}
            rec = self.tape.load_layer(self.request, layer, start, end, self.model.device, fields, **options)
        else:
            rec = {
                k: v.to(self.model.device, non_blocking=True)
                for k, v in self.tape.layer(self.request, layer, start, end).items()
                if fields is None or k in fields
            }
        rec = restore_values(rec)
        return rec

    def _rotated(self, raw, module, recorded, start, end):
        head = self.model.base.model.layers[0].self_attn.head_dim
        raw = raw.view(1, end - start, -1, head)
        norm = recorded_rms(raw, module, torch.empty_like(raw, dtype=self.adjoint_dtype)).transpose(1, 2)
        cos, sin = self.ropes[0][start:end].to(self.adjoint_dtype).chunk(2, -1)
        cos, sin = torch.cat((cos, cos), -1)[None, None], torch.cat((sin, sin), -1)[None, None]
        output = recorded.float() if self.fp32_attention else recorded
        return recorded_rope(norm, cos, sin, output.view(1, end - start, -1, head).transpose(1, 2))

    def append(self, end, inputs, targets, *, policy_version):
        """Inputs and next-token labels cover [previous_end,end); prompt labels are -100."""
        start = self.schedule.end
        if inputs.shape != (end - start,) or targets.shape != inputs.shape:
            raise ValueError("Recorded block tokens and labels must match its positions")
        with self._active(policy_version):
            payload = dict(ids=inputs.to(self.model.device, non_blocking=True), dx={}, dkv={})
            block = self.schedule.append(end, policy_version=policy_version, payload=payload)
            first = max(self.schedule.prefix - 1 - start, 0)
            if hasattr(self.tape, "head"):
                data = self.tape.head(self.request, start, end, first)
                raw, normalized_value, logits = (
                    data[key].to(self.model.device, non_blocking=True) for key in ("raw", "normalized", "logits")
                )
            else:
                offset = self.tape.requests[self.request] * self.tape.capacity
                final = self.tape.final[offset + start : offset + end].to(self.model.device, non_blocking=True)
                raw = (
                    self.tape.norm_inputs[-1][offset + start : offset + end]
                    if hasattr(self.tape, "norm_inputs")
                    else (final[:, : self.tape.hidden])
                )
                normalized_value = final[:, self.tape.hidden :]
                logits = self.tape.logits[offset + start + first : offset + end].to(
                    self.model.device, non_blocking=True
                )
            raw = raw.to(device=self.model.device, dtype=self.adjoint_dtype)
            if self.graph_cache is not None:
                loss, selected_gradient = self.graph_cache.head(
                    raw[first:],
                    normalized_value[first:],
                    logits,
                    targets[first:].to(self.model.device, non_blocking=True),
                )
                gradient = torch.zeros_like(raw)
                gradient[first:].copy_(selected_gradient)
            else:
                x = raw.detach().requires_grad_()
                normalized = recorded_rms(x, self.model.base.model.norm, normalized_value.to(self.adjoint_dtype))
                logits = self._linear(self.model.base.lm_head, normalized[first:], logits)
                loss = torch.nn.functional.cross_entropy(
                    logits.to(self.adjoint_dtype),
                    targets[first:].to(self.model.device, non_blocking=True),
                    reduction="sum",
                )
                loss.backward()
                gradient = x.grad
            self.loss.add_(loss.detach())
            self.loss_tokens += end - start - first
            payload["dx"][self.model.depth] = gradient
            if hasattr(self.tape, "release_head"):
                self.tape.release_head(self.request, start, end)
            self.schedule.advance(self._consume, policy_version=policy_version)
            return block

    def _consume(self, action, block, layer_index):
        if action == "query":
            self._query(block, layer_index)
        elif action == "kv":
            self._kv(block, layer_index)
        elif action == "release_layer":
            if hasattr(self.tape, "release_layer"):
                self.tape.release_layer(self.request, layer_index, block.start, block.end)
        elif action == "embedding":
            weight = self.model.base.model.embed_tokens.weight
            if weight.grad is None:
                weight.grad = torch.zeros_like(weight)
            weight.grad.index_add_(0, block.payload["ids"], block.payload["dx"].pop(0).to(weight.dtype))
            block.payload.clear()
        else:
            raise ValueError(action)

    def _query(self, block, index):
        model, tape = self.model, self.tape
        layer, n = model.base.model.layers[index], block.end - block.start
        attn, head = layer.self_attn, layer.self_attn.head_dim
        rec = self._rec(index, block.start, block.end)
        if self.graph_cache is not None:
            residual_gradient, attention_gradient = self.graph_cache.post(
                index, rec, block.payload["dx"].pop(index + 1)
            )
            q = rec["query"].view(1, n, -1, head).transpose(1, 2)
        else:
            x = rec.get("norm_input", rec["input"]).to(self.adjoint_dtype).detach().requires_grad_()
            normalized = recorded_rms(x, layer.input_layernorm, rec["norm1"].to(self.adjoint_dtype))
            raw = rec["qkv"][:, : tape.qwidth]
            qraw = self._linear(attn.q_proj, normalized, raw.float() if self.fp32_attention else raw)
            q = self._rotated(qraw, attn.q_norm, rec["query"], block.start, block.end)
        pieces, keys, values = [], [], []
        history_start = max(self.schedule.prefix, block.start - model.window + 1)
        for old in self.schedule.blocks:
            start = old.start if old.start == 0 else max(old.start, history_start)
            end = min(old.end, block.end)
            if start >= end:
                continue
            source = (
                {key: rec[key][start - block.start : end - block.start] for key in ("key", "qkv")}
                if (start >= block.start and end <= block.end)
                else (self._rec(index, start, end, ("key", "qkv")))
            )
            k = (
                source["key"]
                .view(1, end - start, -1, head)
                .transpose(1, 2)
                .to(self.adjoint_dtype)
                .detach()
                .requires_grad_()
            )
            v = source["qkv"][:, tape.qwidth + tape.kwidth :].view(1, end - start, -1, head).transpose(1, 2)
            v = v.to(self.adjoint_dtype).detach().requires_grad_()
            pieces.append((old, start - old.start, end - old.start, k, v))
            keys.append(k if self.fp32_attention else k.to(self.dtype))
            values.append(v if self.fp32_attention else v.to(self.dtype))
        if "attention_plan" not in block.payload:
            tile = (
                256 if model.device.type == "cuda" and torch.cuda.get_device_capability(model.device)[0] >= 10 else 128
            )
            if model.device.type == "cuda" and block.start:
                plan = self.plans.get(
                    self.schedule.prefix, block.start, block.end, history_start, model.window, model.device, (tile, 128)
                )
            else:
                qi = torch.arange(block.start, block.end, device=model.device)
                ki = torch.cat(
                    (
                        torch.arange(self.schedule.prefix, device=model.device),
                        torch.arange(history_start, block.end, device=model.device),
                    )
                )
                plan = sparse_ps_plan(qi, ki, self.schedule.prefix, model.window, (tile, 128))
            block.payload["attention_plan"] = plan
        plan = block.payload["attention_plan"]
        attended_value = rec["attention"].view(1, n, -1, head).transpose(1, 2)
        lse = rec["lse"].T[None].contiguous()
        if self.graph_cache is not None:
            dq, all_dk, all_dv = self.graph_cache.attention(
                q,
                torch.cat(keys, 2),
                torch.cat(values, 2),
                attended_value,
                lse,
                attention_gradient.view(1, n, -1, head).transpose(1, 2),
                plan,
                self.schedule.prefix,
            )
            query_gradient = self.graph_cache.query(
                index, rec, dq, self.ropes[index][block.start : block.end], tape.qwidth
            )
            block.payload["dx"][index] = residual_gradient + query_gradient
        else:
            attended = recorded_attention(
                q,
                torch.cat(keys, 2),
                torch.cat(values, 2),
                attended_value,
                lse,
                plan,
                self.schedule.prefix,
                model.window,
                fp32_adjoint=self.fp32_attention,
            )
            projected = self._linear(attn.o_proj, attended.transpose(1, 2).reshape(n, -1), torch.zeros_like(x))
            mid = recorded_add(x, projected, rec.get("norm_mid", rec["mid"]).to(self.adjoint_dtype))
            normalized = recorded_rms(mid, layer.post_attention_layernorm, rec["norm2"].to(self.adjoint_dtype))
            gate, up = rec["gate_up"].chunk(2, -1)
            gate = self._linear(layer.mlp.gate_proj, normalized, gate)
            up = self._linear(layer.mlp.up_proj, normalized, up)
            gated = recorded_silu(gate, up, rec["gated"])
            output = mid + self._linear(layer.mlp.down_proj, gated, torch.zeros_like(mid))
            output.backward(block.payload["dx"].pop(index + 1))
            block.payload["dx"][index] = x.grad
        gradient_offset = 0
        for old, start, end, k, v in pieces:
            if index not in old.payload["dkv"]:
                shape = (1, k.shape[1], old.end - old.start, head)
                old.payload["dkv"][index] = tuple(
                    torch.zeros(shape, device=model.device, dtype=self.adjoint_dtype) for _ in range(2)
                )
            dk, dv = old.payload["dkv"][index]
            key_gradient = (
                k.grad if self.graph_cache is None else all_dk[:, :, gradient_offset : gradient_offset + end - start]
            )
            value_gradient = (
                v.grad if self.graph_cache is None else all_dv[:, :, gradient_offset : gradient_offset + end - start]
            )
            dk[:, :, start:end].add_(key_gradient)
            dv[:, :, start:end].add_(value_gradient)
            gradient_offset += end - start

    def _kv(self, block, index):
        layer, tape = self.model.base.model.layers[index], self.tape
        attn, n = layer.self_attn, block.end - block.start
        rec = self._rec(index, block.start, block.end, ("input", "norm_input", "norm1", "qkv", "key"))
        if self.graph_cache is not None:
            dk, dv = block.payload["dkv"].pop(index)
            gradient = self.graph_cache.kv(
                index, rec, dk, dv, self.ropes[index][block.start : block.end], tape.qwidth, tape.kwidth
            )
            block.payload["dx"][index].add_(gradient)
            return
        x = rec.get("norm_input", rec["input"]).to(self.adjoint_dtype).detach().requires_grad_()
        normalized = recorded_rms(x, layer.input_layernorm, rec["norm1"].to(self.adjoint_dtype))
        _, kraw, vraw = rec["qkv"].split([tape.qwidth, tape.kwidth, tape.kwidth], -1)
        if self.fp32_attention:
            kraw, vraw = kraw.float(), vraw.float()
        k = self._linear(attn.k_proj, normalized, kraw)
        k = self._rotated(k, attn.k_norm, rec["key"], block.start, block.end)
        v = self._linear(attn.v_proj, normalized, vraw).view(1, n, -1, attn.head_dim).transpose(1, 2)
        dk, dv = block.payload["dkv"].pop(index)
        torch.autograd.backward((k, v), (dk.to(k.dtype), dv.to(v.dtype)))
        block.payload["dx"][index].add_(x.grad)

    def finish(self, *, policy_version, normalize=True, layerwise=False):
        self._finishing = True
        with self._active(policy_version):
            if layerwise:
                from .recorded_tail import finish_layerwise

                finish_layerwise(self)
            else:
                self.schedule.advance(self._consume, policy_version=policy_version, finished=True)
            if normalize:
                for parameter in self.parameters.values():
                    if parameter.grad is not None:
                        parameter.grad.div_(self.loss_tokens)
        return self.gradients
