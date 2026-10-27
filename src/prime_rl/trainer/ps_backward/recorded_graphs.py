"""Reusable CUDA graphs for the dense and attention parts of recorded PS VJPs."""

import time
from dataclasses import replace

import torch

from .fa4 import AlignedChunkPlans
from .gradient_accumulation import direct_weight_gradients
from .recorded import (
    recorded_add,
    recorded_attention,
    recorded_linear,
    recorded_rms,
    recorded_rope,
    recorded_silu,
)
from .recorded_qkv import recorded_qkv


class _Stage:
    def __init__(self, function, example, rows=None, fills=None, dtypes=None):
        self.function, self.rows, self.fills = function, rows, fills or {}
        self.static = {}
        for name, value in example.items():
            if isinstance(value, (list, tuple)):
                shape = (
                    *value[0].shape[:2],
                    sum(v.shape[2] for v in value),
                    *value[0].shape[3:],
                )
                value = value[0]
            else:
                shape = (rows, *value.shape[1:]) if rows is not None and name != "gradient_pointers" else value.shape
            dtype = (dtypes or {}).get(name, value.dtype)
            self.static[name] = torch.empty(shape, device=value.device, dtype=dtype)
        self.graph, self.output = None, None

    @torch.no_grad()
    def _load(self, values):
        destinations, sources = [], []
        for name, value in values.items():
            target = self.static[name]
            if isinstance(value, (list, tuple)):
                offset = 0
                for piece in value:
                    destinations.append(target[:, :, offset : offset + piece.shape[2]])
                    sources.append(piece)
                    offset += piece.shape[2]
                continue
            if target.shape != value.shape:
                target.fill_(self.fills.get(name, 0))
                target = target[: value.shape[0]]
            destinations.append(target)
            sources.append(value)
        torch._foreach_copy_(destinations, sources)

    def run(self, values):
        self._load(values)
        if self.graph is None:
            return self.function(self.static)
        self.graph.replay()
        return self.output

    def capture(self, stream):
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.function(self.static)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self.function(self.static)


class RecordedGraphCache:
    def __init__(self, model, chunk):
        self.model, self.chunk = model, chunk
        self.fp32_attention = getattr(model, "recorded_fp32_attention", False)
        self.stages = {}
        self.plans = AlignedChunkPlans()
        self.require_ready = False
        self.replays, self.eager_calls = 0, 0
        self.unwarmed_attention_calls = 0

    def _call(self, key, function, values, *, rows=None, fills=None, dtypes=None):
        direct = getattr(self.model, "recorded_direct_weight_gradients", False)
        if direct and key[0] != "attention":
            values = dict(values, gradient_pointers=self.model._ps_gradient_pointers)
            original = function

            def function(xs):
                with direct_weight_gradients(
                    xs["gradient_pointers"],
                    self.model._ps_parameter_indices,
                    backend=getattr(self.model, "recorded_weight_gradient_backend", "triton"),
                ):
                    return original(xs)

        if key not in self.stages:
            if self.require_ready:
                if key[0] == "attention":
                    self.eager_calls += 1
                    self.unwarmed_attention_calls += 1
                    values = {
                        name: torch.cat(value, 2) if isinstance(value, (tuple, list)) else value
                        for name, value in values.items()
                    }
                    values = {name: value.to((dtypes or {}).get(name, value.dtype)) for name, value in values.items()}
                    return function(values)
                raise RuntimeError(f"Unwarmed recorded-backward graph shape: {key}")
            self.stages[key] = _Stage(function, values, rows, fills, dtypes)
        stage = self.stages[key]
        if stage.graph is not None:
            self.replays += 1
        else:
            if self.require_ready:
                raise RuntimeError(f"Uncaptured recorded-backward stage: {key}")
            self.eager_calls += 1
        return stage.run(values)

    def capture(self):
        start = time.perf_counter()
        stream = torch.cuda.Stream(device=self.model.device)
        stream.wait_stream(torch.cuda.current_stream())
        generator = torch.cuda.default_generators[self.model.device.index or 0]
        inference_state = generator.graphsafe_get_state()
        # vLLM's first graph allocates RNG bookkeeping tensors in inference mode.
        # Backward graphs need their own normal tensors; preserve the sampler state.
        generator.graphsafe_set_state(generator.clone_state())
        try:
            for stage in self.stages.values():
                if stage.graph is None:
                    stage.capture(stream)
        finally:
            generator.graphsafe_set_state(inference_state)
        stream.synchronize()
        self.require_ready = True
        self.replays = self.eager_calls = 0
        return dict(
            graphs=len(self.stages),
            seconds=time.perf_counter() - start,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )

    def _linear(self, module, x, y):
        cast = self.model._weight_cast_cache[id(module.weight)][1]
        return recorded_linear(x, module.weight, y, cast_weight=cast, bias=module.bias)

    @staticmethod
    def _parameters(*modules):
        return tuple(parameter for module in modules for parameter in module.parameters())

    @staticmethod
    @torch.no_grad()
    def _accumulate(parameters, gradients):
        destinations, sources = [], []
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                destinations.append(parameter.grad)
                sources.append(gradient)
        if destinations:
            torch._foreach_add_(destinations, sources)

    @staticmethod
    def _rotate(raw, norm, output, rope, head):
        count = raw.shape[0]
        raw = raw.view(1, count, -1, head)
        normalized = recorded_rms(raw, norm, torch.empty_like(raw, dtype=torch.float32)).transpose(1, 2)
        cos, sin = rope.chunk(2, -1)
        cos, sin = (
            torch.cat((cos, cos), -1)[None, None],
            torch.cat((sin, sin), -1)[None, None],
        )
        return recorded_rope(normalized, cos, sin, output.view(1, count, -1, head).transpose(1, 2))

    def post(self, index, rec, gradient):
        layer = self.model.base.model.layers[index]
        parameters = self._parameters(layer.self_attn.o_proj, layer.post_attention_layernorm, layer.mlp)

        def run(xs):
            x = torch.empty_like(xs["mid"], requires_grad=True)
            attention = xs["attention"].detach().requires_grad_()
            projected = self._linear(layer.self_attn.o_proj, attention, torch.empty_like(x))
            mid = recorded_add(x, projected, xs["mid"])
            normalized = recorded_rms(mid, layer.post_attention_layernorm, xs["norm2"])
            gate, up = xs["gate_up"].chunk(2, -1)
            gate = self._linear(layer.mlp.gate_proj, normalized, gate)
            up = self._linear(layer.mlp.up_proj, normalized, up)
            gated = recorded_silu(gate, up, xs["gated"])
            result = mid + self._linear(layer.mlp.down_proj, gated, torch.zeros_like(mid))
            return torch.autograd.grad(result, (x, attention, *parameters), xs["gradient"], allow_unused=True)

        values = dict(
            mid=rec.get("norm_mid", rec["mid"]).float(),
            norm2=rec["norm2"].float(),
            attention=rec["attention"],
            gate_up=rec["gate_up"],
            gated=rec["gated"],
            gradient=gradient,
        )
        if self.fp32_attention:
            values["attention"] = values["attention"].float()
        result = self._call(("post", index), run, values, rows=self.chunk)
        self._accumulate(parameters, result[2:])
        return result[0][: gradient.shape[0]], result[1][: gradient.shape[0]]

    def query(self, index, rec, gradient, rope, qwidth):
        layer = self.model.base.model.layers[index]
        attn, head = layer.self_attn, layer.self_attn.head_dim
        parameters = self._parameters(layer.input_layernorm, attn.q_proj, attn.q_norm)

        def run(xs):
            x = xs["input"].detach().requires_grad_()
            normalized = recorded_rms(x, layer.input_layernorm, xs["norm1"])
            raw = self._linear(attn.q_proj, normalized, xs["raw"])
            query = self._rotate(raw, attn.q_norm, xs["query"], xs["rope"], head)
            seed = xs["gradient"].view(1, self.chunk, -1, head).transpose(1, 2)
            return torch.autograd.grad(query, (x, *parameters), seed, allow_unused=True)

        count = rec["input"].shape[0]
        values = dict(
            input=rec.get("norm_input", rec["input"]).float(),
            norm1=rec["norm1"].float(),
            raw=rec["qkv"][:, :qwidth],
            query=rec["query"],
            rope=rope.float(),
            gradient=gradient.transpose(1, 2).reshape(count, -1),
        )
        if self.fp32_attention:
            values["raw"], values["query"] = (
                values["raw"].float(),
                values["query"].float(),
            )
        result = self._call(("query", index), run, values, rows=self.chunk)
        self._accumulate(parameters, result[1:])
        return result[0][:count]

    def kv(self, index, rec, dk, dv, rope, qwidth, kwidth):
        layer = self.model.base.model.layers[index]
        attn, head = layer.self_attn, layer.self_attn.head_dim
        parameters = self._parameters(layer.input_layernorm, attn.k_proj, attn.v_proj, attn.k_norm)

        def run(xs):
            x = xs["input"].detach().requires_grad_()
            normalized = recorded_rms(x, layer.input_layernorm, xs["norm1"])
            raw = self._linear(attn.k_proj, normalized, xs["raw_k"])
            key = self._rotate(raw, attn.k_norm, xs["key"], xs["rope"], head)
            value = self._linear(attn.v_proj, normalized, xs["raw_v"]).view(1, self.chunk, -1, head).transpose(1, 2)
            key_seed = xs["dk"].view(1, self.chunk, -1, head).transpose(1, 2)
            value_seed = xs["dv"].view(1, self.chunk, -1, head).transpose(1, 2)
            return torch.autograd.grad(
                (key, value),
                (x, *parameters),
                (key_seed, value_seed),
                allow_unused=True,
            )

        count = rec["input"].shape[0]
        values = dict(
            input=rec.get("norm_input", rec["input"]).float(),
            norm1=rec["norm1"].float(),
            raw_k=rec["qkv"][:, qwidth : qwidth + kwidth],
            raw_v=rec["qkv"][:, qwidth + kwidth :],
            key=rec["key"],
            rope=rope.float(),
            dk=dk.to(torch.float32 if self.fp32_attention else rec["key"].dtype).transpose(1, 2).reshape(count, -1),
            dv=dv.to(torch.float32 if self.fp32_attention else rec["key"].dtype).transpose(1, 2).reshape(count, -1),
        )
        if self.fp32_attention:
            for field in ("raw_k", "raw_v", "key"):
                values[field] = values[field].float()
        result = self._call(("kv", index), run, values, rows=self.chunk)
        self._accumulate(parameters, result[1:])
        return result[0][:count]

    def joint_qkv(self, index, rec, dq, dk, dv, rope, qwidth, kwidth):
        layer = self.model.base.model.layers[index]
        attn, head = layer.self_attn, layer.self_attn.head_dim
        parameters = self._parameters(
            layer.input_layernorm,
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            attn.q_norm,
            attn.k_norm,
        )

        def run(xs):
            x = xs["input"].detach().requires_grad_()
            normalized = recorded_rms(x, layer.input_layernorm, xs["norm1"])
            raw = recorded_qkv(normalized, attn, xs["raw"], self.model._recorded_qkv_weights[index][1])
            qraw, kraw, vraw = raw.split([qwidth, kwidth, kwidth], -1)
            query = self._rotate(qraw, attn.q_norm, torch.empty_like(qraw), xs["rope"], head)
            key = self._rotate(kraw, attn.k_norm, torch.empty_like(kraw), xs["rope"], head)
            value = vraw.view(1, self.chunk, -1, head).transpose(1, 2)
            seeds = tuple(xs[name].view(1, self.chunk, -1, head).transpose(1, 2) for name in ("dq", "dk", "dv"))
            return torch.autograd.grad((query, key, value), (x, *parameters), seeds, allow_unused=True)

        count = rec["input"].shape[0]
        dtype = torch.float32 if self.fp32_attention else rec["qkv"].dtype
        values = dict(
            input=rec.get("norm_input", rec["input"]).float(),
            norm1=rec["norm1"].float(),
            raw=rec["qkv"].to(dtype),
            rope=rope.float(),
            **{
                name: value.to(dtype).transpose(1, 2).reshape(count, -1)
                for name, value in zip(("dq", "dk", "dv"), (dq, dk, dv), strict=True)
            },
        )
        result = self._call(("joint_qkv", index), run, values, rows=self.chunk)
        self._accumulate(parameters, result[1:])
        return result[0][:count]

    def attention(self, q, k, v, output, lse, gradient, plan, prefix):
        kshape = (
            (*k[0].shape[:2], sum(value.shape[2] for value in k), *k[0].shape[3:]) if isinstance(k, list) else k.shape
        )
        key = (
            "attention",
            prefix,
            tuple(q.shape),
            tuple(kshape),
            plan.q_padding,
            plan.q_tail_padding,
            plan.k_indices.numel(),
            plan.source_indices is not None,
        )

        def run(xs):
            query, key, value = (xs[name].detach().requires_grad_() for name in ("q", "k", "v"))
            positions = replace(plan, q_indices=xs["qi"], k_indices=xs["ki"])
            result = recorded_attention(
                query,
                key,
                value,
                xs["out"],
                xs["lse"],
                positions,
                prefix,
                self.model.window,
                fp32_adjoint=self.fp32_attention,
            )
            return torch.autograd.grad(result, (query, key, value), xs["gradient"])

        return self._call(
            key,
            run,
            dict(
                q=q,
                k=k,
                v=v,
                out=output,
                lse=lse,
                gradient=gradient,
                qi=plan.q_indices,
                ki=plan.k_indices,
            ),
            dtypes=dict(q=torch.float32, k=torch.float32, v=torch.float32) if self.fp32_attention else None,
        )

    def head(self, raw, normalized, logits, targets):
        parameters = self._parameters(self.model.base.model.norm, self.model.base.lm_head)

        def run(xs):
            x = xs["raw"].detach().requires_grad_()
            normalized = recorded_rms(x, self.model.base.model.norm, xs["normalized"])
            logits = self._linear(self.model.base.lm_head, normalized, xs["logits"])
            loss = torch.nn.functional.cross_entropy(logits, xs["targets"], reduction="sum")
            return (
                loss.detach(),
                *torch.autograd.grad(loss, (x, *parameters), allow_unused=True),
            )

        result = self._call(
            ("head",),
            run,
            dict(
                raw=raw.float(),
                normalized=normalized.float(),
                logits=logits.float(),
                targets=targets,
            ),
            rows=self.chunk,
            fills={"targets": -100},
        )
        self._accumulate(parameters, result[2:])
        return result[0], result[1][: targets.numel()]
