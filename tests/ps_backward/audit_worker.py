"""Unbounded same-generation-value audit, imported only by GPU validation."""

import math

import torch
from reference import backward_recorded

from prime_rl.inference.vllm.ps_streaming_worker import PSStreamingWorker
from prime_rl.trainer.ps_backward.recorded import unfused_recorded_pointwise
from prime_rl.trainer.ps_backward.types import Trace


class FullAuditTape:
    """Explicitly unbounded validation archive, never enabled in performance runs."""

    def __init__(self, source, capacity, allocate):
        self.capacity, self.requests = capacity, source.requests
        self.hidden, self.fields = source.hidden, source.fields
        self.qwidth, self.kwidth = source.qwidth, source.kwidth

        def buffer(tensor):
            return allocate((source.slots * capacity, tensor.shape[1]), tensor.dtype)

        self.layers = [buffer(tensor) for tensor in source.layers]
        self.norm_inputs = [buffer(tensor) for tensor in source.norm_inputs]
        self.lse = [buffer(tensor) for tensor in source.lse]
        self.final, self.logits = buffer(source.final), buffer(source.logits)
        self.buffers = [*self.layers, *self.norm_inputs, *self.lse, self.final, self.logits]
        self.allocated_bytes = sum(t.numel() * t.element_size() for t in self.buffers)

    def layer(self, request, index, start, end):
        offset = self.requests[request] * self.capacity
        data, result, column = self.layers[index][offset + start : offset + end], {}, 0
        for name, width in self.fields:
            result[name] = data[:, column : column + width]
            column += width
        result.update(
            norm_input=self.norm_inputs[2 * index][offset + start : offset + end],
            norm_mid=self.norm_inputs[2 * index + 1][offset + start : offset + end],
            lse=self.lse[index][offset + start : offset + end],
        )
        return result

    def attach(self, host_tape):
        source, copies = host_tape.source, host_tape.copies

        def save(request, start, end):
            if end > self.capacity:
                raise ValueError("Audit capacity exceeded")
            offset = source.requests[request] * source.capacity
            for target, position, count in host_tape._pieces(start, end, source.capacity):
                src = slice(offset + position, offset + position + count)
                dst_start = source.requests[request] * self.capacity + start + target
                buffers = [*source.layers, *source.norm_inputs, *source.lse, source.final, source.logits]
                for destination, buffer in zip(self.buffers, buffers, strict=True):
                    copies(destination[dst_start : dst_start + count], buffer[src])

        host_tape.on_publish = save


class AuditWorker(PSStreamingWorker):
    def ps_stream_finish(self):
        result = super().ps_stream_finish()
        if hasattr(self, "audit"):
            result["activation_storage"].update(
                bounded_activation_storage=False, full_audit_bytes=self.audit.allocated_bytes
            )
        return result

    def ps_stream_reset(self):
        result = super().ps_stream_reset()
        from prime_rl.inference.vllm.ps_capture import _store

        live = _store.live
        self.audit = FullAuditTape(_store, self.vllm_config.model_config.max_model_len, live.host_tape.copies.allocate)
        self.audit.attach(live.host_tape)
        return result

    def ps_stream_audit(self, traces):
        from prime_rl.inference.vllm.ps_capture import _store

        live = _store.live
        if not live.closed:
            raise RuntimeError("Drain streaming gradients before auditing")
        results = []
        with torch.inference_mode(False), torch.enable_grad(), unfused_recorded_pointwise():
            for item in traces:
                request = next(
                    key for key in live.streams if key == item["request_id"] or key.startswith(item["request_id"] + "-")
                )
                expected = live.streams[request].gradients
                missing = [name for name, value in expected.items() if value is None]
                if missing:
                    raise RuntimeError(f"Streaming backward omitted parameter gradients: {missing}")
                trace = Trace(torch.tensor(item["input_ids"], device=live.model.device), item["prefix_length"])
                live.model.zero_grad(set_to_none=True)
                loss = backward_recorded(
                    live.model, trace, self.audit, request, live.ropes, fp32_boundaries=True, head_block_tokens=1024
                )
                difference, ref_norm, actual_norm, dot = (
                    torch.zeros((), device=live.model.device, dtype=torch.float64) for _ in range(4)
                )
                for name, parameter in live.model.named_parameters():
                    reference, actual = parameter.grad.double(), expected[name].double()
                    difference += (actual - reference).square().sum()
                    ref_norm += reference.square().sum()
                    actual_norm += actual.square().sum()
                    dot += (actual * reference).sum()
                relative = math.sqrt((difference / ref_norm).item())
                cosine = (dot / (ref_norm * actual_norm).sqrt()).item()
                row = dict(
                    request=request,
                    loss=loss,
                    relative_l2=relative,
                    cosine=cosine,
                    passed=relative < 0.01 and cosine >= 0.9999,
                )
                results.append(row)
                live.model.zero_grad(set_to_none=True)
        if not all(row["passed"] for row in results):
            raise RuntimeError(f"Same-value full-gradient audit failed: {results}")
        return results
