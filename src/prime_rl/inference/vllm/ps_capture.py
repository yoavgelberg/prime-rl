"""Qwen3/vLLM recording adapter; enabled only by PSStreamingWorker."""

import torch
import triton
import triton.language as tl

from prime_rl.trainer.ps_backward.config import StreamConfig

_store = None
_installed = False


@triton.jit
def _assign_rows(
    starts,
    lengths,
    requests,
    slots,
    heads,
    overflow,
    capacity: tl.constexpr,
    BLOCK: tl.constexpr,
    RING: tl.constexpr = False,
    REQUESTS: tl.constexpr = 1,
    TOKEN_ROWS: tl.constexpr = 1,
    RESET_BLOCK: tl.constexpr = 1,
):
    row = tl.program_id(0)
    begin, end = tl.load(starts + row), tl.load(starts + row + 1)
    offset = tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    position = length - (end - begin) + offset
    request = tl.load(requests + row)
    valid = (offset < end - begin) & (request >= 0) & ((position < capacity) | RING)
    physical = position % capacity if RING else position
    tl.store(slots + begin + offset, tl.where(valid, request * capacity + physical, -1), offset < end - begin)
    head_valid = (end > begin) & (request >= 0) & ((length <= capacity) | RING)
    last = (length - 1) % capacity if RING else length - 1
    tl.store(heads + row, tl.where(head_valid, request * capacity + last, -1))
    if row == 0:
        padding = tl.arange(0, RESET_BLOCK)
        count = tl.load(starts + REQUESTS)
        tl.store(slots + padding, -1, (padding >= count) & (padding < TOKEN_ROWS))
    bad = (offset < end - begin) & (request >= 0) & (position >= capacity) & (not RING)
    if tl.sum(bad.to(tl.int32), 0) > 0:
        tl.atomic_or(overflow, 1)


@triton.jit
def _write_fields(
    pointers,
    strides: tl.constexpr,
    widths: tl.constexpr,
    target,
    slots,
    row_width: tl.constexpr,
    start: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = 0
    for field in tl.static_range(len(widths)):
        local = columns - base
        mask = (slot >= 0) & (local >= 0) & (local < widths[field])
        values = tl.load(pointers[field] + token * strides[field] + local, mask, 0)
        tl.store(target + slot * row_width + start + columns, values, mask)
        base += widths[field]


def _write(tensors, destination, slots, start=0):
    widths = tuple(t.shape[-1] for t in tensors)
    _write_fields[(tensors[0].shape[0], triton.cdiv(sum(widths), 512))](
        tuple(tensors),
        tuple(t.stride(0) for t in tensors),
        widths,
        destination,
        slots,
        destination.shape[-1],
        start,
        512,
    )


class CaptureStore:
    def __init__(self, config, device):
        hf = config.model_config.hf_config
        self.depth, self.hidden, self.intermediate = hf.num_hidden_layers, hf.hidden_size, hf.intermediate_size
        self.qwidth = hf.num_attention_heads * hf.head_dim
        self.kwidth = hf.num_key_value_heads * hf.head_dim
        self.config = StreamConfig(**config.additional_config["ps_streaming"])
        self.slots, self.capacity = self.config.max_requests, self.config.ring_tokens
        self.ring, self.compact = True, True
        self.policy_version = 0
        self.pending_lse = {}
        self.fields = (
            ("input", self.hidden),
            ("norm1", self.hidden),
            ("qkv", self.qwidth + 2 * self.kwidth),
            ("query", self.qwidth),
            ("key", self.kwidth),
            ("attention", self.qwidth),
            ("norm2", self.hidden),
            ("mid", self.hidden),
            ("gate_up", 2 * self.intermediate),
            ("gated", self.intermediate),
        )
        if self.compact:
            self.fields = tuple((name, size) for name, size in self.fields if name not in ("input", "mid", "gated"))
        self.field_offsets = {}
        offset = 0
        for name, size in self.fields:
            self.field_offsets[name] = offset
            offset += size
        width = sum(size for _, size in self.fields)
        with torch.inference_mode(False):
            self.layers = [
                torch.empty((self.slots * self.capacity, width), dtype=torch.bfloat16, device=device)
                for _ in range(self.depth)
            ]
            self.final = torch.empty((self.slots * self.capacity, 2 * self.hidden), dtype=torch.bfloat16, device=device)
            self.logits = torch.empty((self.slots * self.capacity, hf.vocab_size), dtype=torch.float32, device=device)
            self.lse = [
                torch.empty((self.slots * self.capacity, hf.num_attention_heads), dtype=torch.float32, device=device)
                for _ in range(self.depth)
            ]
            self.norm_inputs = [
                torch.empty((self.slots * self.capacity, self.hidden), dtype=torch.float32, device=device)
                for _ in range(2 * self.depth + 1)
            ]
            max_rows = max(
                config.scheduler_config.max_num_seqs, config.compilation_config.max_cudagraph_capture_size or 0
            )
            self.request_slots = torch.full((max_rows,), -1, device=device, dtype=torch.int32)
            self.head_slots = torch.full((max_rows,), -1, device=device, dtype=torch.int64)
            self.token_slots = torch.full(
                (config.scheduler_config.max_num_batched_tokens,), -1, device=device, dtype=torch.int64
            )
            self.overflow = torch.zeros((), device=device, dtype=torch.int32)
        self.active, self.last_requests, self.live = False, None, None
        self.requests = {}
        self.activation_dtype = self.final.dtype

    def head(self, request, start, end, first):
        offset = self.requests[request] * self.capacity
        return dict(
            raw=self.norm_inputs[-1][offset + start : offset + end],
            normalized=self.final[offset + start : offset + end, self.hidden :],
            logits=self.logits[offset + start + first : offset + end],
        )

    def bind(self, request_ids):
        if not self.active or tuple(request_ids) == self.last_requests:
            return
        values = []
        for request in request_ids:
            if request not in self.requests:
                if len(self.requests) == self.slots:
                    raise RuntimeError("PS recording slots exhausted; drain before admitting more requests")
                self.requests[request] = len(self.requests)
            values.append(self.requests[request])
        values += [-1] * (self.request_slots.numel() - len(values))
        self.request_slots.copy_(torch.tensor(values, dtype=torch.int32))
        self.last_requests = tuple(request_ids)

    def metadata(self, metadata):
        if self.active and self.live is not None and self.live.host_tape is not None:
            self.live.host_tape.before_write(self.live.runner)
        _assign_rows[(metadata.seq_lens.numel(),)](
            metadata.query_start_loc,
            metadata.seq_lens,
            self.request_slots,
            self.token_slots,
            self.head_slots,
            self.overflow,
            self.capacity,
            triton.next_power_of_2(metadata.max_query_len),
            self.ring,
            metadata.seq_lens.numel(),
            self.token_slots.numel(),
            triton.next_power_of_2(self.token_slots.numel()),
        )

    def report(self):
        torch.cuda.synchronize(self.final.device)
        if self.overflow.item():
            raise RuntimeError("PS activation tape capacity exceeded; no gradient may be accepted")
        return dict(
            requests=self.requests,
            capacity=self.capacity,
            policy_version=self.policy_version,
            overflow=bool(self.overflow.item()),
            allocated_bytes=sum(
                t.numel() * t.element_size()
                for t in [*self.layers, *self.lse, *self.norm_inputs, self.final, self.logits]
            ),
            streaming_backward_integrated=self.live is not None,
            ring_recording=self.ring,
            compact_activation_storage=self.compact,
        )

    def layer(self, request, index, start, end):
        row = self.requests[request] * self.capacity
        data, result, offset = self.layers[index][row + start : row + end], {}, 0
        for name, width in self.fields:
            result[name] = data[:, offset : offset + width]
            offset += width
        result["lse"] = self.lse[index][row + start : row + end]
        result["norm_input"] = self.norm_inputs[2 * index][row + start : row + end]
        result["norm_mid"] = self.norm_inputs[2 * index + 1][row + start : row + end]
        return result


def prepare_metadata(metadata):
    if _store is not None:
        _store.metadata(metadata)


def _fused_layer(self, positions, hidden_states, residual):
    from .ps_capture_kernels import fused_norm_record, fused_silu_record

    hidden_states, residual = fused_norm_record(
        self.ps_layer_index,
        0,
        hidden_states,
        residual,
        self.input_layernorm.weight,
        self.input_layernorm.variance_epsilon,
    )
    attn = self.self_attn
    qkv, _ = attn.qkv_proj(hidden_states)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.view(-1, attn.num_heads, attn.head_dim)).view(q.shape)
    k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim)).view(k.shape)
    q, k = attn.rotary_emb(positions, q, k)
    attended = attn.attn(q, k, v)
    projected, _ = attn.o_proj(attended)
    normalized, residual = fused_norm_record(
        self.ps_layer_index,
        1,
        projected,
        residual,
        self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon,
        qkv,
        q,
        k,
        attended,
    )
    gate_up, _ = self.mlp.gate_up_proj(normalized)
    gated = fused_silu_record(self.ps_layer_index, gate_up)
    hidden_states, _ = self.mlp.down_proj(gated)
    return hidden_states, residual


def attention_normalizers(layer_name, lse):
    if _store is not None:
        index = int(layer_name.split(".layers.")[1].split(".")[0])
        _store.pending_lse[index] = lse


def install():
    """Install before Qwen3 model construction in a dedicated streaming worker."""
    global _installed
    if _installed:
        return
    from vllm.model_executor.models.qwen3 import (
        Qwen3DecoderLayer,
        Qwen3ForCausalLM,
        Qwen3Model,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_init = Qwen3Model.__init__
    original_update = GPUModelRunner._update_states
    original_logits = Qwen3ForCausalLM.compute_logits
    original_sample = GPUModelRunner._sample

    def init(self, *, vllm_config, prefix=""):
        global _store
        _store = CaptureStore(vllm_config, torch.device("cuda", torch.cuda.current_device()))
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        for index, layer in enumerate(self.layers):
            layer.ps_layer_index = index
        from .ps_capture_kernels import fused_norm_record

        norm = self.norm

        def final_forward(hidden, residual=None):
            return fused_norm_record(0, 2, hidden, residual, norm.weight, norm.variance_epsilon)

        self.norm.forward = final_forward

    def update(self, scheduler_output):
        if _store.active and any(req.num_computed_tokens for req in scheduler_output.scheduled_new_reqs):
            raise ValueError("Recorded backward requires a fresh prefill, without prefix-cache reuse")
        original_update(self, scheduler_output)
        _store.bind(self.input_batch.req_ids)

    def logits(self, hidden_states):
        result = original_logits(self, hidden_states)
        _write((result,), _store.logits, _store.head_slots)
        return result

    def sample(self, logits, spec_decode_metadata):
        result = original_sample(self, logits, spec_decode_metadata)
        if _store.active and _store.live is not None:
            _store.live.observe(self, result, spec_decode_metadata)
        return result

    Qwen3Model.__init__ = init
    Qwen3DecoderLayer.forward = _fused_layer
    GPUModelRunner._update_states = update
    Qwen3ForCausalLM.compute_logits = logits
    GPUModelRunner._sample = sample
    _installed = True
