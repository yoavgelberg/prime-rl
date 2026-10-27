"""Event-protected activation pools whose capacity depends on PS depth/window, not trace length."""

import ctypes
import math
import threading
import time
import weakref
from collections import deque
from pathlib import Path

import torch


class RowCopies:
    def __init__(self, mode="dma"):
        if mode not in ("dma", "sm"):
            raise ValueError("Activation transfer mode must be dma or sm")
        self.mode = mode
        paths = {
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/libcudart.so" in line
        }
        packaged = {path for path in paths if Path(torch.__file__).resolve().parents[1] in Path(path).parents}
        if packaged:
            paths = packaged
        if len(paths) != 1:
            raise RuntimeError(f"Expected one loaded CUDA runtime, found {paths}")
        self.runtime = ctypes.CDLL(paths.pop())
        self.copy = self.runtime.cudaMemcpy2DAsync
        self.copy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.copy.restype = ctypes.c_int
        self.runtime.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.runtime.cudaHostAlloc.restype = ctypes.c_int
        self.runtime.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.runtime.cudaFreeHost.restype = ctypes.c_int
        self.bytes = dict(h2d=0, d2h=0)

    def allocate(self, shape, dtype):
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        address = ctypes.c_void_p()
        status = self.runtime.cudaHostAlloc(ctypes.byref(address), size, 0)
        if status:
            raise RuntimeError(f"cudaHostAlloc({size} bytes) failed with CUDA error {status}")
        buffer = (ctypes.c_byte * size).from_address(address.value)
        cleanup = weakref.finalize(buffer, self.runtime.cudaFreeHost, address)
        cleanup.atexit = False
        return torch.frombuffer(buffer, dtype=dtype).view(shape)

    def __call__(self, destination, source, *, mode=None):
        if destination.shape != source.shape or destination.dtype != source.dtype or source.ndim != 2:
            raise ValueError("DMA requires matching two-dimensional tensors")
        if destination.stride(1) != 1 or source.stride(1) != 1:
            raise ValueError("DMA columns must be contiguous")
        if destination.is_cuda == source.is_cuda:
            raise ValueError("Activation DMA must connect pinned host and CUDA memory")
        host = source if destination.is_cuda else destination
        if not host.is_pinned():
            raise ValueError("Asynchronous activation DMA requires owned pinned storage")
        size = source.element_size()
        direction, kind = ("h2d", 1) if destination.is_cuda else ("d2h", 2)
        mode = self.mode if mode is None else mode
        if mode not in ("dma", "sm"):
            raise ValueError("Activation transfer mode must be dma or sm")
        if mode == "sm":
            from .ps_mapped_copy import mapped_copy

            mapped_copy(destination, source)
        else:
            status = self.copy(
                destination.data_ptr(),
                destination.stride(0) * size,
                source.data_ptr(),
                source.stride(0) * size,
                source.shape[1] * size,
                source.shape[0],
                kind,
                torch.cuda.current_stream().cuda_stream,
            )
            if status:
                raise RuntimeError(f"cudaMemcpy2DAsync failed with CUDA error {status}")
        self.bytes[direction] += source.numel() * size


class _Pool:
    def __init__(self, slots, rows, specs, allocate):
        self.buffers = {name: allocate((slots, rows, width), dtype) for name, (width, dtype) in specs.items()}
        self.free = deque((index, None) for index in range(slots))
        self.condition = threading.Condition()
        self.allocated_bytes = sum(t.numel() * t.element_size() for t in self.buffers.values())
        self.wait_seconds, self.peak_in_use = 0.0, 0
        self.slots = slots

    def acquire(self, rows, check):
        begin = time.perf_counter()
        with self.condition:
            while not self.free:
                check()
                if time.perf_counter() - begin > 60:
                    raise RuntimeError("Activation pool stalled; refusing to overwrite an unconsumed block")
                self.condition.wait(0.05)
            index, released = self.free.popleft()
            self.peak_in_use = max(self.peak_in_use, self.slots - len(self.free))
        self.wait_seconds += time.perf_counter() - begin
        return dict(
            index=index, released=released, values={name: value[index, :rows] for name, value in self.buffers.items()}
        )

    def release(self, record):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        with self.condition:
            self.free.append((record["index"], event))
            self.condition.notify()


class PinnedActivationTape:
    def __init__(self, source, chunk, window, check, reuse_gpu=True, copy_mode="dma", independent_reads=False):
        self.source, self.chunk, self.check = source, chunk, check
        self.on_publish = None
        self.hidden, self.qwidth, self.kwidth = source.hidden, source.qwidth, source.kwidth
        self.activation_dtype, self.policy_version = source.activation_dtype, source.policy_version
        self.copies = RowCopies(copy_mode)
        self.stream = torch.cuda.Stream(device=source.final.device)
        if independent_reads and not reuse_gpu:
            raise ValueError("Independent ring snapshots require the GPU tape cache")
        self.read_stream = torch.cuda.Stream(device=source.final.device, priority=-1) if independent_reads else None
        self.generation_ready = {}
        self.layer_records, self.head_records, self.pending_copies, self.prefixes = {}, {}, {}, {}
        self.reuse_gpu = reuse_gpu
        self.ring_lock = threading.Lock()
        self.latest_end, self.ring_reads = {}, {}
        self.ring_hit_bytes, self.ring_hits = 0, 0
        self.ring_waits = 0
        halo = (window - 1 + chunk - 1) // chunk
        dtype = source.activation_dtype
        layer_specs = dict(
            data=(source.layers[0].shape[1], dtype),
            norm_input=(source.hidden, torch.float32),
            norm_mid=(source.hidden, torch.float32),
            lse=(source.lse[0].shape[1], torch.float32),
        )
        head_specs = dict(
            raw=(source.hidden, torch.float32),
            normalized=(source.hidden, dtype),
            logits=(source.logits.shape[1], torch.float32),
            inputs=(1, torch.long),
            targets=(1, torch.long),
        )
        with torch.inference_mode(False):
            self.layer_pools = [
                _Pool(source.slots * ((source.depth - index) * halo + 5), chunk, layer_specs, self.copies.allocate)
                for index in range(source.depth)
            ]
            self.head_pool = _Pool(source.slots * 4, chunk, head_specs, self.copies.allocate)

    @staticmethod
    def _pieces(start, end, capacity):
        position = start
        while position < end:
            count = min(end - position, capacity - position % capacity)
            yield position - start, position % capacity, count
            position += count

    def before_write(self, runner):
        current = torch.cuda.current_stream()
        with self.ring_lock:
            for row, request in enumerate(runner.input_batch.req_ids):
                end = int(runner.seq_lens.np[row])
                pending = self.pending_copies.get(request, [])
                keep = []
                for start, event in pending:
                    if end > start + self.source.capacity:
                        current.wait_event(event)
                        self.ring_waits += 1
                    else:
                        keep.append((start, event))
                self.pending_copies[request] = keep
                reads = self.ring_reads.get(request, {})
                for start in list(reads):
                    if end > start + self.source.capacity:
                        current.wait_event(reads.pop(start))
                        self.ring_waits += 1
                self.latest_end[request] = end

    def _read_ring(self, request, start, end, sources):
        if not self.reuse_gpu:
            return None
        with self.ring_lock:
            latest = self.latest_end.get(request, 0)
            if latest < end or latest > start + self.source.capacity:
                return None
            offset = self.source.requests[request] * self.source.capacity
            consumer = torch.cuda.current_stream()
            copying = self.read_stream if self.read_stream is not None else consumer
            with torch.cuda.stream(copying):
                if self.read_stream is not None:
                    prefix = self.prefixes[request]
                    block_start = 0 if start < prefix else prefix + (start - prefix) // self.chunk * self.chunk
                    copying.wait_event(self.generation_ready[(request, block_start)])
                result = {
                    name: torch.empty((end - start, source.shape[1]), device=source.device, dtype=source.dtype)
                    for name, source in sources.items()
                }
                for target, position, count in self._pieces(start, end, self.source.capacity):
                    for name, source in sources.items():
                        destination = result[name][target : target + count]
                        value = source[offset + position : offset + position + count]
                        if self.read_stream is None:
                            destination.copy_(value)
                        else:
                            from .ps_mapped_copy import mapped_copy

                            mapped_copy(destination, value)
                copied = torch.cuda.Event()
                copied.record(copying)
            if self.read_stream is not None:
                consumer.wait_event(copied)
                for value in result.values():
                    value.record_stream(consumer)
            self.ring_reads.setdefault(request, {})[start] = copied
            self.ring_hit_bytes += sum(value.numel() * value.element_size() for value in result.values())
            self.ring_hits += 1
            return result

    def publish(self, request, start, end, ready, inputs, targets):
        if end - start > self.chunk:
            raise ValueError("Pinned activation blocks, including the prompt, must fit --stream-chunk")
        if request not in self.prefixes:
            if start != 0:
                raise ValueError("The first published block must contain the prompt")
            self.prefixes[request] = end
        self.generation_ready[(request, start)] = ready
        layers = [pool.acquire(end - start, self.check) for pool in self.layer_pools]
        head = self.head_pool.acquire(end - start, self.check)
        self.layer_records[(request, start)] = layers
        self.head_records[(request, start)] = head
        offset = self.source.requests[request] * self.source.capacity
        with torch.inference_mode(False), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            for record in [*layers, head]:
                if record["released"] is not None:
                    self.stream.wait_event(record["released"])
            for destination_start, source_start, count in self._pieces(start, end, self.source.capacity):
                src = slice(offset + source_start, offset + source_start + count)
                dst = slice(destination_start, destination_start + count)
                for index, record in enumerate(layers):
                    values = record["values"]
                    for name, buffer in (
                        ("data", self.source.layers[index]),
                        ("norm_input", self.source.norm_inputs[2 * index]),
                        ("norm_mid", self.source.norm_inputs[2 * index + 1]),
                        ("lse", self.source.lse[index]),
                    ):
                        self.copies(values[name][dst], buffer[src])
                values = head["values"]
                for name, buffer in (
                    ("raw", self.source.norm_inputs[-1]),
                    ("normalized", self.source.final[:, self.hidden :]),
                    ("logits", self.source.logits),
                    ("inputs", inputs),
                    ("targets", targets),
                ):
                    self.copies(values[name][dst], buffer[src])
            if self.on_publish is not None:
                self.on_publish(request, start, end)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.pending_copies.setdefault(request, []).append((start, event))
        return event

    def tokens(self, request, start):
        values = self.head_records[(request, start)]["values"]
        result = []
        for name in ("inputs", "targets"):
            tensor = torch.empty(values[name].shape, dtype=torch.long, device=self.source.final.device)
            self.copies(tensor, values[name])
            result.append(tensor[:, 0])
        return tuple(result)

    def head(self, request, start, end, first):
        cached = self._read_ring(
            request,
            start,
            end,
            dict(
                raw=self.source.norm_inputs[-1],
                normalized=self.source.final[:, self.hidden :],
                logits=self.source.logits,
            ),
        )
        if cached is not None:
            cached["logits"] = cached["logits"][first:]
            return cached
        values = self.head_records[(request, start)]["values"]
        result = {}
        for name in ("raw", "normalized", "logits"):
            source = values[name][first:] if name == "logits" else values[name]
            result[name] = torch.empty(source.shape, dtype=source.dtype, device=self.source.final.device)
            self.copies(result[name], source)
        return result

    def load_layer(self, request, index, start, end, device, fields=None, *, copy_mode=None):
        if self.reuse_gpu:
            buffers, offset = {}, 0
            if fields is None:
                buffers["data"] = self.source.layers[index]
            else:
                for name, width in self.source.fields:
                    if name in fields:
                        buffers[name] = self.source.layers[index][:, offset : offset + width]
                    offset += width
            for name, source in (
                ("norm_input", self.source.norm_inputs[2 * index]),
                ("norm_mid", self.source.norm_inputs[2 * index + 1]),
                ("lse", self.source.lse[index]),
            ):
                if fields is None or name in fields:
                    buffers[name] = source
            cached = self._read_ring(request, start, end, buffers)
            if cached is not None:
                if fields is None:
                    data, offset = cached.pop("data"), 0
                    for name, width in self.source.fields:
                        cached[name] = data[:, offset : offset + width]
                        offset += width
                return cached
        prefix = self.prefixes[request]
        block_start = 0 if start < prefix else prefix + (start - prefix) // self.chunk * self.chunk
        record = self.layer_records[(request, block_start)][index]["values"]
        if end > block_start + record["data"].shape[0]:
            raise ValueError("A layer read must stay within its recorded block")
        rows, offset, result = slice(start - block_start, end - block_start), 0, {}
        data = None
        if fields is None:
            data = torch.empty(record["data"][rows].shape, dtype=self.activation_dtype, device=device)
            self.copies(data, record["data"][rows], mode=copy_mode)
        for name, width in self.source.fields:
            if fields is None or name in fields:
                if data is not None:
                    result[name] = data[:, offset : offset + width]
                else:
                    value = record["data"][rows, offset : offset + width]
                    result[name] = torch.empty(value.shape, dtype=value.dtype, device=device)
                    self.copies(result[name], value, mode=copy_mode)
            offset += width
        for name in ("norm_input", "norm_mid", "lse"):
            if fields is None or name in fields:
                value = record[name][rows]
                result[name] = torch.empty(value.shape, dtype=value.dtype, device=device)
                self.copies(result[name], value, mode=copy_mode)
        return result

    def release_head(self, request, start, end):
        self.head_pool.release(self.head_records.pop((request, start)))

    def release_layer(self, request, index, start, end):
        records = self.layer_records[(request, start)]
        self.layer_pools[index].release(records[index])
        records[index] = None
        if all(record is None for record in records):
            del self.layer_records[(request, start)]
            del self.generation_ready[(request, start)]

    def report(self):
        pools = [*self.layer_pools, self.head_pool]
        return dict(
            pinned_activation_bytes=sum(p.allocated_bytes for p in pools),
            transfer_mode=self.copies.mode,
            host_allocator="cudaHostAlloc with exact requested sizes, CUDA page rounding only",
            host_pool_wait_seconds=sum(p.wait_seconds for p in pools),
            layer_pool_slots=[p.slots for p in self.layer_pools],
            peak_layer_pool_usage=[p.peak_in_use for p in self.layer_pools],
            dma_bytes=self.copies.bytes,
            gpu_ring_tokens_per_request=self.source.capacity,
            gpu_ring_cache_enabled=self.reuse_gpu,
            gpu_ring_cache_hits=self.ring_hits,
            independent_ring_copies=self.read_stream is not None,
            gpu_ring_cache_bytes=self.ring_hit_bytes,
            gpu_ring_reuse_waits=self.ring_waits,
            live_layer_blocks=sum(
                sum(record is not None for record in records) for records in self.layer_records.values()
            ),
            live_head_blocks=len(self.head_records),
            bounded_working_pools=True,
            bounded_activation_storage=True,
        )

    def reset(self, check):
        if self.layer_records or self.head_records:
            raise RuntimeError("Cannot reuse pinned pools with live activation records")
        self.check = check
        self.pending_copies.clear()
        self.generation_ready.clear()
        self.prefixes.clear()
        self.latest_end.clear()
        self.ring_reads.clear()
        self.ring_hit_bytes, self.ring_hits = 0, 0
        self.ring_waits = 0
        self.copies.bytes = dict(h2d=0, d2h=0)
        for pool in [*self.layer_pools, self.head_pool]:
            if len(pool.free) != pool.slots:
                raise RuntimeError("Pinned activation pool leaked a slot")
            pool.wait_seconds, pool.peak_in_use = 0.0, 0
