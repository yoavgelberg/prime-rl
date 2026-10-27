"""Run fixed-policy recorded VJPs concurrently with the native vLLM sampler."""

import queue
import threading
import time
import traceback

import torch

from prime_rl.trainer.ps_backward.fa4 import compile_cache_sizes
from prime_rl.trainer.ps_backward.recorded_graphs import RecordedGraphCache
from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream

from .ps_host_tape import PinnedActivationTape


class LiveBackward:
    def __init__(self, tape, runner, model, *, warmup=False, graph_cache=None, host_tape=None, cuda_stream=None):
        self.tape, self.runner, self.model = tape, runner, model
        self.config, self.warmup = tape.config, warmup
        self.chunk = self.config.chunk_tokens
        model.recorded_fp32_attention = self.config.fp32_attention
        with torch.inference_mode(False), torch.enable_grad():
            self.inputs = torch.empty((tape.slots * tape.capacity, 1), device=model.device, dtype=torch.long)
            self.targets = torch.empty_like(self.inputs)
        self.graph_cache = graph_cache if graph_cache is not None else RecordedGraphCache(model, self.chunk)
        decoder = runner.model.runnable if hasattr(runner.model, "runnable") else runner.model
        self.ropes = [layer.self_attn.rotary_emb.cos_sin_cache for layer in decoder.model.layers]
        self.cuda_stream = cuda_stream if cuda_stream is not None else torch.cuda.Stream(device=model.device)
        self.cuda_stream.wait_stream(torch.cuda.current_stream())
        self.tasks = queue.Queue(maxsize=2 * tape.slots)
        self.streams, self.submitted, self.observed, self.prefixes = {}, {}, {}, {}
        self.finished_requests = set()
        self.error, self.closed = None, False
        if host_tape is None:
            self.host_tape = PinnedActivationTape(
                tape, self.chunk, model.window, self._check, reuse_gpu=True, copy_mode="sm", independent_reads=True
            )
        else:
            self.host_tape = host_tape
            host_tape.reset(self._check)
        self.published = {}
        self.completed_blocks, self.cpu_backpressure_seconds = 0, 0.0
        self.compiles_before = compile_cache_sizes()
        self.worker = threading.Thread(target=self._work, name="ps-recorded-backward", daemon=True)
        self.worker.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError("Production PS backward failed:\n" + self.error)
        if self.closed:
            raise RuntimeError("Production PS backward is already closed")

    def _submit(self, item):
        self._check()
        start = time.perf_counter()
        self.tasks.put(item)
        self.cpu_backpressure_seconds += time.perf_counter() - start

    def observe(self, runner, sampler_output, spec_decode_metadata):
        from .ps_capture import _write

        self._check()
        if spec_decode_metadata is not None or sampler_output.sampled_token_ids.shape[1] != 1:
            raise ValueError("Initial recorded streaming integration requires ordinary autoregressive sampling")
        count = int(runner.query_start_loc.np[runner.input_batch.num_reqs])
        _write((runner.input_ids.gpu[:count, None],), self.inputs, self.tape.token_slots)
        _write((sampler_output.sampled_token_ids,), self.targets, self.tape.head_slots)
        ready = []
        for row, request in enumerate(runner.input_batch.req_ids):
            state = runner.requests[request]
            if request not in self.prefixes:
                params = state.sampling_params
                if (
                    params.temperature != 1.0
                    or params.top_p != 1.0
                    or params.top_k not in (-1, 0)
                    or params.repetition_penalty != 1.0
                    or params.presence_penalty != 0.0
                    or params.frequency_penalty != 0.0
                    or params.min_p != 0.0
                    or params.min_tokens != 0
                    or params.logits_processors
                    or params.logit_bias
                    or params.allowed_token_ids
                    or params.bad_words
                    or params.structured_outputs
                ):
                    raise ValueError("Score reuse currently requires unfiltered unit-temperature sampling")
                if state.num_prompt_tokens > self.chunk:
                    raise ValueError("The prompt must fit chunk_tokens; increase the chunk and ring sizes")
                self.prefixes[request] = state.num_prompt_tokens
                self.submitted[request] = 0
            end = int(runner.seq_lens.np[row])
            self.observed[request] = end
            prefix, submitted = self.prefixes[request], self.submitted[request]
            if end < prefix or bool(runner.discard_request_mask.np[row]):
                continue
            if submitted == 0:
                ready.append((request, prefix))
                submitted = prefix
            while submitted + self.chunk <= end:
                submitted += self.chunk
                ready.append((request, submitted))
            self.submitted[request] = submitted
        if ready:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            for request, end in ready:
                self._publish(request, end, event)

    def _publish(self, request, end, event):
        if self.host_tape is not None:
            event = self.host_tape.publish(
                request, self.published.get(request, 0), end, event, self.inputs, self.targets
            )
        self.published[request] = end
        self._submit(("block", request, end, event))

    def _work(self):
        torch.cuda.set_device(self.model.device)
        with torch.inference_mode(False), torch.enable_grad(), torch.cuda.stream(self.cuda_stream):
            while True:
                item = self.tasks.get()
                try:
                    if item is None:
                        return
                    if self.error is not None:
                        continue
                    kind, request, end, event = item
                    self.cuda_stream.wait_event(event)
                    if request not in self.streams:
                        self.streams[request] = RecordedStream(
                            self.model,
                            self.host_tape,
                            request,
                            self.prefixes[request],
                            self.ropes,
                            chunk=self.chunk,
                            policy_version=self.tape.policy_version,
                            graph_cache=self.graph_cache,
                            tail_block_tokens=self.config.tail_block_tokens,
                            tail_prefetch="dma",
                        )
                    stream = self.streams[request]
                    if kind == "block":
                        inputs, targets = self.host_tape.tokens(request, stream.schedule.end)
                        stream.append(end, inputs, targets, policy_version=self.tape.policy_version)
                        self.completed_blocks += 1
                    elif kind == "finish":
                        if self.warmup:
                            from prime_rl.trainer.ps_backward.recorded_tail import (
                                warm_layerwise,
                            )

                            warm_layerwise(stream, policy_version=self.tape.policy_version)
                        stream.finish(policy_version=self.tape.policy_version, layerwise=not self.warmup)
                    else:
                        raise ValueError(kind)
                except BaseException:
                    # Propagate worker failures to the sampler and drain queued tasks.
                    self.error = traceback.format_exc()
                finally:
                    self.tasks.task_done()

    def finish_request(self, request):
        self._check()
        if request in self.finished_requests:
            raise ValueError("Request finalization must arrive exactly once")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        end = self.observed[request]
        if self.submitted[request] < end:
            self._publish(request, end, event)
        self._submit(("finish", request, end, event))
        self.finished_requests.add(request)

    def finish(self):
        if self.closed:
            raise RuntimeError("Streaming backward is already closed")
        start = time.perf_counter()
        try:
            self._check()
            for request in self.observed:
                if request not in self.finished_requests:
                    self.finish_request(request)
        finally:
            self.tasks.join()
            self.cuda_stream.synchronize()
            self.tasks.put(None)
            self.tasks.join()
            self.worker.join()
            self.closed = True
        if self.error is not None:
            raise RuntimeError("Streaming backward failed:\n" + self.error)
        return dict(
            drain_seconds=time.perf_counter() - start,
            completed_blocks=self.completed_blocks,
            cpu_backpressure_seconds=self.cpu_backpressure_seconds,
            fa4_new_cache_entries=[
                after - before for before, after in zip(self.compiles_before, compile_cache_sizes(), strict=True)
            ],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_graph_replays=self.graph_cache.replays,
            eager_stage_calls=self.graph_cache.eager_calls,
            activation_storage=self.host_tape.report(),
            requests={
                request: dict(
                    loss=(stream.loss / stream.loss_tokens).item(),
                    loss_tokens=stream.loss_tokens,
                    peak_live_token_layers=stream.schedule.peak_live_token_layers,
                    retired_tokens=stream.schedule.retired_tokens,
                )
                for request, stream in self.streams.items()
            },
        )
