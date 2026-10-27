"""Run fixed-policy recorded VJPs concurrently with the native vLLM sampler."""

import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import torch

from prime_rl.trainer.ps_backward.fa4 import compile_cache_sizes
from prime_rl.trainer.ps_backward.recorded_graphs import RecordedGraphCache
from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream

from .ps_host_tape import PinnedActivationTape


@dataclass
class BackwardStage:
    action: str
    blocks: list
    layer: int
    run: object
    wait: object = None
    end: int | None = None


class LiveBackward:
    def __init__(
        self,
        tape,
        runner,
        model,
        *,
        warmup=False,
        graph_cache=None,
        host_tape=None,
        cuda_stream=None,
    ):
        self.tape, self.runner, self.model = tape, runner, model
        self.config, self.warmup = tape.config, warmup
        self.chunk = self.config.chunk_tokens
        model.recorded_fp32_attention = self.config.fp32_attention
        model.recorded_joint_qkv = self.config.joint_qkv
        model.recorded_direct_weight_gradients = self.config.direct_weight_gradients
        model.recorded_weight_gradient_backend = self.config.weight_gradient_backend
        model.recorded_compact_kv = self.config.compact_kv or self.config.kv_cache
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
                tape,
                self.chunk,
                model.window,
                self._check,
                reuse_gpu=True,
                copy_mode=self.config.copy_mode,
                independent_reads=True,
                copy_programs=self.config.copy_programs,
                head_on_gpu=self.config.gpu_head_staging,
                fine_grained_events=self.config.fine_grained_events,
            )
        else:
            self.host_tape = host_tape
            host_tape.reset(self._check)
        self.published = {}
        self.completed_blocks, self.cpu_backpressure_seconds = 0, 0.0
        self.compiles_before = compile_cache_sizes()
        self.gpu_credit_wait_seconds = 0.0
        self.stage_counts = {}
        self.peak_gpu_stages = 0
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
                request,
                self.published.get(request, 0),
                end,
                event,
                self.inputs,
                self.targets,
            )
        self.published[request] = end
        self._submit(("block", request, end, event))

    def _work(self):
        torch.cuda.set_device(self.model.device)
        if self.config.stage_scheduling:
            self._work_stages()
            return
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.cuda.stream(self.cuda_stream),
        ):
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
                            kv_working_cache=self.config.kv_cache,
                        )
                    stream = self.streams[request]
                    if kind == "block":
                        inputs, targets = self.host_tape.tokens(request, stream.schedule.end)
                        stream.append(
                            end,
                            inputs,
                            targets,
                            policy_version=self.tape.policy_version,
                        )
                        self.completed_blocks += 1
                    elif kind == "finish":
                        if self.warmup:
                            from prime_rl.trainer.ps_backward.recorded_tail import (
                                warm_layerwise,
                            )

                            warm_layerwise(stream, policy_version=self.tape.policy_version)
                        stream.finish(
                            policy_version=self.tape.policy_version,
                            layerwise=not self.warmup,
                        )
                    else:
                        raise ValueError(kind)
                except BaseException:
                    # Propagate worker failures to the sampler and drain queued tasks.
                    self.error = traceback.format_exc()
                finally:
                    self.tasks.task_done()

    def _request_stream(self, request):
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
                kv_working_cache=self.config.kv_cache,
            )
        return self.streams[request]

    def _stages(self, kind, request, end):
        stream = self._request_stream(request)
        if kind == "block":
            for stop in end if isinstance(end, tuple) else (end,):

                def head(stop=stop):
                    tokens = self.host_tape.tokens(request, stream.schedule.end)
                    stream.append(stop, *tokens, policy_version=self.tape.policy_version, advance=False)
                    self.completed_blocks += 1

                yield BackwardStage("head", [], self.model.depth, head, end=stop)
            transitions = stream.schedule.transitions(
                policy_version=self.tape.policy_version,
                max_tokens=self.config.backward_batch_tokens,
            )
            for action, blocks, layer in transitions:
                yield BackwardStage(
                    action,
                    blocks,
                    layer,
                    lambda a=action, b=blocks, i=layer: stream.consume_group(a, b, i),
                )
        elif kind == "finish":
            if self.warmup:

                def finish_warmup():
                    from prime_rl.trainer.ps_backward.recorded_tail import (
                        warm_layerwise,
                    )

                    warm_layerwise(stream, policy_version=self.tape.policy_version)
                    stream.finish(policy_version=self.tape.policy_version)

                yield BackwardStage("warmup_finish", [], -1, finish_warmup)
                return
            from prime_rl.trainer.ps_backward.recorded_tail import iter_finish_layerwise

            stream._finishing = True
            tail = iter_finish_layerwise(stream)
            for layer in reversed(range(self.model.depth)):
                yield BackwardStage("tail", [], layer, lambda: next(tail))

            def finalize():
                try:
                    next(tail)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError("Layerwise tail yielded an unexpected extra stage")
                for parameter in stream.parameters.values():
                    if parameter.grad is not None:
                        parameter.grad.div_(stream.loss_tokens)
                if stream.kv_cache is not None:
                    stream.kv_cache.clear()

            yield BackwardStage("finalize", [], -1, finalize)
        else:
            raise ValueError(kind)

    def _stage_ready(self, request, end, stage):
        if stage.action == "head":
            return self.host_tape.ready(request, self.streams[request].schedule.end, stage.end or end)
        if stage.action not in ("query", "kv"):
            return True
        stream = self.streams[request]
        for block in stage.blocks:
            if not self.host_tape.ready(request, block.start, block.end, stage.layer):
                return False
        if stage.action == "query":
            begin, end = stage.blocks[0].start, stage.blocks[-1].end
            history = max(stream.schedule.prefix, begin - self.model.window + 1)
            for block in stream.schedule.blocks:
                start = block.start if block.start == 0 else max(block.start, history)
                stop = min(block.end, end)
                if start >= stop or (
                    stream.kv_cache is not None and stream.kv_cache.contains(stage.layer, start, stop)
                ):
                    continue
                if not self.host_tape.ready(request, start, stop, stage.layer):
                    return False
        return True

    def _work_stages(self):
        pending, active, credits = deque(), {}, deque()
        outstanding = 0
        served, turn, tail_owner = {}, 0, None
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.cuda.stream(self.cuda_stream),
        ):
            while True:
                try:
                    if not pending and not active:
                        item = self.tasks.get()
                        if item is None:
                            self.tasks.task_done()
                            return
                        pending.append(item)
                        outstanding += 1
                    while True:
                        try:
                            item = self.tasks.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            self.tasks.task_done()
                            return
                        pending.append(item)
                        outstanding += 1
                    if self.error is not None:
                        while outstanding:
                            self.tasks.task_done()
                            outstanding -= 1
                        pending.clear()
                        active.clear()
                        continue
                    waiting = deque()
                    while pending:
                        kind, request, end, event = pending.popleft()
                        if request in active:
                            waiting.append((kind, request, end, event))
                            continue
                        ends = [end]
                        if kind == "block":
                            rest, stopped = deque(), False
                            for item in pending:
                                if item[1] == request and item[0] != "block":
                                    stopped = True
                                if (
                                    item[1] == request
                                    and item[0] == "block"
                                    and not stopped
                                    and len(ends) * self.chunk < self.config.backward_batch_tokens
                                ):
                                    ends.append(item[2])
                                else:
                                    rest.append(item)
                            pending = rest
                        iterator = self._stages(kind, request, tuple(ends) if kind == "block" else end)
                        active[request] = dict(
                            iterator=iterator, stage=next(iterator), end=ends[-1], wait=None, tasks=len(ends)
                        )
                    pending = waiting
                    while credits and credits[0].query():
                        credits.popleft()
                    if len(credits) >= self.config.gpu_stages_in_flight:
                        start = time.perf_counter()
                        credits.popleft().synchronize()
                        self.gpu_credit_wait_seconds += time.perf_counter() - start
                    ready = []
                    for request, job in active.items():
                        if job["wait"] is not None and not job["wait"].query():
                            continue
                        stage = job["stage"]
                        if stage.action == "tail" and tail_owner not in (None, request):
                            continue
                        if self._stage_ready(request, job["end"], stage):
                            priority = 0 if stage.action == "head" else 2 if stage.action == "tail" else 1
                            age = self.observed.get(request, job["end"]) - job["end"]
                            ready.append((priority, -age, served.get(request, -1), request))
                    if not ready:
                        time.sleep(0.0005)
                        continue
                    _, _, _, request = min(ready)
                    job, stream = active[request], self.streams[request]
                    stage = job["stage"]
                    if stage.action == "tail":
                        tail_owner = request
                    with stream._active(self.tape.policy_version):
                        result = stage.run()
                    completed = torch.cuda.Event()
                    completed.record(self.cuda_stream)
                    credits.append(completed)
                    self.peak_gpu_stages = max(self.peak_gpu_stages, len(credits))
                    self.stage_counts[stage.action] = self.stage_counts.get(stage.action, 0) + 1
                    served[request], turn = turn, turn + 1
                    if stage.action == "finalize" and tail_owner == request:
                        tail_owner = None
                    job["wait"] = result if hasattr(result, "query") else None
                    try:
                        job["stage"] = next(job["iterator"])
                    except StopIteration:
                        del active[request]
                        if tail_owner == request:
                            tail_owner = None
                        for _ in range(job["tasks"]):
                            self.tasks.task_done()
                            outstanding -= 1
                except BaseException:
                    self.error = traceback.format_exc()
                    while outstanding:
                        self.tasks.task_done()
                        outstanding -= 1
                    pending.clear()
                    active.clear()

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
            self.cuda_stream.wait_stream(self.host_tape.stream)
            self.cuda_stream.wait_stream(self.host_tape.head_stream)
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
            gpu_credit_wait_seconds=self.gpu_credit_wait_seconds,
            peak_gpu_stages=self.peak_gpu_stages,
            stage_counts=self.stage_counts,
            optimizations=dict(
                joint_qkv=self.config.joint_qkv,
                direct_weight_gradients=self.config.direct_weight_gradients,
                weight_gradient_backend=self.config.weight_gradient_backend,
                compact_kv=self.model.recorded_compact_kv,
                kv_cache=self.config.kv_cache,
                stage_scheduling=self.config.stage_scheduling,
            ),
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
                    kv_cache=stream.kv_cache.report() if stream.kv_cache is not None else None,
                )
                for request, stream in self.streams.items()
            },
        )
