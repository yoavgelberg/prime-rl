import queue
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.ps_live import BackwardStage, LiveBackward


def test_scheduler_skips_unready_requests_and_bounds_submitted_gpu_stages(monkeypatch):
    events, calls = [], []

    class Event:
        def __init__(self):
            self.complete = False
            events.append(self)

        def record(self, stream):
            pass

        def query(self):
            return self.complete

        def synchronize(self):
            self.complete = True

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    live = LiveBackward.__new__(LiveBackward)
    live.tasks = queue.Queue()
    live.config = SimpleNamespace(gpu_stages_in_flight=2, backward_batch_tokens=8)
    live.chunk = 8
    live.cuda_stream = object()
    live.tape = SimpleNamespace(policy_version=0)
    live.streams = {name: SimpleNamespace(_active=lambda _: nullcontext()) for name in ("slow", "fast")}
    live.observed = dict(slow=8, fast=8)
    live.gpu_credit_wait_seconds, live.peak_gpu_stages = 0.0, 0
    live.stage_counts, live.error = {}, None

    def stages(kind, request, end):
        for action in ("head", "query", "tail"):
            yield BackwardStage(action, [], 0, lambda r=request, a=action: calls.append((r, a)))

    live._stages = stages
    live._stage_ready = lambda request, end, stage: request == "fast" or bool(calls)
    for request in ("slow", "fast"):
        live.tasks.put(("block", request, 8, None))
    worker = threading.Thread(target=live._work_stages, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while live.tasks.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.001)
    live.tasks.put(None)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert live.tasks.unfinished_tasks == 0
    assert live.error is None
    assert calls[0] == ("fast", "head")
    assert calls[1] == ("slow", "head")
    assert live.peak_gpu_stages == 2
    assert sum(event.complete for event in events) >= 4
    assert live.stage_counts == dict(head=2, query=2, tail=2)


@pytest.mark.parametrize("joint,direct,compact,cache", [
    (False, False, False, False), (True, False, True, True), (True, True, True, True),
])
def test_grouped_retirement_and_interleaved_trajectory_gradients_match_full(joint, direct, compact, cache):
    from helpers import assert_grads, dense_tape, gradients, make_trace, tiny_model

    from prime_rl.trainer.ps_backward.baseline import backward_full
    from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream
    from prime_rl.trainer.ps_backward.types import Trace

    model = tiny_model(depth=3, window=4)
    model.recorded_joint_qkv = joint
    model.recorded_direct_weight_gradients = direct
    model.recorded_compact_kv = compact
    streams, expected, traces = {}, {}, {}
    for index in range(2):
        original = make_trace(prefix=4, length=37)
        trace = Trace(original.input_ids.roll(index), 4, original.positions)
        traces[index] = trace
        backward_full(model, trace)
        expected[index] = gradients(model)
        model.zero_grad(set_to_none=True)
        tape, ropes = dense_tape(model, trace)
        streams[index] = RecordedStream(
            model, tape, "request", 4, ropes, chunk=3, policy_version=0, tail_block_tokens=9, kv_working_cache=cache
        )
    ends = [4, *range(7, 40, 3), 40]
    for offset in range(0, len(ends), 2):
        for index, stream in streams.items():
            for end in ends[offset : offset + 2]:
                start = stream.schedule.end
                labels = traces[index].input_ids[start + 1 : end + 1].clone()
                if not start:
                    labels[:3] = -100
                with stream._active(0):
                    stream.append(end, traces[index].input_ids[start:end], labels, policy_version=0, advance=False)
        iterators = {i: s.schedule.transitions(policy_version=0, max_tokens=9) for i, s in streams.items()}
        while iterators:
            for index in list(iterators):
                try:
                    action, blocks, layer = next(iterators[index])
                except StopIteration:
                    del iterators[index]
                    continue
                with streams[index]._active(0):
                    streams[index].consume_group(action, blocks, layer)
    live = LiveBackward.__new__(LiveBackward)
    live.streams, live.model = streams, model
    live.tape, live.warmup = SimpleNamespace(policy_version=0), False
    tails = {index: live._stages("finish", index, stream.schedule.end) for index, stream in streams.items()}
    while tails:
        for index in list(tails):
            try:
                stage = next(tails[index])
            except StopIteration:
                del tails[index]
                continue
            with streams[index]._active(0):
                stage.run()
    for index, stream in streams.items():
        assert_grads(stream.gradients, expected[index], rtol=1e-8, atol=1e-10)
        assert stream.schedule.closed
        assert stream.kv_cache is None or stream.kv_cache.bytes == 0


@pytest.mark.parametrize("direct", [False, True])
def test_nested_gradient_context_preserves_new_gradients_and_restores_trajectory_pointers(direct):
    from helpers import dense_tape, make_trace, tiny_model

    from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream

    model = tiny_model(depth=1, window=1)
    model.recorded_direct_weight_gradients = direct
    trace = make_trace(prefix=4, length=7)
    tape, ropes = dense_tape(model, trace)
    streams = [RecordedStream(model, tape, "request", 4, ropes, chunk=4) for _ in range(2)]
    name, parameter = next(iter(streams[0].parameters.items()))
    with streams[0]._active(0):
        if direct:
            parameter.grad.fill_(1)
        else:
            parameter.grad = torch.ones_like(parameter)
        first = parameter.grad
        with streams[0]._active(0):
            assert parameter.grad is first
            parameter.grad.add_(2)
        assert parameter.grad is first
        with streams[1]._active(0):
            if direct:
                parameter.grad.fill_(7)
            else:
                parameter.grad = torch.full_like(parameter, 7)
        assert parameter.grad is first
        if direct:
            assert model._ps_gradient_pointers is streams[0].gradient_pointers
    assert parameter.grad is None
    torch.testing.assert_close(streams[0].gradients[name], torch.full_like(parameter, 3))
    torch.testing.assert_close(streams[1].gradients[name], torch.full_like(parameter, 7))
