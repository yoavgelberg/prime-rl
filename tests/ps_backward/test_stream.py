import pytest
from helpers import assert_grads, dense_tape, gradients, make_trace, tiny_model

from prime_rl.trainer.ps_backward.baseline import backward_full
from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream
from prime_rl.trainer.ps_backward.types import Trace


@pytest.mark.parametrize("chunk", [1, 3, 8])
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("depth,window,length", [(1, 1, 3), (3, 4, 31), (4, 7, 41)])
def test_streaming_recorded_vjps_match_full_and_never_read_retired_layers(chunk, layerwise, depth, window, length):
    model = tiny_model(depth=depth, window=window)
    source = make_trace(length=length)
    trace = Trace(source.input_ids, source.prefix_length, source.positions)
    tape, ropes = dense_tape(model, trace)
    full_loss = backward_full(model, trace).loss
    expected = gradients(model)
    model.zero_grad(set_to_none=True)
    original_layer = tape.layer
    released = set()

    def layer(request, index, start, end):
        assert all((index, token) not in released for token in range(start, end))
        return original_layer(request, index, start, end)

    def release_layer(request, index, start, end):
        released.update((index, token) for token in range(start, end))

    tape.layer, tape.release_layer = layer, release_layer
    stream = RecordedStream(
        model, tape, "request", trace.prefix_length, ropes, chunk=chunk, policy_version=3, tail_block_tokens=2 * chunk
    )
    n = trace.input_ids.numel() - 1
    starts = [0, *range(trace.prefix_length, n, chunk)]
    ends = [trace.prefix_length, *(min(start + chunk, n) for start in starts[1:])]
    for start, end in zip(starts, ends, strict=True):
        labels = trace.input_ids[start + 1 : end + 1].clone()
        if start == 0:
            labels[: trace.prefix_length - 1] = -100
        stream.append(end, trace.input_ids[start:end], labels, policy_version=3)
    if layerwise:
        from prime_rl.trainer.ps_backward.recorded_tail import warm_layerwise

        before = set(released)
        warm_layerwise(stream, policy_version=3)
        assert released == before
    actual = stream.finish(policy_version=3, layerwise=layerwise)
    assert_grads(actual, expected, rtol=1e-8, atol=1e-10)
    assert (stream.loss / stream.loss_tokens).item() == pytest.approx(full_loss, abs=1e-12)
    assert len(released) == n * model.depth
    assert stream.schedule.peak_live_token_layers <= stream.schedule.activation_bound
    assert all(p.grad is None for p in model.parameters())
