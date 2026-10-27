import pytest

from prime_rl.trainer.ps_backward.retirement import RetirementSchedule


@pytest.mark.parametrize("depth,window,chunk", [(1, 1, 1), (4, 1, 3), (4, 7, 3), (3, 4, 7), (5, 13, 8)])
def test_retirement_waits_for_every_causal_attention_edge_and_upper_derivative(depth, window, chunk):
    prefix, count = 5, 199
    schedule = RetirementSchedule(depth, window, prefix, chunk, policy_version=7)
    queries, keys, releases, embeddings = set(), set(), set(), set()
    maximum = 0

    def consume(action, block, layer):
        nonlocal maximum
        tokens = set(range(block.start, block.end))
        if action == "query":
            if layer + 1 < depth:
                assert all((layer + 1, t) in keys for t in tokens)
            assert all((layer, t) not in queries for t in tokens)
            queries.update((layer, t) for t in tokens)
        elif action == "kv":
            for key in tokens:
                required = (q for q in range(key, count) if key < prefix or key >= q - window + 1)
                assert all((layer, q) in queries for q in required)
                assert (layer, key) not in keys
                keys.add((layer, key))
        elif action == "release_layer":
            assert all((layer, t) in keys and (layer, t) in queries for t in tokens)
            releases.update((layer, t) for t in tokens)
        elif action == "embedding":
            assert all((0, t) in keys for t in tokens)
            assert not (tokens & embeddings)
            embeddings.update(tokens)
        else:
            raise AssertionError(action)
        maximum = max(maximum, schedule.peak_live_token_layers)

    schedule.append(prefix, policy_version=7)
    schedule.advance(consume, policy_version=7)
    for start in range(prefix, count, chunk):
        schedule.append(min(start + chunk, count), policy_version=7)
        schedule.advance(consume, policy_version=7)
    schedule.advance(consume, policy_version=7, finished=True)
    assert len(queries) == len(keys) == len(releases) == count * depth
    assert embeddings == set(range(count))
    assert maximum <= schedule.activation_bound
    assert schedule.closed and not schedule.blocks


def test_retirement_memory_stays_bounded_after_many_windows_and_stale_versions_fail():
    def simulate(count):
        schedule = RetirementSchedule(8, 257, 19, 64, 3)
        schedule.append(19, policy_version=3)
        for end in range(83, count, 64):
            schedule.append(end, policy_version=3)
            schedule.advance(lambda *args: None, policy_version=3)
        return schedule

    short, long = simulate(10000), simulate(100000)
    assert long.peak_live_token_layers == short.peak_live_token_layers
    assert long.peak_live_token_layers <= long.activation_bound
    assert long.retired_tokens > 90000
    with pytest.raises(ValueError, match="stale-policy"):
        long.advance(lambda *args: None, policy_version=4)
    long.advance(lambda *args: None, policy_version=3, finished=True)
    with pytest.raises(ValueError, match="Closed"):
        long.append(long.end + 1, policy_version=3)


def test_failed_consumer_does_not_release_its_tape():
    schedule = RetirementSchedule(2, 3, 5, 4, 0)
    block = schedule.append(5, policy_version=0)

    def fail(*args):
        raise RuntimeError("backpressure")

    with pytest.raises(RuntimeError, match="backpressure"):
        schedule.advance(fail, policy_version=0)
    assert not any(block.query_done) and not any(block.kv_done)
    assert schedule.blocks == [block]
