from types import SimpleNamespace

import torch

from prime_rl.trainer.ps_backward.recorded_graphs import _Stage, RecordedGraphCache


def test_attention_pieces_pack_into_static_buffer_with_one_dtype_conversion():
    pieces = [torch.randn(1, 2, n, 8, dtype=torch.bfloat16) for n in (3, 5, 2)]
    stage = _Stage(lambda values: values["k"], dict(k=pieces), dtypes=dict(k=torch.float32))
    actual = stage.run(dict(k=pieces))
    torch.testing.assert_close(actual, torch.cat(pieces, 2).float(), atol=0, rtol=0)
    pieces = [value + 1 for value in pieces]
    pointer = actual.data_ptr()
    torch.testing.assert_close(stage.run(dict(k=pieces)), torch.cat(pieces, 2).float(), atol=0, rtol=0)
    assert stage.output is None
    assert stage.static["k"].data_ptr() == pointer


def test_trajectory_pointer_table_is_never_padded_to_token_chunk_size():
    values = dict(raw=torch.randn(2, 7), gradient_pointers=torch.tensor([123, 456, 789]))
    stage = _Stage(lambda xs: xs, values, rows=8)
    actual = stage.run(values)
    assert actual["raw"].shape == (8, 7)
    torch.testing.assert_close(actual["gradient_pointers"], values["gradient_pointers"])


def test_attention_retention_keeps_recent_shapes_and_preserves_dense_stages():
    cache = RecordedGraphCache(SimpleNamespace(), 16)
    values = dict(x=torch.tensor([2.]))
    run = lambda xs: xs['x'] * 3
    for key in [('post',0), ('attention',1), ('attention',2), ('attention',3)]:
        torch.testing.assert_close(cache._call(key, run, values), torch.tensor([6.]))
    dense = cache.stages[('post',0)]
    cache._call(('attention',1), run, values)
    result = cache.trim_attention(2)
    assert result['attention_evicted'] == 1 and result['attention_retained'] == 2
    assert ('attention',2) not in cache.stages
    assert cache.stages[('post',0)] is dense
    assert list(cache.stages)[-2:] == [('attention',3), ('attention',1)]


def test_attention_cache_stays_bounded_across_cohorts_without_changing_derivatives():
    cache = RecordedGraphCache(SimpleNamespace(), 16)
    def derivative(xs):
        x = xs['x'].detach().requires_grad_()
        return torch.autograd.grad((x*x).sum(), x)[0]
    for cohort in range(32):
        for position in range(4):
            value = torch.tensor([float(cohort + position)])
            actual = cache._call(('attention',cohort,position), derivative, dict(x=value))
            torch.testing.assert_close(actual, 2*value, atol=0, rtol=0)
        result = cache.trim_attention(8)
        assert result['attention_retained'] <= 8 and len(cache.stages) <= 8
