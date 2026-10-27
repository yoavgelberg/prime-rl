import torch

from prime_rl.trainer.ps_backward.recorded_graphs import _Stage


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
