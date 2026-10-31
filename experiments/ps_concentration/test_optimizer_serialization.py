import pytest
import torch

from prime_rl.trainer.ps_backward import optimizer_serialization as transport


def test_optimizer_object_roundtrip_uses_buffer_view_and_preserves_nested_state(monkeypatch):
    rank, messages, buffers = [0], [], []
    original = torch.frombuffer

    def frombuffer(buffer, **kwargs):
        buffers.append(type(buffer))
        return original(buffer, **kwargs)

    def broadcast(value, *, src, group, async_op):
        assert src == 0 and not async_op
        if rank[0] == 0:
            messages.append(value.clone())
        else:
            value.copy_(messages.pop(0))

    monkeypatch.setattr(transport.dist, 'get_rank', lambda: rank[0])
    monkeypatch.setattr(transport.dist, 'broadcast', broadcast)
    monkeypatch.setattr(torch, 'frombuffer', frombuffer)
    state = dict(state={7: dict(step=torch.tensor(3.), exp_avg=torch.randn(64, 32),
        exp_avg_sq=torch.rand(64, 32))}, param_groups=[dict(params=[7], lr=5e-6, betas=(.9,.95))])
    assert transport.broadcast_optimizer_object(state,0) is state
    rank[0] = 1
    restored = transport.broadcast_optimizer_object(None,0)
    assert buffers == [memoryview] and not messages
    assert restored['param_groups'] == state['param_groups']
    for name, tensor in state['state'][7].items():
        assert torch.equal(tensor, restored['state'][7][name])


def test_optimizer_transport_override_is_scoped_even_on_error():
    import torch.distributed.optim.zero_redundancy_optimizer as zero

    original = zero._broadcast_object
    with pytest.raises(RuntimeError, match='sentinel'):
        with transport.fast_optimizer_serialization():
            assert zero._broadcast_object is transport.broadcast_optimizer_object
            raise RuntimeError('sentinel')
    assert zero._broadcast_object is original
