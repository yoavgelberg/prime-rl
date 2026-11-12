"""Buffer-based transport for large sharded optimizer checkpoint objects."""

from contextlib import contextmanager
import io

import torch
import torch.distributed as dist


def broadcast_optimizer_object(obj, src_rank, group=dist.group.WORLD, device=torch.device('cpu')):
    if dist.get_rank() == src_rank:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        payload = torch.frombuffer(buffer.getbuffer(), dtype=torch.uint8).to(device)
        length = torch.tensor([payload.numel()], dtype=torch.int64, device=device)
        dist.broadcast(length, src=src_rank, group=group, async_op=False)
        dist.broadcast(payload, src=src_rank, group=group, async_op=False)
    else:
        length = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(length, src=src_rank, group=group, async_op=False)
        payload = torch.empty(int(length.item()), dtype=torch.uint8, device=device)
        dist.broadcast(payload, src=src_rank, group=group, async_op=False)
        obj = torch.load(io.BytesIO(payload.cpu().numpy()), map_location=device, weights_only=False)
    return obj


@contextmanager
def fast_optimizer_serialization():
    import torch.distributed.optim.zero_redundancy_optimizer as zero

    original = zero._broadcast_object
    zero._broadcast_object = broadcast_optimizer_object
    try:
        yield
    finally:
        zero._broadcast_object = original
