"""Offload saved activations while keeping resident model parameters on device."""
from contextlib import contextmanager
import weakref

import torch


@contextmanager
def activation_offload(parameters, min_bytes=1024 * 1024):
    resident = {p.untyped_storage().data_ptr() for p in parameters}
    cpu = torch.autograd.graph.save_on_cpu(pin_memory=True)
    copies = {}

    def pack(tensor):
        if tensor.device.type != 'cuda' or tensor.numel() * tensor.element_size() < min_bytes:
            return False, tensor
        if tensor.untyped_storage().data_ptr() in resident:
            return False, tensor
        key = id(tensor)
        previous = copies.get(key)
        if previous is None or previous[0]() is not tensor:
            previous = weakref.ref(tensor), cpu.pack_hook(tensor)
            copies[key] = previous
        return True, previous[1]

    def unpack(saved):
        offloaded, value = saved
        return cpu.unpack_hook(value) if offloaded else value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
