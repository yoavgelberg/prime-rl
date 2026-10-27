"""Accumulate linear weight gradients into the selected trajectory's buffers."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_targets = ContextVar("ps_weight_gradient_targets", default=None)


@contextmanager
def direct_weight_gradients(pointers, indices, targets=None, backend="triton"):
    token = _targets.set((pointers, indices, targets, backend))
    try:
        yield
    finally:
        _targets.reset(token)


def weight_gradient_target(parameters):
    target = _targets.get()
    if target is None:
        return None
    pointers, indices, buffers, backend = target
    return pointers, tuple(indices[id(p)] for p in parameters), buffers, parameters, backend


def accumulate_weight_gradient(dy, x, target):
    pointers, indices, buffers, parameters, backend = target
    if not x.is_cuda:
        offset = 0
        for parameter in parameters:
            width = parameter.shape[0]
            buffers[id(parameter)].addmm_(dy[:, offset : offset + width].T, x)
            offset += width
        return
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Direct CUDA weight gradients require FP32 master parameters")
    if backend == "cublas":
        from .gradient_accumulation_cublas import accumulate
    else:
        from .gradient_accumulation_kernel import accumulate

    accumulate(dy, x, pointers, indices, tuple(p.shape[0] for p in parameters))
