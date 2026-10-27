"""Full-graph FA4-PS and the paper's approximate final-window baseline."""

import torch

from .loss import LossConfig, trace_loss
from .types import BackwardResult


def backward_full(model, trace, loss_config=LossConfig()):
    indices = torch.arange(trace.input_ids.numel() - 1, device=model.device)
    hidden = model.segment(trace.input_ids[:-1], trace.positions[:-1], indices, trace.prefix_length)
    loss = trace_loss(model, trace, hidden, indices, loss_config)
    loss.backward()
    count = trace.completion_length if trace.loss_mask is None else int(trace.loss_mask.sum())
    return BackwardResult(loss.detach().item(), count)


def backward_truncated(model, trace, loss_config=LossConfig()):
    """Train the final W labels with 3W history plus the full prompt; not exact."""
    start = max(0, trace.completion_length - model.window)
    query_start = trace.prefix_length - 1 + start
    query_end = trace.input_ids.numel() - 1
    context_start = max(trace.prefix_length, query_start - 3 * model.window)
    indices = torch.cat(
        (
            torch.arange(trace.prefix_length, device=model.device),
            torch.arange(context_start, query_end, device=model.device),
        )
    )
    hidden = model.segment(
        trace.input_ids[indices],
        trace.positions[indices],
        indices,
        trace.prefix_length,
        coordinate_gap=context_start - trace.prefix_length,
    )
    count = trace.completion_length - start if trace.loss_mask is None else int(trace.loss_mask[start:].sum())
    loss = trace_loss(model, trace, hidden, indices, loss_config, target_start=start, denominator=max(count, 1))
    loss.backward()
    return BackwardResult(loss.detach().item(), count)
