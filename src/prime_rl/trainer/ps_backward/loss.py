from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossConfig:
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    batch_scale: float = 1.0
    head_block_tokens: int = 256


def trace_loss(model, trace, hidden, query_indices, config, *, target_start=0, target_end=None, denominator=None):
    if target_end is None:
        target_end = trace.completion_length
    response_indices = query_indices + 1 - trace.prefix_length
    selected = (response_indices >= target_start) & (response_indices < target_end)
    if not selected.any():
        return None
    response_indices = response_indices[selected]
    targets = trace.input_ids[query_indices[selected] + 1]
    temperature = trace.temperatures[response_indices] if trace.temperatures is not None else None
    logps = model.logps(hidden[selected], targets, config.head_block_tokens, temperature)
    advantage = (
        torch.ones_like(logps) if trace.advantages is None else trace.advantages[response_indices].to(logps.dtype)
    )
    if trace.old_logps is None:
        losses = -advantage * logps
    else:
        ratio = (logps - trace.old_logps[response_indices].to(logps.dtype)).exp()
        clipped = ratio.clamp(1 - config.epsilon_low, 1 + config.epsilon_high)
        losses = -torch.minimum(ratio * advantage, clipped * advantage)
    if trace.loss_mask is not None:
        losses = losses * trace.loss_mask[response_indices]
    if denominator is None:
        denominator = trace.completion_length if trace.loss_mask is None else trace.loss_mask.sum().clamp_min(1)
    return losses.sum() * config.batch_scale / denominator
