"""Group-standardized GRPO through the orchestrator custom-advantage interface."""

import math

import torch


def normalized_advantages(rewards, *, epsilon=1e-4):
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("GRPO requires a matrix with at least two rollouts per board")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("GRPO epsilon must be finite and positive")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, correction=1, keepdim=True)
    return torch.where(std > 0, centered / (std + epsilon), torch.zeros_like(centered))


def grpo_advantage_fn(inputs, *, epsilon=1e-4):
    from .advantage import AdvantageOutputs

    return AdvantageOutputs(advantages=normalized_advantages(inputs.rewards, epsilon=epsilon))
