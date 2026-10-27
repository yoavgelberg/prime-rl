"""Exact delayed GRPO centering with two gradient sums per unfinished group."""

import math

import torch


class RewardGradientGroup:
    """Reduce completed rollout gradients before the group baseline is known.

    Each input gradient already uses the desired token/trajectory normalization.
    The default baseline matches Prime-RL's reward mean; pass completion length
    as baseline_weight for its length-weighted variant. Pending rollouts still
    need their own score-gradient state until their individual reward arrives.
    """

    def __init__(self, size, policy_version):
        if size < 1:
            raise ValueError("A reward group must be nonempty")
        self.size, self.policy_version = size, policy_version
        self.seen = set()
        self.gradient_sum, self.reward_gradient_sum = {}, {}
        self.reward_sum, self.weight_sum = 0.0, 0.0
        self.closed = False

    @torch.no_grad()
    def add(self, identity, gradient, reward, *, policy_version, baseline_weight=1.0):
        if self.closed or len(self.seen) == self.size:
            raise RuntimeError("Reward group is already complete")
        if policy_version != self.policy_version:
            raise ValueError("Cannot combine gradients from different policy versions")
        if identity in self.seen:
            raise ValueError("Duplicate rollout gradient")
        if not math.isfinite(reward) or not math.isfinite(baseline_weight) or baseline_weight <= 0:
            raise ValueError("Reward must be finite and baseline weight positive")
        if self.seen and gradient.keys() != self.gradient_sum.keys():
            raise ValueError("Rollout gradients have different parameter sets")
        for name, value in gradient.items():
            if value.requires_grad:
                raise ValueError("Score-gradient accumulators must be detached")
            dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            value = value.to(dtype)
            if not self.seen:
                self.gradient_sum[name] = value.clone()
                self.reward_gradient_sum[name] = value.clone().mul_(reward)
            else:
                if value.shape != self.gradient_sum[name].shape:
                    raise ValueError("Rollout gradient shape changed")
                self.gradient_sum[name].add_(value)
                self.reward_gradient_sum[name].add_(value, alpha=reward)
        self.reward_sum += reward * baseline_weight
        self.weight_sum += baseline_weight
        self.seen.add(identity)

    @torch.no_grad()
    def finish(self, *, policy_version, scale=1.0):
        if self.closed or len(self.seen) != self.size:
            raise RuntimeError("All rewards must arrive exactly once before finalization")
        if policy_version != self.policy_version:
            raise ValueError("The optimizer policy changed while this group was in flight")
        baseline = self.reward_sum / self.weight_sum
        for name, value in self.reward_gradient_sum.items():
            value.add_(self.gradient_sum[name], alpha=-baseline).mul_(scale)
        self.closed = True
        self.gradient_sum.clear()
        return self.reward_gradient_sum
