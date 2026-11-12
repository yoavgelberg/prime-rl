"""Synchronous actor-only GRPO updates from complete recorded score gradients."""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist

from .delayed_reward import RewardGradientGroup


@dataclass(frozen=True)
class OutcomeConfig:
    group_size: int = 8
    global_batch_size: int = 64
    denominator: int = 26624
    epsilon: float = 1e-4
    learning_rate: float = 5e-6
    warmup_updates: int = 8
    max_gradient_norm: float = 1.

    def __post_init__(self):
        if self.group_size < 2 or self.global_batch_size % self.group_size:
            raise ValueError('The global batch must contain complete multi-rollout groups')
        if min(self.denominator, self.warmup_updates) < 1:
            raise ValueError('Denominator and warmup must be positive')
        scales = (self.epsilon, self.learning_rate, self.max_gradient_norm)
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError('GRPO and optimizer scales must be positive')


class OutcomeTrainer:
    """Input score derivatives must already be divided by the fixed denominator."""

    def __init__(self, model, optimizer, config=OutcomeConfig(), *, group=None, world_size=1, policy_version=0):
        if world_size < 1 or config.global_batch_size % world_size or (config.global_batch_size // world_size) % config.group_size:
            raise ValueError('Each GPU replica must own complete board groups')
        self.model, self.optimizer, self.config = model, optimizer, config
        self.group, self.world_size, self.policy_version = group, world_size, policy_version
        self.parameters = dict(model.named_parameters())
        self.groups, self.records = {}, {}
        self.materialized = False

    @torch.no_grad()
    def add(self, identity, board, gradient, reward, *, policy_version, generated_tokens, negative_logp_sum):
        if self.materialized or identity in self.records:
            raise ValueError('A rollout must be admitted exactly once before materialization')
        if policy_version != self.policy_version:
            raise ValueError('Recorded gradient policy version is stale')
        if gradient.keys() != self.parameters.keys():
            raise ValueError('Recorded gradient does not cover every trainable parameter')
        if generated_tokens < 1 or not math.isfinite(negative_logp_sum):
            raise ValueError('Every episode needs finite likelihoods for its sampled tokens')
        if board not in self.groups:
            self.groups[board] = RewardGradientGroup(self.config.group_size, self.policy_version,
                baseline='grpo', device='cpu', epsilon=self.config.epsilon)
        self.groups[board].add(identity, gradient, reward, policy_version=policy_version)
        self.records[identity] = dict(board=board, reward=reward, generated_tokens=generated_tokens,
                                      negative_logp_sum=negative_logp_sum)

    @torch.no_grad()
    def materialize(self):
        expected = self.config.global_batch_size // self.world_size
        if self.materialized or len(self.records) != expected:
            raise RuntimeError(f'Need exactly {expected} fresh episodes before materialization')
        if any(len(group.seen) != self.config.group_size for group in self.groups.values()):
            raise RuntimeError('Every board must have all its rewards before materialization')
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.groups.values():
            gradient = group.finish(policy_version=self.policy_version, scale=1 / self.config.global_batch_size)
            for name, parameter in self.parameters.items():
                if parameter.grad is None:
                    parameter.grad = gradient[name].to(device=parameter.device, dtype=parameter.dtype, copy=True)
                else:
                    parameter.grad.add_(gradient[name].to(device=parameter.device, dtype=parameter.dtype))
            group.reward_gradient_sum.clear()
        self.materialized = True
        advantages, score_loss, clipped_loss = [], 0., 0.
        for row in self.records.values():
            group = self.groups[row['board']]
            mean = math.fsum(group.rewards) / group.size
            advantage = 0. if group.reward_std == 0 else (row['reward'] - mean) / (group.reward_std + group.epsilon)
            advantages.append(advantage)
            scale = advantage / (self.config.denominator * self.config.global_batch_size)
            score_loss += scale * row['negative_logp_sum']
            clipped_loss -= scale * row['generated_tokens']
        return dict(local_episodes=len(self.records), local_groups=len(self.groups),
            zero_advantage_groups=sum(group.reward_std == 0 for group in self.groups.values()),
            group_reward_stds=[group.reward_std for group in self.groups.values()], advantages=advantages,
            normalized_score_surrogate_loss=score_loss,
            clipped_grpo_surrogate_at_origin=clipped_loss, clip_fraction_at_origin=0.)

    @torch.no_grad()
    def step(self):
        if not self.materialized:
            raise RuntimeError('Materialize task gradients and add reference KL before stepping')
        for parameter in self.parameters.values():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if self.world_size > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=self.group)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters.values(), self.config.max_gradient_norm,
                                             error_if_nonfinite=True)
        update = self.policy_version + 1
        lr = self.config.learning_rate * min(update / self.config.warmup_updates, 1.)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        local_optimizer = getattr(self.optimizer, 'optim', self.optimizer)
        owned = [parameter for group in local_optimizer.param_groups for parameter in group['params']]
        previous = [parameter.detach().cpu().clone() for parameter in owned]
        self.optimizer.step()
        delta = torch.zeros((), device=next(iter(self.parameters.values())).device, dtype=torch.float64)
        for parameter, old in zip(owned, previous, strict=True):
            current, old = parameter.reshape(-1), old.reshape(-1)
            for start in range(0, current.numel(), 1048576):
                before = old[start:start + 1048576].to(parameter.device, dtype=torch.float64)
                delta += (current[start:start + 1048576].double() - before).square().sum()
        if self.world_size > 1:
            dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=self.group)
        self.policy_version = update
        self.model._ps_update_version = update
        self.optimizer.zero_grad(set_to_none=True)
        self.groups.clear()
        self.records.clear()
        self.materialized = False
        return dict(update=update, learning_rate=lr, gradient_norm=float(norm),
            gradient_clipped=float(norm) > self.config.max_gradient_norm, weight_update_norm=float(delta.sqrt()),
            policy_version=self.policy_version)
