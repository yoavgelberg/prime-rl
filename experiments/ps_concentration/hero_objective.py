"""Masked GRPO, frozen-reference KL, and entropy for the hero run."""

import math


def shaped_return(row, *, format_penalty=.02, invalid_penalty=.02):
    native = row['training_return']
    return native - format_penalty * row['format_errors'] - invalid_penalty * row['invalid_actions']


def advantages(returns, epsilon=1e-4):
    mean = math.fsum(returns) / len(returns)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in returns) / (len(returns) - 1))
    return ([0.] * len(returns) if std == 0 else [(value - mean) / (std + epsilon) for value in returns]), std


def token_objective(logits, labels, reference, old, advantage, *, temperature=.6,
                    clip=.2, kl_coefficient=.01, entropy_coefficient=.001):
    import torch

    values = logits.float() / temperature
    log_normalizer = values.logsumexp(-1)
    logps = values.gather(-1, labels[:, None]).squeeze(-1) - log_normalizer
    probabilities = values.softmax(-1)
    entropy = log_normalizer - (probabilities * values).sum(-1)
    log_ratio = (logps - old).clamp(-20, 20)
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if reference.ndim == 2:
        if reference.shape != values.shape:
            raise ValueError('Categorical reference log probabilities must cover the same token vocabulary')
        kl = (probabilities * (values - log_normalizer[:, None] - reference.detach())).sum(-1)
    else:
        reference_ratio = (reference - logps).clamp(-20, 20)
        kl = reference_ratio.expm1() - reference_ratio
    loss = policy + kl_coefficient * kl - entropy_coefficient * entropy
    statistics = torch.stack((policy.detach().sum(), kl.detach().sum(), entropy.detach().sum(),
        logps.detach().sum(), (ratio.detach() - 1).abs().gt(clip).float().sum(),
        (logps.detach() - old).abs().sum(), (logps.detach() - old).abs().max()))
    return loss.sum(), statistics


def trajectory_backward(policy, reference, record, advantage, config, *, adapters=None, kl_estimator='k3'):
    import torch
    from torch.utils.checkpoint import checkpoint
    from .native_teacher import NativeTeacher

    tokens = record['tokens']
    if kl_estimator not in ('k3', 'categorical'):
        raise ValueError('Use sampled k3 or exact categorical reference KL')
    masks = record['loss_mask']
    assert len(tokens) == len(masks) == len(record['old_tempered_logps'])
    assert sum(masks) == record['generated_tokens'] > 0
    assert record['prefix_length'] <= 512 and record['window'] == config.window
    assert all((value is not None) == keep for value, keep in zip(record['old_tempered_logps'], masks, strict=True))
    device = policy.device
    ids = torch.tensor(tokens, device=device, dtype=torch.long)
    padding = (-(len(ids) - 1)) % 1024
    if padding:
        ids = torch.nn.functional.pad(ids, (0, padding), value=int(ids[-1]))
    indices = torch.arange(len(ids) - 1, device=device)
    selected = torch.tensor(masks[1:] + [False] * padding, device=device)
    labels = ids[1:][selected]
    old = torch.tensor([value for value, keep in zip(record['old_tempered_logps'], masks, strict=True) if keep], device=device)
    actor, teacher = adapters if adapters is not None else (
        NativeTeacher(policy, config.window, backend='fa4', checkpoint_layers=True),
        NativeTeacher(reference, config.window, backend='fa4'))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ref_hidden = teacher.segment(ids[:-1], indices, indices, record['prefix_length'])[selected]
        temperatures = torch.full((len(labels),), config.temperature, device=device)
        if kl_estimator == 'k3':
            ref = teacher.logps(ref_hidden, labels, block_size=config.head_block_tokens, temperatures=temperatures)
            del ref_hidden
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = actor.segment(ids[:-1], indices, indices, record['prefix_length'])[selected]
        losses, statistics = [], []

        def head(h, weight, targets, reference_logps, rollout_logps):
            return token_objective(actor.linear(h, weight), targets, reference_logps, rollout_logps, advantage,
                temperature=config.temperature, clip=config.clip_ratio,
                kl_coefficient=config.kl_coefficient, entropy_coefficient=config.entropy_coefficient)

        for start in range(0, len(labels), config.head_block_tokens):
            stop = start + config.head_block_tokens
            if kl_estimator == 'categorical':
                with torch.no_grad():
                    reference_logits = teacher.linear(ref_hidden[start:stop], reference.lm_head.weight)
                    reference_block = (reference_logits.float() / config.temperature).log_softmax(-1)
                    del reference_logits
            else:
                reference_block = ref[start:stop]
            loss, stats = checkpoint(head, hidden[start:stop], policy.lm_head.weight,
                labels[start:stop], reference_block, old[start:stop], use_reentrant=False)
            losses.append(loss)
            statistics.append(stats)
        objective = torch.stack(losses).sum() / (config.denominator * config.batch_size)
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError('Nonfinite GRPO/KL/entropy objective')
    objective.backward()
    rows = torch.stack(statistics).detach()
    totals = rows[:, :6].sum(0).cpu().tolist()
    maximum = float(rows[:, 6].max())
    count = len(labels)
    return dict(sampled_tokens=count, normalized_loss=float(objective.detach()),
        policy_sum=totals[0], kl_sum=totals[1], entropy_sum=totals[2], logp_sum=totals[3],
        clipped_tokens=int(totals[4]), rollout_teacher_abs_logp_sum=totals[5],
        rollout_teacher_max_abs_logp=maximum, all_sampled_tokens_have_loss=True,
        forced_tokens_have_direct_loss=False)
