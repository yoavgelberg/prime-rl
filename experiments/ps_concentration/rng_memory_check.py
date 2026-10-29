"""Measure complete-episode PS backprop with reference and Adam state resident."""
import argparse
from contextlib import nullcontext
import gzip
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .check_policy import teacher_logps
from .activation_offload import activation_offload
from .policy import load_model


def check_offload_gradients():
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(19)
    weight = torch.nn.Parameter(torch.randn(32, 32, device='cuda') / 8)
    inputs = torch.randn(256, 32, device='cuda')
    results = []
    for offload in (False, True):
        weight.grad = None
        x = inputs.clone().requires_grad_()
        with activation_offload([weight], min_bytes=1) if offload else nullcontext():
            hidden = x
            for _ in range(4):
                hidden = checkpoint(lambda h, w: (h @ w).tanh(), hidden, weight, use_reentrant=False)
            loss = hidden.square().mean()
        loss.backward()
        results.append((loss.detach(), x.grad.clone(), weight.grad.clone()))
    errors = [float((a - b).norm() / b.norm().clamp_min(1e-12))
              for a, b in zip(results[1], results[0], strict=True)]
    assert max(errors) < 1e-6, errors
    return dict(loss_input_weight_gradient_relative_l2=errors, limit=1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--zero-step-smoke', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    offload_audit = check_offload_gradients() if args.offload else None
    print(json.dumps(dict(event='offload_gradient_audit', result=offload_audit)), flush=True)
    with gzip.open(args.record, 'rt') as handle:
        record = json.load(handle)
    assert record['attention'] == 'ps'
    policy = load_model(args.model, train=True)
    reference = load_model(args.model)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0, foreach=True)
    for param in policy.parameters():
        optimizer.state[param].update(step=torch.tensor(0.), exp_avg=torch.zeros_like(param),
                                      exp_avg_sq=torch.zeros_like(param))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        ref = teacher_logps(reference, record)
    reference_seconds = time.perf_counter() - started
    print(json.dumps(dict(event='reference_complete', tokens=len(ref), seconds=reference_seconds,
                          allocated_bytes=torch.cuda.memory_allocated())), flush=True)
    started = time.perf_counter()
    context = activation_offload(policy.parameters()) if args.offload else nullcontext()
    with context:
        logps = teacher_logps(policy, record, grad=True)
        ratio = ref - logps
        loss = (-logps + .01 * (ratio.exp() - ratio - 1)).sum() / (26624 * 64)
    forward_seconds = time.perf_counter() - started
    print(json.dumps(dict(event='forward_complete', seconds=forward_seconds,
                          allocated_bytes=torch.cuda.memory_allocated())), flush=True)
    started = time.perf_counter()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
    backward_seconds = time.perf_counter() - started
    assert float(norm) > 0
    old = torch.tensor([p for p, keep in zip(record['old_logps'], record['loss_mask'], strict=True) if keep], device=policy.device)
    diffs = (ref - old).abs().cpu().numpy()
    if args.zero_step_smoke:
        for parameter in policy.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        optimizer.step()
        assert all(torch.equal(p, q) for p, q in zip(policy.parameters(), reference.parameters(), strict=True)), 'Zero-gradient audit changed model weights'
    result = dict(status='GO' if diffs.mean() <= .01 and np.quantile(diffs, .99) <= .05 else 'HOLD',
                  source=str(args.record), model_revision=Path(args.model).name, raw_tokens=len(record['tokens']),
                  generated_tokens=len(logps), prefix=record['prefix_length'], window=record['window'],
                  early_loss_tokens=sum(record['loss_mask'][:record['prefix_length'] + record['window']]),
                  reference_seconds=reference_seconds, policy_forward_seconds=forward_seconds,
                  backward_seconds=backward_seconds, mean_abs=float(diffs.mean()), p99=float(np.quantile(diffs, .99)),
                  gradient_norm=float(norm), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  offload_equivalence=offload_audit,
                  activation_offload=args.offload, optimizer_states_resident=True,
                  source_context_censored=bool(record.get('censored')),
                  optimizer_steps=int(args.zero_step_smoke), learned_parameter_updates=0,
                  zero_gradient_optimizer_weights_unchanged=args.zero_step_smoke,
                  source_sha256=os.environ.get('PS_CONCENTRATION_SOURCE_SHA', ''),
                  likelihood_temperature=1., scope='Memory, timing and numerical audit only; weights never updated')
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
