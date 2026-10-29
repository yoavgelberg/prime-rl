"""Measure complete-episode PS backprop with reference and Adam state resident."""
import argparse
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .check_policy import teacher_logps
from .policy import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offload', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with gzip.open(args.record, 'rt') as handle:
        record = json.load(handle)
    assert record['attention'] == 'ps' and not record.get('censored')
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
    context = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload else nullcontext()
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
    result = dict(status='GO' if diffs.mean() <= .01 and np.quantile(diffs, .99) <= .05 else 'HOLD',
                  source=str(args.record), model_revision=Path(args.model).name, raw_tokens=len(record['tokens']),
                  generated_tokens=len(logps), prefix=record['prefix_length'], window=record['window'],
                  early_loss_tokens=sum(record['loss_mask'][:record['prefix_length'] + record['window']]),
                  reference_seconds=reference_seconds, policy_forward_seconds=forward_seconds,
                  backward_seconds=backward_seconds, mean_abs=float(diffs.mean()), p99=float(np.quantile(diffs, .99)),
                  gradient_norm=float(norm), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  activation_offload=args.offload, optimizer_states_resident=True, optimizer_steps=0,
                  likelihood_temperature=1., scope='Memory, timing and numerical audit only; weights never updated')
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
