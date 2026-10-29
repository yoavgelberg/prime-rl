"""Check saved native-thinking likelihoods and an all-token backward pass."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from .check_policy import parity, teacher_logps, toy_reference
from .policy import load_model


def prefix_record(path, tokens):
    with gzip.open(path, 'rt') as handle:
        record = json.load(handle)
    end = min(tokens, len(record['tokens']))
    for key in ('tokens', 'roles', 'loss_mask', 'old_logps', 'positions', 'old_tempered_logps'):
        if key in record:
            record[key] = record[key][:end]
    assert record['positions'] == list(range(end))
    assert record['loss_mask'] == [role == 'assistant' for role in record['roles']]
    assert not any(record['loss_mask'][:record['prefix_length']])
    assert sum(record['loss_mask'][:record['prefix_length'] + record['window']]) > 0
    assert sum(record['loss_mask'][record['prefix_length'] + record['window']:]) > 0
    record['audit_prefix_only'] = True
    record['audit_source'] = str(path)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, default=8193)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_model(args.model)
    result = dict(model_revision=Path(args.model).name, gpu=torch.cuda.get_device_name(),
                  source='Saved actual rollout tokens/logps, clipped only for a numerical audit',
                  tokens=args.tokens, toy=toy_reference(), conditions={})
    for attention in ('full', 'ps'):
        path = Path(args.run_dir) / attention / f'seed-{args.seed}-tokens.json.gz'
        record = prefix_record(path, args.tokens)
        started = time.perf_counter()
        audit = parity(base, [record])
        audit['seconds'] = time.perf_counter() - started
        with torch.no_grad():
            actual = teacher_logps(base, record)
        expected = torch.tensor([p for p, keep in zip(record['old_logps'], record['loss_mask'], strict=True) if keep], device=base.device)
        differences = (actual - expected).abs().cpu().numpy()
        positions = np.array(record['positions'])[record['loss_mask']]
        audit['position_bins'] = {}
        for name, selected in [('before_eviction', positions < record['prefix_length'] + record['window']),
                               ('after_eviction', positions >= record['prefix_length'] + record['window'])]:
            values = differences[selected]
            audit['position_bins'][name] = dict(tokens=len(values), mean_abs=float(values.mean()), p99=float(np.quantile(values, .99)))
        result['conditions'][attention] = audit
        print(json.dumps(dict(event='parity', attention=attention, **audit)), flush=True)
        del actual, expected
    if args.backward:
        record = prefix_record(Path(args.run_dir) / 'ps' / f'seed-{args.seed}-tokens.json.gz', args.tokens)
        base.requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        logps = teacher_logps(base, record, grad=True)
        (-logps.sum() / 26624).backward()
        norm = torch.nn.utils.clip_grad_norm_(base.parameters(), float('inf'), error_if_nonfinite=True)
        assert float(norm) > 0
        result['backward'] = dict(seconds=time.perf_counter() - started,
            sampled_tokens=len(logps), early_tokens=sum(record['loss_mask'][:record['prefix_length'] + record['window']]),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), gradient_norm=float(norm), optimizer_steps=0)
    result['status'] = 'GO' if all(row['status'] == 'GO' for row in result['conditions'].values()) else 'HOLD'
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'GO':
        raise RuntimeError('Investigate actual rollout/teacher likelihood differences before training')


if __name__ == '__main__':
    main()
