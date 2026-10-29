"""Exercise repeated native training batches without an optimizer update."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .check_policy import teacher_logps
from .policy import load_model
from .rng_collect import NativeCollector
from .rng_training_config import MatchingConfig, training_rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = MatchingConfig(model=args.model, source=args.source, max_responses=2)
    config.save(output / 'resolved_config.yaml')
    policy = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collector = NativeCollector(policy, tokenizer, config)
    audits = []
    for batch in range(2):
        specs = [dict(board_seed=7 + batch, generation_seed=training_rng(11, 0, batch, rollout),
                      episode_id=f'smoke-b{batch}-r{rollout}', board_index=batch, rollout_index=rollout)
                 for rollout in range(2)]
        records = collector.collect(specs, output / f'batch-{batch}')
        for record in records:
            actual = teacher_logps(policy, record, temperature=config.temperature)
            old = torch.tensor([value for value in record['old_tempered_logps'] if value is not None], device=policy.device)
            differences = (actual - old).abs().cpu().numpy()
            audit = dict(episode_id=record['episode_id'], board_seed=record['board_seed'],
                         generation_seed=record['generation_seed'], generated_tokens=len(actual),
                         early_loss_tokens=sum(record['loss_mask'][:record['prefix_length'] + record['window']]),
                         mean_abs=float(differences.mean()), p99=float(np.quantile(differences, .99)),
                         return_value=record['training_return'])
            assert audit['early_loss_tokens'] > 0
            assert audit['mean_abs'] <= .02 and audit['p99'] <= .15, audit
            audits.append(audit)
        assert records[0]['generation_seed'] != records[1]['generation_seed']
    result = dict(scope='Two response numerical smoke, not task-performance evidence; no optimizer step',
                  status='GO' if all(a['mean_abs'] <= .01 and a['p99'] <= .05 for a in audits) else 'HOLD',
                  audits=audits, optimizer_steps=0)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
