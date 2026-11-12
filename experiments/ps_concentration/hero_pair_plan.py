"""Freeze the approved short sweep and paired continuation comparison."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity


WINDOWS = (1024, 512, 256, 128, 64)
CONTINUATION_VERSIONS = (128, 144, 160, 176, 192)


def prepare(parent_run, output, *, source_root, source_sha256):
    parent_run, output = Path(parent_run), Path(output)
    if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('Pin the qualified paired implementation SHA256')
    parent = HeroConfig(**json.loads((parent_run / 'resolved_config.json').read_text()))
    if parent.seed != 11 or parent.window != 1024 or parent.updates != 128:
        raise ValueError('The approved parent is seed11,W1024,128updates')
    policies = output / 'policies'
    files, layouts = {}, {}
    for gpus in (4, 8):
        entries = []
        conditions = [(f'parent-w{window}', 'parent', 'ps', window) for window in WINDOWS]
        conditions.extend([('base-full', 'base', 'full', 1024), ('parent-full', 'parent', 'full', 1024)])
        for name, policy, attention, window in conditions:
            folder = output / 'initial-sweep' / f'{gpus}gpu' / name
            config = replace(inference_config(parent, output=folder, gpus=gpus), window=window,
                source_sha256=source_sha256,
                evaluation_concurrency=4 if attention == 'full' else 32 // gpus)
            path = folder / 'config.json'
            data = (json.dumps(config.model_dump(), indent=2, allow_nan=False) + '\n').encode()
            files[path] = data
            argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
                '--config', str(path), '--attention', attention, '--split', 'validation', '--count', '32']
            if policy == 'parent':
                argv.extend(['--checkpoint', str(policies / 'parent-0128')])
            entries.append(dict(condition=name, argv=argv, config_sha256=hashlib.sha256(data).hexdigest(),
                attention=attention, window=window, policy=policy, policy_version=128 if policy == 'parent' else 0,
                count=32, first_board_seed=1100000, last_board_seed=1100031,
                summary=str(folder / 'evaluation/validation/summary.json')))
        layouts[str(gpus)] = entries
    plan = dict(status='PREPARED_NOT_EXECUTED', parent_run=str(parent_run),
        parent_identity=identity(parent), parent_barrier=128,
        source_root=str(source_root), source_sha256=source_sha256, output=str(output),
        policy_directory=str(policies), initial_sweep=layouts,
        branches=[dict(name=f'w{window}', window=window, gpus=4, target_update=192,
            initial_update=128, additional_updates=64, additional_episodes=8192,
            global_batch_size=128, group_size=8, schedule_updates=128, terminal_learning_rate=4e-7,
            output=str(output / 'arms' / f'w{window}'),
            wandb_id=f'rng-4b-ps-grpo-s11-hero-v2-cont-w{window}-a1') for window in (1024, 256)],
        paired_data=dict(seed=11, board_indices=[2048, 3071], permutation_pool=[0, 999999],
            generation_rng='training_rng(seed,update-1,board_index,rollout_index)',
            reference='Original starting checkpoint under each arm attention mask'),
        storage=dict(archive_reference='parent-0128', archive_kind='Exact BF16 XOR delta plus full reference',
            shared_checkpoint_io='Serialize asynchronous full checkpoint writes across branches',
            required_shared_headroom_gib=185),
        final_evaluation=dict(validation_boards=128, test_boards=512,
            candidate_versions=list(CONTINUATION_VERSIONS),
            selection='Highest complete unshaped efficiency return under each training window; earlier on ties; parent128 remains an explicitly labeled baseline',
            primary_attention_window=256, reference_attention_window=1024,
            diagnostic_windows=[64, 128, 256, 512, 1024, 2048, 4096], diagnostic_boards=128,
            larger_boards=256, intervention_collection_boards=512,
            causal_interventions='After continuation training and validation selection; cold rebuild,edit,sham'),
        scope='One shared parent seed; W256 is a transfer continuation from W1024training. '
              'The two branches are not independent training seeds. No memory prompt,teacher or external memory.',
        guard='Finish parent128 and its workers before GPU evaluation or continuation; keep all final tests after the paired continuations.')
    files[output / 'plan.json'] = (json.dumps(plan, indent=2, allow_nan=False) + '\n').encode()
    if any(path.exists() and path.read_bytes() != data for path, data in files.items()):
        raise ValueError('Preserve the frozen paired plan; use a new directory for a changed experiment')
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    result = prepare(args.parent_run, args.output, source_root=args.source_root, source_sha256=args.source_sha256)
    print(json.dumps(dict(status=result['status'], parent_barrier=128,
        branch_targets=[row['target_update'] for row in result['branches']])))


if __name__ == '__main__':
    main()
