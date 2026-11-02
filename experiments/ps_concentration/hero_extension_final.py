"""Evaluate the fixed320 extension on fresh boards after both arms finish."""

import argparse
from dataclasses import replace
import fcntl
from pathlib import Path

from .hero_exports import file_hash
from .hero_finish import execute_conditions, ready, require_idle_devices
from .hero_memory import save_fixed
from .hero_pair_final import condition_result
from .hero_policy_archive import hydrate, verify_policy
from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity, read
from .overlap_config import FULL_EVALUATION_CONCURRENCY, FULL_EVALUATION_TOKEN_BOUND, SPLITS


def configurations(plan):
    if plan['target_update'] != 320 or plan['parent_policy_version'] != 192:
        raise ValueError('The extension fixes the common192 baseline and final320 checkpoints')
    parent = HeroConfig(**read(Path(plan['parent_run']) / 'resolved_config.json'))
    arms = {f'w{row["window"]}': HeroConfig(**row) for row in plan['branches']}
    if set(arms) != {'w1024', 'w512'} or parent.window != 1024 or parent.updates != 192:
        raise ValueError('Use the common W1024parent192 and both declared extension windows')
    if any(c.updates != 320 or c.source_sha256 != plan['source_sha256'] for c in arms.values()):
        raise ValueError('Both final checkpoints must follow the captured320-update plan')
    return {'parent': parent, **arms}


def prepare(plan_path, output, *, gpus, source_root, source_sha256):
    if gpus not in (4, 8):
        raise ValueError('Use the assigned four-or-eight-B200 evaluation layout')
    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    configs = configurations(plan)
    fingerprint = file_hash(plan_path)
    cache = Path('/dev/shm/ps-hero-extension-final-cache') / fingerprint[:16]
    entries = []

    def add(label, attention, window, *, split='extension_test', count=512):
        config = configs['parent' if label == 'base' else label]
        version = 0 if label == 'base' else 192 if label == 'parent' else 320
        name = f'{label}-{attention}-w{window}-{split}'
        if count == 128:
            name += '-first128'
        folder = output / name
        inference = replace(inference_config(config, output=folder, gpus=gpus), window=window,
            evaluation_concurrency=16 if attention == 'full' else 128 // gpus)
        config_path = folder / 'config.json'
        save_fixed(config_path, inference.model_dump())
        argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
            '--config', str(config_path), '--attention', attention, '--split', split, '--count', str(count)]
        checkpoint = cache / label if version else None
        archive = ((Path(plan['parent_run']).parents[1] / 'policy-archives/w1024-0192')
            if label == 'parent' else Path(plan['output']) / f'policy-archives/{label}-0320') if version else None
        if checkpoint:
            argv.extend(['--checkpoint', str(checkpoint)])
        start, maximum = SPLITS[split]
        if not 1 <= count <= maximum:
            raise ValueError('Preserve the fixed fresh evaluation seed range')
        entries.append(dict(condition=name, argv=argv, config_sha256=file_hash(config_path),
            arm=label, policy_version=version, checkpoint=str(checkpoint) if checkpoint else None,
            archive=str(archive) if archive else None,
            policy_identity=str(checkpoint) if checkpoint else config.revision,
            policy_training_identity=identity(config) if version else None,
            attention=attention, window=window, split=split, count=count,
            first_board_seed=start, last_board_seed=start+count-1,
            summary=str(folder / 'evaluation' / split / 'summary.json')))

    for label in ('w512', 'w1024', 'parent'):
        add(label, 'ps', 512)
    for window in (1024,):
        for label in ('w512', 'w1024', 'parent'):
            add(label, 'ps', window)
    for label in ('w1024', 'w512', 'parent'):
        add(label, 'full', 1024)
    for attention, window in (('full', 1024), ('ps', 512), ('ps', 1024)):
        add('base', attention, window)
    for label in ('w512', 'w1024', 'parent'):
        add(label, 'ps', 512, split='extension_larger', count=256)
    for label in ('w512', 'w1024'):
        for window in (256, 2048, 4096):
            add(label, 'ps', window, count=128)
        add(label, 'ps', 512, split='intervention')
    add('w1024', 'ps', 1024, split='intervention')
    result = dict(format='hero-extension-final-v1', plan=str(plan_path), plan_sha256=fingerprint,
        evaluation_source_root=str(source_root), evaluation_source_sha256=source_sha256,
        gpus=gpus, conditions=entries, fixed_final_update=320, baseline_update=192,
        primary=['w512-ps-w512-extension_test', 'w1024-ps-w512-extension_test'],
        secondary_baseline='parent-ps-w512-extension_test',
        primary_metric='Unshaped efficiency reward: pairs/8 - responses/96',
        reward_reporting='Report shaped reward, pairs/8, completion, and response count separately',
        diagnostic_windows=[256, 512, 1024, 2048, 4096],
        derived_sweep_windows=dict(windows=[512, 1024],
            source='First128 fresh test records from the corresponding complete512-board conditions'),
        intervention=dict(split='intervention', first_board_seed=4000000, last_board_seed=4000511,
            maximum_states=128, minimum_for_textual_claim=32,
            branches=['unedited', 'swap', 'sham'], cold_rebuild_all_branches=True),
        checkpoint_rule='Final320 only; no window/checkpoint selection using these evaluations',
        scope='Exploratory continuations after inspecting earlier tests; fresh test seeds5000000–5000511 '
            'and larger seeds6000000–6000255. One common seed11 lineage, not independent replications.')
    save_fixed(output / 'plan.json', result)
    return result


def execute(final_path, *, gpus):
    final_path = Path(final_path)
    final = read(final_path)
    if file_hash(final['plan']) != final['plan_sha256']:
        raise ValueError('Preserve the frozen training and fresh evaluation plans')
    source = Path(final['evaluation_source_root'])
    if (source.resolve() != Path(__file__).parents[2].resolve()
            or file_hash(source.with_suffix('.tar.gz')) != final['evaluation_source_sha256']):
        raise ValueError('Execute the captured evaluation source')
    for label, config in configurations(read(final['plan'])).items():
        if label != 'parent':
            ready(config.output, dict(training_barrier=320, training_identity=identity(config)))
    require_idle_devices(gpus)
    output = final_path.parent
    with (output / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in final['conditions']:
            if entry['policy_version']:
                metadata = verify_policy(entry['archive'])
                if (metadata['catalog']['policy_version'] != entry['policy_version']
                        or metadata['catalog']['identity'] != entry['policy_training_identity']):
                    raise ValueError('The inference archive differs from its fixed policy identity')
                hydrate(entry['archive'], entry['checkpoint'])
            resources = dict(evaluation_concurrency=FULL_EVALUATION_CONCURRENCY,
                full_kv_token_bound=FULL_EVALUATION_TOKEN_BOUND) if entry['attention'] == 'full' else {}
            execute_conditions([entry], output, gpus=gpus, **resources)
            condition_result(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--plan', required=True)
    preparation.add_argument('--output', required=True)
    preparation.add_argument('--source-root', required=True)
    preparation.add_argument('--source-sha256', required=True)
    preparation.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    run = commands.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan, args.output, gpus=args.gpus, source_root=args.source_root,
            source_sha256=args.source_sha256)
    else:
        execute(args.plan, gpus=args.gpus)


if __name__ == '__main__':
    main()
