"""Prepare and run fixed post320 tests after the continuation has completed."""

import argparse
from dataclasses import replace
import fcntl
from pathlib import Path
import subprocess
import sys

from .hero_exports import file_hash
from .hero_finish import execute_conditions, ready, require_idle_devices
from .hero_memory import save_fixed
from .hero_pair_final import condition_result
from .hero_policy_archive import hydrate, verify_policy
from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity, read
from .overlap_config import FULL_EVALUATION_CONCURRENCY, FULL_EVALUATION_TOKEN_BOUND, SPLITS


def configurations(plan):
    goal_stop = plan['format'] == 'hero-w1024-goal25-frozen-v1'
    if not goal_stop and (plan['format'] != 'hero-w1024-post320-v1'
            or plan['initial_target_update'] != 448):
        raise ValueError('Preserve the declared320 parent and fixed448 continuation budget')
    if plan['parent_policy_version'] != 320:
        raise ValueError('Preserve the declared320 parent')
    snapshots = plan.get('configuration_snapshots', {}) if goal_stop else {}
    def load(label, fallback):
        entry = snapshots.get(label)
        path = Path(entry['path']) if entry else fallback
        if entry and file_hash(path) != entry['sha256']:
            raise ValueError('Preserve the frozen configuration snapshots')
        return HeroConfig(**read(path))
    parent = load('parent320', Path(plan['parent_run']) / 'resolved_config.json')
    continued = load('continued', Path(plan['output']) / 'resolved_config.json')
    target = plan['final_update'] if goal_stop else 448
    horizon = plan['configured_horizon'] if goal_stop else 448
    if (isinstance(target, bool) or not isinstance(target, int) or not 448 <= target <= horizon
            or parent.updates != 320 or parent.window != 1024 or continued.updates != horizon
            or continued.window != 1024 or continued.temperature != .8
            or continued.continuation_start_update != 320 or identity(continued) != plan['identity']):
        raise ValueError('The final test must follow the same declared W1024 continuation')
    shared = ('model', 'source', 'revision', 'seed', 'window', 'max_responses', 'transcript_cap',
        'thinking_token_limit', 'action_token_limit', 'format_penalty', 'invalid_penalty')
    if any(getattr(parent, field) != getattr(continued, field) for field in shared):
        raise ValueError('The comparison must preserve the parent model, task and generation controller')
    expected = dict(standard=list(range(SPLITS['post320_test'][0], sum(SPLITS['post320_test']))),
        larger=list(range(SPLITS['post320_larger'][0], sum(SPLITS['post320_larger']))))
    if any(plan['fresh_future_test_reserved'][name] != [values[0], values[-1]]
           for name, values in expected.items()):
        raise ValueError('Preserve the reserved fresh7M/8M board ranges')
    return dict(continued=continued, parent320=parent)


def prepare(plan_path, output, *, gpus, source_root, source_sha256):
    if gpus not in (4, 8):
        raise ValueError('Use the assigned four-or-eight-B200 layout')
    if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('Capture the exact evaluation source archive SHA256')
    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    configs = configurations(plan)
    cache = Path('/dev/shm/ps-hero-post320-final-cache') / file_hash(plan_path)[:16]
    target = plan.get('final_update', 448)
    goal_stop = plan['format'] == 'hero-w1024-goal25-frozen-v1'
    entries = []

    def add(label, attention='ps', window=1024, *, split='post320_test', count=512):
        config = configs['continued' if label == 'base' else label]
        version = dict(continued=target, parent320=320, base=0)[label]
        name = f'{label}-{attention}-w{window}-{split}' + ('-first128' if count == 128 else '')
        folder = output / name
        inference = replace(inference_config(config, output=folder, gpus=gpus), window=window,
            evaluation_concurrency=FULL_EVALUATION_CONCURRENCY if attention == 'full' else 64)
        config_path = folder / 'config.json'
        save_fixed(config_path, inference.model_dump())
        argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
            '--config', str(config_path), '--attention', attention, '--split', split, '--count', str(count)]
        checkpoint = cache / label if version else None
        archive = (Path(plan['output']) / f'policy-archives/w1024-{target:04d}' if label == 'continued'
            else Path(plan['parent_run']).parents[1] / 'policy-archives/w1024-0320') if version else None
        if checkpoint:
            argv.extend(['--checkpoint', str(checkpoint)])
        start, maximum = SPLITS[split]
        if not 1 <= count <= maximum:
            raise ValueError('Stay within the fixed reserved evaluation split')
        entries.append(dict(condition=name, argv=argv, config_sha256=file_hash(config_path),
            arm=label, policy_version=version, checkpoint=str(checkpoint) if checkpoint else None,
            archive=str(archive) if archive else None,
            policy_identity=str(checkpoint) if checkpoint else config.revision,
            policy_training_identity=identity(config) if version else None,
            attention=attention, window=window, split=split, count=count,
            first_board_seed=start, last_board_seed=start + count - 1,
            summary=str(folder / 'evaluation' / split / 'summary.json')))

    for label in ('continued', 'parent320', 'base'):
        add(label)
    add('continued', split='intervention')
    for label in ('continued', 'parent320', 'base'):
        for window in (256, 512, 2048, 4096):
            add(label, window=window, count=128)
        add(label, split='post320_larger', count=256)
    for label in ('continued', 'parent320', 'base'):
        add(label, attention='full')
    result = dict(format='hero-goal25-final-v1' if goal_stop else 'hero-post320-final-v1', plan=str(plan_path), plan_sha256=file_hash(plan_path),
        evaluation_source_root=str(source_root), evaluation_source_sha256=source_sha256,
        gpus=gpus, fixed_final_update=target, baseline_update=320, conditions=entries,
        primary=['continued-ps-w1024-post320_test', 'parent320-ps-w1024-post320_test'],
        original_baseline='base-ps-w1024-post320_test',
        full_attention_conditions=[row['condition'] for row in entries if row['attention'] == 'full'],
        default_phase='ps', primary_metric='Unshaped efficiency: pairs/8 - responses/96',
        checkpoint_rule=(f'Goal-stopped update{target}; no test-based checkpoint or window selection' if goal_stop
            else 'Fixed448 only; no test-based checkpoint or window selection'),
        diagnostic_windows=[256, 512, 1024, 2048, 4096],
        derived_window_1024='First128records of each matching complete512-boardPS1024condition',
        intervention=dict(split='intervention', first_board_seed=4000000, last_board_seed=4000511,
            maximum_states=128, minimum_for_textual_claim=32, boundary='before_thinking',
            branches=['unedited', 'swap', 'sham'], common_cold_rebuild=True),
        evaluation_capacity=dict(ps_games_per_gpu=64, full_games_per_gpu=16,
            full_kv_token_bound=FULL_EVALUATION_TOKEN_BOUND),
        scope='Exploratory single-lineage W1024 continuation after inspecting earlier5Mtests. '
            f'Fresh7Mstandard and8Mlargerboards; original/parent320/final{target} share the prompt and controller. '
            'Parent320 is PS-trained, not a Full-RL control. Numerical native/learner parity remains unresolved.')
    save_fixed(output / 'plan.json', result)
    return result


def inputs(final_path):
    final_path = Path(final_path)
    final = read(final_path)
    if file_hash(final['plan']) != final['plan_sha256']:
        raise ValueError('Preserve the frozen continuation plan')
    configs = configurations(read(final['plan']))
    config = configs['continued']
    ready(config.output, dict(training_barrier=final['fixed_final_update'], training_identity=identity(config)))
    source = Path(final['evaluation_source_root'])
    if (source.resolve() != Path(__file__).parents[2].resolve()
            or file_hash(source.with_suffix('.tar.gz')) != final['evaluation_source_sha256']):
        raise ValueError('Execute the captured evaluation implementation')
    return final, configs


def restore(entry):
    if not entry['policy_version']:
        return
    metadata = verify_policy(entry['archive'])
    if (metadata['catalog']['policy_version'] != entry['policy_version']
            or metadata['catalog']['identity'] != entry['policy_training_identity']):
        raise ValueError('The verified archive differs from the fixed evaluation checkpoint')
    hydrate(entry['archive'], entry['checkpoint'])


def execute(final_path, *, gpus, phase='ps'):
    if phase not in ('ps', 'full', 'all'):
        raise ValueError('Choose PS, full or all prepared conditions')
    final, _ = inputs(final_path)
    require_idle_devices(gpus)
    output = Path(final_path).parent
    with (output / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in final['conditions']:
            if phase != 'all' and entry['attention'] != phase:
                continue
            restore(entry)
            resources = dict(evaluation_concurrency=FULL_EVALUATION_CONCURRENCY,
                full_kv_token_bound=FULL_EVALUATION_TOKEN_BOUND) if entry['attention'] == 'full' else {}
            execute_conditions([entry], output, gpus=gpus, **resources)
            condition_result(entry)


def memory(final_path, *, gpus):
    with (Path(final_path).parent / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return memory_conditions(final_path, gpus=gpus)


def memory_conditions(final_path, *, gpus):
    from .hero_memory import execute as intervene

    final, configs = inputs(final_path)
    require_idle_devices(gpus)
    entry, = [row for row in final['conditions'] if row['split'] == 'intervention']
    restore(entry)
    condition_result(entry)
    config = configs['continued']
    output = Path(final_path).parent / 'memory/continued-w1024'
    collection = output / 'collection'
    episodes = Path(entry['summary']).parent / 'episodes.jsonl'
    save_fixed(output / 'collection-inputs.json', dict(episodes=str(episodes),
        episodes_sha256=file_hash(episodes), checkpoint=entry['checkpoint'], maximum=128,
        boundary='before_thinking', state_collector_sha256=file_hash(Path(__file__).with_name('rng_intervention_states.py'))))
    if not (collection / 'collection-summary.json').exists():
        subprocess.run([sys.executable, '-m', 'experiments.ps_concentration.rng_intervention_states',
            '--episodes', str(episodes), '--tokenizer', config.model, '--output', str(collection),
            '--maximum', '128', '--boundary', 'before_thinking'], check=True)
    return intervene(entry['checkpoint'], collection / 'states.jsonl.gz', config.model,
        config.source, output, gpus=gpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--plan', required=True)
    preparation.add_argument('--output', required=True)
    preparation.add_argument('--source-root', required=True)
    preparation.add_argument('--source-sha256', required=True)
    preparation.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    for name in ('run', 'memory'):
        command = commands.add_parser(name)
        command.add_argument('--plan', required=True)
        command.add_argument('--gpus', type=int, choices=(4, 8), required=True)
        if name == 'run':
            command.add_argument('--phase', choices=('ps', 'full', 'all'), default='ps')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan, args.output, gpus=args.gpus, source_root=args.source_root, source_sha256=args.source_sha256)
    elif args.command == 'memory':
        memory(args.plan, gpus=args.gpus)
    else:
        execute(args.plan, gpus=args.gpus, phase=args.phase)


if __name__ == '__main__':
    main()
