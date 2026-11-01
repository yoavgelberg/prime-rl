"""Validate, select and evaluate both completed window continuations."""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .hero_continue import branch_config
from .hero_exports import file_hash
from .hero_finish import completed_condition, execute_conditions, ready, require_idle_devices
from .hero_memory import execute as intervene, save_fixed
from .hero_pair_plan import CONTINUATION_VERSIONS
from .hero_policy_archive import hydrate, verify_policy
from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity, read, utc, write
from .overlap_config import RunConfig, SPLITS
from .rng_report import load_condition


def branch_configs(plan):
    parent = HeroConfig(**read(Path(plan['parent_run']) / 'resolved_config.json'))
    return {row['name']: branch_config(parent, window=row['window'], output=row['output'],
        node_dir=f'/dev/shm/ps-hero-pair-final-config/{row["name"]}/control',
        source_sha256=plan['source_sha256'], deadline=0.) for row in plan['branches']}


def completed_branches(plan):
    configs = branch_configs(plan)
    for config in configs.values():
        ready(config.output, dict(training_barrier=192, training_identity=identity(config)))
    return configs


def policy_location(plan, arm, version):
    if version == 128:
        return Path(plan['policy_directory']) / 'parent-0128'
    return Path(plan['output']) / 'policy-archives' / f'{arm}-{version:04d}'


def policy_cache(plan_path, arm, version):
    plan = read(plan_path)
    if version == 128:
        return policy_location(plan, arm, version)
    return Path('/dev/shm/ps-hero-pair-policy-cache') / file_hash(plan_path)[:16] / arm / f'policy-{version:04d}'


def prepare_condition(plan_path, output, *, name, arm, version, attention, window, split, count, gpus):
    plan = read(plan_path)
    config = branch_configs(plan)[arm]
    folder = Path(output) / name
    inference = replace(inference_config(config, output=folder, gpus=gpus), window=window,
        evaluation_concurrency=4 if attention == 'full' else 128 // gpus)
    config_path = folder / 'config.json'
    save_fixed(config_path, inference.model_dump())
    argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
        '--config', str(config_path), '--attention', attention, '--split', split, '--count', str(count)]
    checkpoint = policy_cache(plan_path, arm, version) if version else None
    if checkpoint:
        argv.extend(['--checkpoint', str(checkpoint)])
    start, maximum = SPLITS[split]
    if not 1 <= count <= maximum:
        raise ValueError('Evaluation stays within the fixed board split')
    return dict(condition=name, argv=argv, config_sha256=file_hash(config_path), arm=arm,
        policy_version=version, checkpoint=str(checkpoint) if checkpoint else None,
        policy_identity=str(checkpoint) if checkpoint else config.revision,
        attention=attention, window=window, split=split, count=count,
        first_board_seed=start, last_board_seed=start + count - 1,
        summary=str(folder / 'evaluation' / split / 'summary.json'),
        policy_training_identity=plan['parent_identity'] if version == 128 else identity(config) if version else None)


def hydrate_condition(plan_path, entry):
    if not entry['policy_version']:
        return
    plan = read(plan_path)
    archive = policy_location(plan, entry['arm'], entry['policy_version'])
    metadata = verify_policy(archive)
    if (metadata['catalog']['policy_version'] != entry['policy_version']
            or metadata['catalog']['identity'] != entry['policy_training_identity']):
        raise ValueError('Candidate archive differs from its declared training policy')
    if archive != Path(entry['checkpoint']):
        hydrate(archive, entry['checkpoint'])


def condition_result(entry):
    if file_hash(entry['argv'][entry['argv'].index('--config') + 1]) != entry['config_sha256']:
        raise ValueError('Prespecified evaluation configuration changed')
    if not completed_condition(entry):
        raise ValueError('A complete greedy evaluation is required')
    if entry['policy_version']:
        metadata = read(Path(entry['checkpoint']) / 'experiment.json')
        if (metadata['policy_version'] != entry['policy_version'] or any(
                metadata.get(key) != value for key, value in entry['policy_training_identity'].items())):
            raise ValueError('Export metadata differs from its declared training policy')
    config = RunConfig(**read(entry['argv'][entry['argv'].index('--config') + 1]))
    return load_condition(dict(name=entry['condition'], policy=entry['arm'] if entry['policy_version'] else 'base',
        training_seed=config.seed if entry['policy_version'] else None,
        policy_version=entry['policy_version'], policy_identity=entry['policy_identity'],
        attention=entry['attention'], window=entry['window'], split=entry['split'], greedy=True,
        expected_boards=entry['count'], summary=entry['summary'],
        episodes=str(Path(entry['summary']).parent / 'episodes.jsonl'),
        controller={key: getattr(config, key) for key in
            ('thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')}))


def prepare_validation(plan_path, output, *, gpus):
    plan = read(plan_path)
    entries = [prepare_condition(plan_path, output, name=f'{arm}-v{version:04d}', arm=arm,
        version=version, attention='ps', window=config.window, split='validation', count=128, gpus=gpus)
        for arm, config in branch_configs(plan).items() for version in CONTINUATION_VERSIONS]
    result = dict(plan=str(plan_path), plan_sha256=file_hash(plan_path), layout=gpus, conditions=entries,
        candidates=list(CONTINUATION_VERSIONS), selection='Maximum validation efficiency return; earliest version on ties',
        scope='Shared parent128 is an eligible baseline, not a newly trained continuation checkpoint')
    save_fixed(Path(output) / 'plan.json', result)
    return result


def select(validation):
    plan = read(validation['plan'])
    if file_hash(validation['plan']) != validation['plan_sha256']:
        raise ValueError('The paired experiment plan changed')
    completed_branches(plan)
    selected = {}
    for arm, config in branch_configs(plan).items():
        entries = [row for row in validation['conditions'] if row['arm'] == arm]
        if len(entries) != 5 or {row['policy_version'] for row in entries} != set(CONTINUATION_VERSIONS):
            raise ValueError('Selection requires every prespecified continuation checkpoint')
        candidates = []
        for entry in entries:
            if (entry['split'] != 'validation' or entry['count'] != 128
                    or entry['attention'] != 'ps' or entry['window'] != config.window):
                raise ValueError('Select only on the fixed validation split under the training window')
            result = condition_result(entry)
            candidates.append(dict(version=entry['policy_version'], checkpoint=entry['checkpoint'],
                mean_return=result['mean_return'], mean_pairs=result['mean_pairs'],
                summary=entry['summary'], board_seeds_sha256=result['board_seeds_sha256'],
                episodes_sha256=file_hash(Path(entry['summary']).parent / 'episodes.jsonl')))
        if len({row['board_seeds_sha256'] for row in candidates}) != 1:
            raise ValueError('All checkpoint candidates must use identical validation boards')
        winner = max(candidates, key=lambda row: (row['mean_return'], -row['version']))
        selected[arm] = dict(winner, candidates=candidates, training_window=config.window,
            shared_parent_baseline=winner['version'] == 128,
            additional_updates=winner['version'] - 128,
            additional_episodes=(winner['version'] - 128) * config.batch_size,
            split='validation', boards=128,
            selection=validation['selection'])
    return dict(selected=selected,
        validation_protocol_sha256=hashlib.sha256(json.dumps(validation, sort_keys=True).encode()).hexdigest(),
        plan_sha256=validation['plan_sha256'],
        scope='One shared parent training seed11. Branches are paired transfer continuations, not independent seed replications.')


def prepare_final(plan_path, selection, output, *, gpus):
    plan = read(plan_path)
    if selection['plan_sha256'] != file_hash(plan_path):
        raise ValueError('Final evaluation must use the frozen paired plan selection')
    files = []

    def add(name, arm, version, attention, window, split='test', count=512):
        files.append(prepare_condition(plan_path, output, name=name, arm=arm, version=version,
            attention=attention, window=window, split=split, count=count, gpus=gpus))

    for label, arm, version in [('base', 'w1024', 0), ('parent', 'w1024', 128),
            *[(arm, arm, row['version']) for arm, row in selection['selected'].items()]]:
        for window in (256, 1024):
            add(f'{label}-ps-w{window}-test', arm, version, 'ps', window)
        add(f'{label}-full-test', arm, version, 'full', 1024)
        add(f'{label}-ps-w256-larger', arm, version, 'ps', 256, 'larger', 256)
    for arm, row in selection['selected'].items():
        for window in plan['final_evaluation']['diagnostic_windows']:
            add(f'{arm}-window-{window}', arm, row['version'], 'ps', window, count=128)
        add(f'{arm}-intervention-collection', arm, row['version'], 'ps', 256, 'intervention', 512)
    result = dict(plan=str(plan_path), plan_sha256=file_hash(plan_path), selection=selection,
        conditions=files, primary=['w256-ps-w256-test', 'w1024-ps-w256-test'],
        comparison_reference='w1024-full-test',
        scope='All final tests follow both192-update branches and validation-only selection. '
              'Interventions compare both selected branches under W256. Shared parent128 selection is explicitly labeled.')
    save_fixed(Path(output) / 'plan.json', result)
    return result


def execute_entries(plan_path, entries, output, *, gpus):
    for entry in entries:
        hydrate_condition(plan_path, entry)
        execute_conditions([entry], output, gpus=gpus)
        condition_result(entry)


def memory(plan_path, final, output, *, gpus):
    plan = read(plan_path)
    config = branch_configs(plan)['w1024']
    summaries = {}
    for arm in ('w1024', 'w256'):
        entry = next(row for row in final['conditions'] if row['condition'] == f'{arm}-intervention-collection')
        hydrate_condition(plan_path, entry)
        condition_result(entry)
        collection = Path(output) / arm / 'collection'
        inputs = dict(episodes=str(Path(entry['summary']).parent / 'episodes.jsonl'),
            episodes_sha256=file_hash(Path(entry['summary']).parent / 'episodes.jsonl'),
            checkpoint=entry['checkpoint'], window=256, maximum=128, boundary='before_thinking')
        save_fixed(collection.parent / 'collection-inputs.json', inputs)
        if not (collection / 'collection-summary.json').exists():
            subprocess.run([sys.executable, '-m', 'experiments.ps_concentration.rng_intervention_states',
                '--episodes', inputs['episodes'], '--tokenizer', config.model, '--output', str(collection),
                '--maximum', '128', '--boundary', 'before_thinking'], check=True)
        summaries[arm] = intervene(entry['checkpoint'], collection / 'states.jsonl.gz',
            config.model, config.source, Path(output) / arm, gpus=gpus)
    return summaries


def finish(plan_path, output, *, gpus):
    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    completed_branches(plan)
    require_idle_devices(gpus)
    session = output / 'session.json'
    if session.exists():
        current = read(session)
        if current['plan_sha256'] != file_hash(plan_path):
            raise ValueError('Resume the same paired final experiment')
    else:
        current = dict(plan_sha256=file_hash(plan_path), layout=gpus)
        save_fixed(session, current)
    validation = prepare_validation(plan_path, output / 'validation', gpus=current['layout'])
    execute_entries(plan_path, validation['conditions'], output / 'validation-execution', gpus=gpus)
    selection = select(validation)
    save_fixed(output / 'selection.json', selection)
    final = prepare_final(plan_path, selection, output / 'final', gpus=current['layout'])
    execute_entries(plan_path, final['conditions'], output / 'final-execution', gpus=gpus)
    memory(plan_path, final, output / 'memory', gpus=gpus)
    subprocess.run([sys.executable, '-m', 'experiments.ps_concentration.hero_pair_report',
        '--final-plan', str(output / 'final/plan.json'), '--output', str(output / 'results'),
        '--memory', str(output / 'memory')], check=True)
    result = dict(utc=utc(), status='PAIRED_RESULTS_AND_INTERVENTIONS_COMPLETE',
        selection=str(output / 'selection.json'), results=str(output / 'results'),
        scope='One shared parent seed; independent seed replication remains unmeasured. '
              'No automatic goal completion or allocation release.')
    write(output / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    with (Path(plan['parent_run']).parent / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(finish(args.plan, args.output, gpus=args.gpus), indent=2))


if __name__ == '__main__':
    main()
