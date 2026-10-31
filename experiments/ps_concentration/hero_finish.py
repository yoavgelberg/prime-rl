"""Resume deferred validation and final performance evaluation after hero training."""

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .hero_exports import file_hash, write
from .overlap_config import RunConfig
from .rng_final_plan import prepare_final_plan
from .rng_report import load_condition
from .rng_select import argument, select_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def ready(run, plan):
    run = Path(run)
    state = read(run / 'state.json')
    target = plan['training_barrier']
    if state['policy_version'] != target or not (run / 'training-complete.json').exists():
        raise ValueError('Finish the complete training budget before any GPU evaluation')
    complete = read(run / 'training-complete.json')
    sidecar = read(run / 'checkpoints/latest.pt.json')
    if (state['policy_version'] != target or complete.get('reached_target') is not True
            or complete['policy_version'] != target or sidecar['policy_version'] != target):
        raise ValueError('Finish the complete training budget before any GPU evaluation')
    if state['identity'] != plan['training_identity'] or sidecar['identity'] != state['identity']:
        raise ValueError('Completed training differs from the frozen evaluation identity')
    for path in run.glob('processes-*.json'):
        processes = read(path)
        for pid in [processes['coordinator'], *[row['pid'] for row in processes['children']]]:
            command = Path(f'/proc/{pid}/cmdline')
            if command.exists():
                argv = command.read_text().split('\0')
                if 'experiments.ps_concentration.hero_v2' in argv:
                    config_path = Path(argument(argv, '--config'))
                    if not config_path.exists() or Path(read(config_path)['output']).resolve() == run.resolve():
                        raise ValueError('Tracked training workers are still live; defer GPU evaluation')
    if (run / 'checkpoints/latest.pt').stat().st_size != sidecar['bytes']:
        raise ValueError('Final snapshot does not match its committed metadata')
    return state


def require_idle_devices(gpus):
    assigned = [value.strip() for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value.strip()]
    if len(assigned) != gpus or gpus not in (4, 8):
        raise ValueError('Use the four or eight GPUs assigned to this allocation')
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {index.strip(): uuid.strip() for index, uuid in csv.reader(devices.splitlines())}
    uuids = {mapping.get(value, value) for value in assigned}
    if not uuids <= set(mapping.values()):
        raise ValueError('Assigned GPU identity cannot be resolved')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True)
    if any(row[0].strip() in uuids for row in csv.reader(processes.splitlines()) if row):
        raise ValueError('Assigned GPUs still have compute processes; defer evaluation')


def preserve_candidates(run, plan):
    run = Path(run)
    destination = Path(plan['exports']['directory'])
    destination.mkdir(parents=True, exist_ok=True)
    catalogs = [read(run / f'export-catalog/policy-{version:04d}.json')
        for version in plan['exports']['versions']]
    pending_bytes = 0
    for version, catalog in zip(plan['exports']['versions'], catalogs, strict=True):
        if (catalog.get('status') != 'VERIFIED' or catalog['policy_version'] != version
                or catalog['identity'] != plan['training_identity'] or catalog['unique_parameters'] != 4022468096):
            raise ValueError('Every scheduled policy requires matching exact-weight verification')
        if not (destination / f'policy-{version:04d}').exists():
            pending_bytes += sum(row['bytes'] for row in catalog['files'])
    if shutil.disk_usage(destination).free < pending_bytes:
        raise ValueError('Insufficient shared storage to preserve all candidate exports')
    receipts = []
    for version, catalog in zip(plan['exports']['versions'], catalogs, strict=True):
        target = destination / f'policy-{version:04d}'
        temporary = target.with_name(target.name + '.copying')
        if target.exists():
            for row in catalog['files']:
                path = target / row['name']
                if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
                    raise ValueError('Previously preserved policy changed')
        else:
            temporary.mkdir(exist_ok=True)
            for row in catalog['files']:
                source = Path(catalog['policy']) / row['name']
                path = temporary / row['name']
                value = hashlib.sha256()
                with source.open('rb') as src, path.open('wb') as dst:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        dst.write(block)
                        value.update(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if path.stat().st_size != row['bytes'] or value.hexdigest() != row['sha256']:
                    raise ValueError('Copied inference export differs from its verified source')
            temporary.rename(target)
        receipts.append(dict(policy_version=version, policy=str(target), files=catalog['files']))
    write(run / 'preserved-policy-exports.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        policies=receipts, scope='All scheduled verified BF16 inference exports copied without retiring source weights'))
    return receipts


def completed_condition(entry):
    summary_path = Path(entry['summary'])
    if not summary_path.exists():
        return False
    summary = read(summary_path)
    config = RunConfig(**read(argument(entry['argv'], '--config')))
    if '--window' in entry['argv']:
        config = replace(config, window=int(argument(entry['argv'], '--window')))
    actual = RunConfig(**read(summary_path.parent / 'evaluation-protocol.json')['config'])
    if actual.scientific_identity() != config.scientific_identity():
        raise ValueError('Saved evaluation has a different model or controller')
    checkpoint = argument(entry['argv'], '--checkpoint') if '--checkpoint' in entry['argv'] else None
    identity = str(Path(checkpoint).resolve()) if checkpoint else config.revision
    split = argument(entry['argv'], '--split')
    window = int(argument(entry['argv'], '--window')) if '--window' in entry['argv'] else config.window
    if actual.window != window:
        raise ValueError('Saved evaluation uses a different attention window')
    condition = load_condition(dict(name=entry['condition'], policy='PS RL' if checkpoint else 'base',
        policy_version=entry['policy_version'], policy_identity=identity, training_seed=config.seed if checkpoint else None,
        split=split, attention=entry['attention'], window=window, greedy=True, expected_boards=entry['count'],
        controller={key: getattr(config, key) for key in
            ('thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')},
        summary=str(summary_path), episodes=str(summary_path.parent / 'episodes.jsonl')))
    if any(abs(condition[key] - summary[key]) > 1e-10 for key in
        ('mean_return', 'matched_fraction', 'completion_rate', 'mean_attempts')):
        raise ValueError('Saved evaluation summary disagrees with its complete episode records')
    return True


def execute_conditions(entries, output, *, gpus, invoke=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        config_path = Path(argument(entry['argv'], '--config'))
        if file_hash(config_path) != entry['config_sha256']:
            raise ValueError('Prespecified evaluation configuration changed')
        if completed_condition(entry):
            continue
        argv = [sys.executable, *entry['argv'][1:], '--gpus', str(gpus)]
        write(output / 'live.json', dict(utc=datetime.now(timezone.utc).isoformat(),
            condition=entry['condition'], argv=argv, physical_gpus=gpus, status='EVALUATING'))
        if invoke is None:
            with (output / f'{entry["condition"]}.log').open('a') as handle:
                subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=True)
        else:
            invoke(argv)
        if not completed_condition(entry):
            raise ValueError('Evaluation command exited without a complete validated result')


def finish(run, plan_path, output, *, gpus):
    run, plan_path, output = map(Path, (run, plan_path, output))
    plan = read(plan_path)
    ready(run, plan)
    require_idle_devices(gpus)
    output.mkdir(parents=True, exist_ok=True)
    session_path = output / 'session.json'
    session = read(session_path) if session_path.exists() else dict(
        plan_layout=gpus, plan=str(plan_path), plan_sha256=file_hash(plan_path))
    if session['plan'] != str(plan_path) or session['plan_sha256'] != file_hash(plan_path):
        raise ValueError('Resume the same frozen validation plan')
    write(session_path, session)
    preserve_candidates(run, plan)
    layout = session['plan_layout']
    execute_conditions(plan['layouts'][str(layout)], output / 'validation', gpus=gpus)
    selected = select_checkpoint(plan_path, run, gpus=layout)
    selection_path = output / 'selected-checkpoint.json'
    if selection_path.exists() and read(selection_path) != selected:
        raise ValueError('Preserve the existing checkpoint selection receipt')
    write(selection_path, selected)
    final = prepare_final_plan(plan_path, run, selection_path, output / 'final',
        gpus=layout, validation_gpus=layout)
    execute_conditions(final['commands'], output / 'final-execution', gpus=gpus)
    argv = [sys.executable, '-m', 'experiments.ps_concentration.rng_report',
        '--manifest', final['report_manifest'], '--output', str(output / 'results'),
        '--tokenizer', RunConfig(**read(argument(plan['layouts'][str(layout)][0]['argv'], '--config'))).model]
    if selected['update'] > 0:
        argv.extend(['--examples-condition', final['selected_test']['ps']])
    with (output / 'report.log').open('a') as handle:
        subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
        status='PERFORMANCE_COMPLETE_INTERVENTIONS_PENDING', selected_update=selected['update'],
        selection=str(selection_path), results=str(output / 'results'),
        intervention_status=final['intervention_status'],
        scope='One PS training seed; native greedy performance results complete. '
            'Memory collection is saved, but causal interventions and training replication remain incomplete.')
    write(output / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    with (Path(args.run_dir).parent / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(finish(args.run_dir, args.plan, args.output, gpus=args.gpus)))


if __name__ == '__main__':
    main()
