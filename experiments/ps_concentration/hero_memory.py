"""Run all collected hero memory interventions on independent assigned GPUs."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hero_exports import file_hash, write
from .hero_finish import completed_condition, ready, require_idle_devices
from .rng_intervene import summarize_results


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def save_fixed(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Resume the same intervention inputs and implementation')
    if not path.exists():
        write(path, value)


def read_states(path):
    with gzip.open(path, 'rt') as handle:
        states = [json.loads(line) for line in handle if line.strip()]
    if len(states) > 128 or len({state['state_id'] for state in states}) != len(states):
        raise ValueError('Use at most 128 distinct preselected intervention states')
    if any(state.get('state_boundary') != 'before_thinking'
           or not 4000000 <= state['board_seed'] <= 4000511 for state in states):
        raise ValueError('Primary interventions precede thinking on the reserved separate board split')
    return states


def state_shards(states, gpus):
    return [states[rank::gpus] for rank in range(gpus)]


def aggregate(states, results, *, checkpoint, samples=32):
    rows = []
    for state in states:
        row = read(Path(results) / (state['state_id'] + '.json'))
        if (row.get('checkpoint') != str(Path(checkpoint).resolve()) or row.get('samples') != samples
                or row.get('state_sha256') != state_hash(state)):
            raise ValueError('An intervention result has a different state, policy or sample count')
        if ('edits' in state) != ('unedited' in row):
            raise ValueError('Every testable state needs all three branches; untestable states remain counted')
        if 'edits' in state and any(branch not in row for branch in ('unedited', 'swap', 'sham')):
            raise ValueError('Every testable state needs all three branches')
        rows.append(row)
    summary = summarize_results(states, rows)
    summary.update(checkpoint=str(Path(checkpoint).resolve()), samples=samples,
        result_files=[dict(path=str(Path(results) / (state['state_id'] + '.json')),
            sha256=file_hash(Path(results) / (state['state_id'] + '.json'))) for state in states])
    return summary


def save_table(summary, output):
    fields = ['subset', 'branch', 'qualified_states', 'executed_states', 'untestable_states',
        'pending_states', 'denominator', 'greedy_correct', 'greedy_follows_false',
        'probability_correct_partner', 'probability_following_false', 'causal_textual_memory_established']
    with (Path(output) / 'intervention-table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subset in ('branches', 'conditional_unedited_selects_p'):
            for branch in ('unedited', 'swap', 'sham'):
                writer.writerow(dict(subset=subset, branch=branch, **{key: summary[key] for key in
                    ('qualified_states', 'executed_states', 'untestable_states', 'pending_states',
                     'causal_textual_memory_established')}, denominator=summary[subset]['denominator'],
                    **summary[subset][branch]))


def execute(checkpoint, states_path, reference, source, output, *, gpus, invoke=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    states = read_states(states_path)
    implementation = [Path(__file__), Path(__file__).with_name('rng_intervene.py'),
        Path(__file__).with_name('native_cold.py'), Path(__file__).with_name('native_teacher.py'),
        Path(__file__).with_name('rng_intervention_states.py')]
    from prime_rl.inference.vllm import ps_interactive
    implementation.append(Path(ps_interactive.__file__))
    session = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_metadata=file_hash(Path(checkpoint) / 'experiment.json'),
        reference=str(reference), source=str(source), samples=32,
        states=[dict(state_id=state['state_id'], sha256=state_hash(state)) for state in states],
        implementation={str(path.relative_to(Path(__file__).parents[2])): file_hash(path) for path in implementation})
    save_fixed(output / 'session.json', session)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != gpus or gpus not in (4, 8):
        raise ValueError('Use the assigned four- or eight-GPU layout')
    require_idle_devices(gpus)
    results = output / 'states'
    results.mkdir(exist_ok=True)
    commands = []
    for rank, subset in enumerate(state_shards(states, gpus)):
        if not subset:
            continue
        shard = output / f'worker-{rank}-states.jsonl.gz'
        shard.write_bytes(gzip.compress((''.join(json.dumps(state) + '\n' for state in subset)).encode(), mtime=0))
        argv = [sys.executable, '-u', '-m', 'experiments.ps_concentration.rng_intervene',
            '--checkpoint', str(checkpoint), '--states', str(shard), '--reference', str(reference),
            '--source', str(source), '--output', str(results), '--summary-output',
            str(output / f'worker-{rank}-summary.json'), '--samples', '32', '--device', 'cuda:0']
        commands.append(dict(rank=rank, device=devices[rank], argv=argv,
            states=[state['state_id'] for state in subset]))
    write(output / 'execution.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        gpus=gpus, commands=commands, session_sha256=file_hash(output / 'session.json')))
    if invoke is not None:
        invoke(commands)
    else:
        children, handles = [], []
        try:
            for command in commands:
                handle = (output / f'worker-{command["rank"]}.log').open('a')
                handles.append(handle)
                child = subprocess.Popen(command['argv'], stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=command['device']), start_new_session=True)
                children.append(child)
            write(output / 'processes.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                coordinator=os.getpid(), children=[dict(pid=child.pid, **command)
                    for child, command in zip(children, commands, strict=True)]))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('Intervention worker failed; inspect its preserved log and resume')
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError('Intervention worker failed; inspect its preserved log and resume')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            for handle in handles:
                handle.close()
    summary = aggregate(states, results, checkpoint=checkpoint)
    write(output / 'summary.json', summary)
    save_table(summary, output)
    return summary


def finish_memory(run, finish, output, *, gpus):
    run, finish, output = map(Path, (run, finish, output))
    status = read(finish / 'status.json')
    if status['status'] != 'PERFORMANCE_COMPLETE_INTERVENTIONS_PENDING':
        raise ValueError('Finish deferred performance evaluation before memory interventions')
    session = read(finish / 'session.json')
    plan = read(session['plan'])
    if file_hash(session['plan']) != session['plan_sha256']:
        raise ValueError('Frozen performance validation plan changed')
    ready(run, plan)
    final = read(finish / 'final/plan.json')
    if final['selected']['update'] == 0:
        output.mkdir(parents=True, exist_ok=True)
        result = dict(status='NO_TRAINED_CHECKPOINT_SELECTED', causal_textual_memory_established=False)
        write(output / 'status.json', result)
        return result
    entry = next(item for item in final['commands'] if item['condition'] == 'selected-ps-intervention-collection')
    if not completed_condition(entry):
        raise ValueError('Collect every reserved intervention board before selecting states')
    checkpoint = final['selected']['checkpoint']
    hero = read(run / 'resolved_config.json')
    episodes = Path(entry['summary']).parent / 'episodes.jsonl'
    collection = output / 'collection'
    output.mkdir(parents=True, exist_ok=True)
    inputs = dict(episodes=str(episodes), episodes_sha256=file_hash(episodes),
        tokenizer=hero['model'], checkpoint=checkpoint, maximum=128, boundary='before_thinking',
        annotation_source_sha256=file_hash(Path(__file__).with_name('rng_intervention_states.py')))
    save_fixed(output / 'collection-inputs.json', inputs)
    if not (collection / 'collection-summary.json').exists():
        subprocess.run([sys.executable, '-m', 'experiments.ps_concentration.rng_intervention_states',
            '--episodes', str(episodes), '--tokenizer', hero['model'], '--output', str(collection),
            '--maximum', '128', '--boundary', 'before_thinking'], check=True)
    summary = execute(checkpoint, collection / 'states.jsonl.gz', hero['model'], hero['source'],
        output, gpus=gpus)
    manifest = read(final['report_manifest'])
    manifest['interventions'] = [dict(name='hero-before-thinking', training_seed=hero['seed'],
        summary=str(output / 'summary.json'))]
    write(output / 'report-manifest.json', manifest)
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='INTERVENTIONS_FINISHED',
        qualified_states=summary['qualified_states'], executed_states=summary['executed_states'],
        causal_textual_memory_established=summary['causal_textual_memory_established'],
        summary=str(output / 'summary.json'), table=str(output / 'intervention-table.csv'),
        report_manifest=str(output / 'report-manifest.json'),
        scope='All collected states counted; effective-BF16 cold branches and native streaming parity reported separately. '
            'Independent training replication remains incomplete. No allocation release or goal completion.')
    write(output / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--finish-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    with (Path(args.run_dir).parent / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(finish_memory(args.run_dir, args.finish_dir, args.output, gpus=args.gpus)))


if __name__ == '__main__':
    main()
