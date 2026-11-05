"""Freeze reserved evaluations after an explicit user-selected RL budget."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .hero_exports import file_hash
from .hero_finish import ready
from .hero_memory import save_fixed
from .hero_post320_final import prepare
from .hero_v2 import HeroConfig, identity, read


def freeze(run, output, *, updates, gpus, source_root, source_sha256):
    run, output = Path(run), Path(output)
    config = HeroConfig(**read(run / 'resolved_config.json'))
    version = read(run / 'state.json')['policy_version']
    stop_path = run / 'stop-after-update.json'
    stop = read(stop_path)
    if (type(updates) is not int or updates < 448 or version != updates
            or stop.get('requested_by_user') is not True or stop['after_update'] != updates
            or stop['identity'] != identity(config)):
        raise ValueError('Finish the explicit user-requested training budget before freezing')
    ready(run, dict(training_barrier=updates, training_identity=identity(config)))
    complete = read(run / 'training-complete.json')
    if complete['target'] != updates:
        raise ValueError('The completed budget differs from the user-requested stop')
    criterion = read(run / 'goal25-plan-a1.json')['criterion']
    first = updates - criterion['consecutive_complete_batches'] + 1
    points, sources = [], {}
    for update in range(first, updates + 1):
        path = run / f'update-metrics/update-{update:04d}.json'
        point = read(path)['metrics']
        if point['update'] != update or point['episodes'] != 128:
            raise ValueError('Require every complete128-game batch in the final stopping window')
        points.append(point)
        sources[str(path)] = file_hash(path)
    count = len(points)
    responses = sum(p['mean_attempts'] for p in points) / count
    pairs = sum(p['mean_pairs'] for p in points) / count
    completion = sum(p['completed_only_denominator'] for p in points) / (128 * count)
    resources = sum(p['resource_terminations'] for p in points)
    met = (responses <= criterion['mean_responses_at_most']
        and pairs >= criterion['mean_pairs_at_least']
        and completion >= criterion['completion_rate_at_least'] and resources == 0)
    evidence = dict(reason='explicit_user_budget', requested_update=updates,
        request=str(stop_path), request_sha256=file_hash(stop_path),
        performance_target_met=met, criterion=criterion, first_update=first, last_update=updates,
        episodes=128 * count, mean_responses=responses, mean_pairs=pairs,
        completion_rate=completion, resource_terminations=resources, input_sha256=sources,
        scope='The user budget selects the checkpoint; performance-target attainment is measured separately.')
    evidence_path = output / 'training-stop-evidence.json'
    initial_path = run / 'plan.json'
    initial = read(initial_path)
    parent = HeroConfig(**read(Path(initial['parent_run']) / 'resolved_config.json'))
    snapshots = {}
    for label, value in [('continued', config), ('parent320', parent)]:
        path = output / f'frozen-{label}-config.json'
        save_fixed(path, asdict(value))
        snapshots[label] = dict(path=str(path), sha256=file_hash(path))
    save_fixed(evidence_path, evidence)
    plan = dict(format='hero-w1024-user-budget-frozen-v1', parent_policy_version=320,
        parent_run=initial['parent_run'], output=str(run), final_update=updates,
        configured_horizon=config.updates, identity=identity(config),
        fresh_future_test_reserved=initial['fresh_future_test_reserved'],
        configuration_snapshots=snapshots, initial_plan=str(initial_path),
        initial_plan_sha256=file_hash(initial_path), stopping_evidence=str(evidence_path),
        stopping_evidence_sha256=file_hash(evidence_path),
        checkpoint_rule=f'Explicit user budget update{updates}; no test-based checkpoint or window selection')
    path = output / 'frozen-training-plan.json'
    save_fixed(path, plan)
    return prepare(path, output, gpus=gpus, source_root=source_root, source_sha256=source_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    freeze(args.run_dir, args.output, updates=args.updates, gpus=args.gpus,
        source_root=args.source_root, source_sha256=args.source_sha256)


if __name__ == '__main__':
    main()
