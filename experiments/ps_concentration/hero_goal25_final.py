"""Freeze deferred tests at the checkpoint selected by the training-only turn target."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path

from .hero_exports import file_hash
from .hero_finish import ready
from .hero_memory import save_fixed
from .hero_post320_final import prepare
from .hero_v2 import HeroConfig, identity, read


def freeze(run, output, *, gpus, source_root, source_sha256):
    run, output = Path(run), Path(output)
    config = HeroConfig(**read(run / 'resolved_config.json'))
    version = read(run / 'state.json')['policy_version']
    ready(run, dict(training_barrier=version, training_identity=identity(config)))
    stop = read(run / 'stop-after-update.json')
    evidence_path = run / 'goal25-target-evidence.json'
    evidence = read(evidence_path)
    criterion = read(run / 'goal25-plan-a1.json')['criterion']
    if (stop['identity'] != identity(config) or stop['after_update'] != version
            or evidence['status'] != 'TARGET_MET' or evidence['criterion'] != criterion
            or version not in (evidence['last_update'], evidence['last_update'] + 1)):
        raise ValueError('Freeze only the checkpoint chosen by the declared training-only turn target')
    first, last = evidence['first_update'], evidence['last_update']
    if last - first + 1 != criterion['consecutive_complete_batches']:
        raise ValueError('Require every consecutive batch in the stopping window')
    points = []
    for update in range(first, last + 1):
        path = run / f'update-metrics/update-{update:04d}.json'
        if evidence['input_sha256'].get(str(path)) != file_hash(path):
            raise ValueError('Stopping evidence differs from the committed training batches')
        point = read(path)['metrics']
        if point['update'] != update or point['episodes'] != 128:
            raise ValueError('Require all128trajectories per stopping batch')
        points.append(point)
    responses = sum(p['mean_attempts'] for p in points) / len(points)
    pairs = sum(p['mean_pairs'] for p in points) / len(points)
    completion = sum(p['completed_only_denominator'] for p in points) / (128 * len(points))
    resources = sum(p['resource_terminations'] for p in points)
    if (responses > criterion['mean_responses_at_most'] or pairs < criterion['mean_pairs_at_least']
            or completion < criterion['completion_rate_at_least'] or resources
            or not all(math.isclose(actual, evidence[key], rel_tol=1e-10, abs_tol=1e-10)
                for actual, key in [(responses, 'mean_responses'), (pairs, 'mean_pairs'),
                    (completion, 'completion_rate')])):
        raise ValueError('The saved stopping window has not met the declared performance target')
    initial_path = run / 'plan.json'
    initial = read(initial_path)
    parent = HeroConfig(**read(Path(initial['parent_run']) / 'resolved_config.json'))
    snapshots = {}
    for label, value in [('continued', config), ('parent320', parent)]:
        path = output / f'frozen-{label}-config.json'
        save_fixed(path, asdict(value))
        snapshots[label] = dict(path=str(path), sha256=file_hash(path))
    save_fixed(output / 'training-stop-evidence.json', evidence)
    plan = dict(format='hero-w1024-goal25-frozen-v1', parent_policy_version=320,
        parent_run=initial['parent_run'], output=str(run), final_update=version,
        configured_horizon=config.updates, identity=identity(config),
        fresh_future_test_reserved=initial['fresh_future_test_reserved'],
        configuration_snapshots=snapshots, initial_plan=str(initial_path),
        initial_plan_sha256=file_hash(initial_path), stopping_evidence_sha256=file_hash(evidence_path),
        checkpoint_rule='Training-only target; freeze before inspecting fresh test boards')
    path = output / 'frozen-training-plan.json'
    save_fixed(path, plan)
    return prepare(path, output, gpus=gpus, source_root=source_root, source_sha256=source_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    freeze(args.run_dir, args.output, gpus=args.gpus,
        source_root=args.source_root, source_sha256=args.source_sha256)


if __name__ == '__main__':
    main()
