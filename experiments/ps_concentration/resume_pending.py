"""Commit one saved native task-gradient batch without sampling new trajectories."""

import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

from prime_rl.orchestrator.ps_games import worker_rpc
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig
from .overlap_config import RunConfig, permutation, training_specs
from .rng_overlap import aggregate_update, lock_run, log_metrics, make_engine, setup_wandb, write_json


def validate_manifest(config, path):
    path = Path(path)
    pending = json.loads(path.read_text())
    identity = dict(config.scientific_identity(), evaluation_deferred=True)
    if pending['identity'] != identity:
        raise ValueError('Pending batch scientific identity differs from this run')
    if len(pending['files']) != config.gpus or len(pending['materialized']) != config.gpus:
        raise ValueError('Resume a pending batch with its original GPU replica count')
    for rank, receipt in enumerate(pending['files']):
        expected = path.with_suffix('') / f'rank-{rank}.pt'
        if Path(receipt['path']).resolve() != expected.resolve() or expected.stat().st_size != receipt['bytes']:
            raise ValueError('Pending gradient file identity or byte count differs')
        if receipt['policy_version'] != pending['policy_version']:
            raise ValueError('Pending rank policy versions differ')
    episodes = [row for report in pending['reports'] for row in report['episodes']]
    if len(episodes) != 64 or len({row['episode_id'] for row in episodes}) != 64:
        raise ValueError('A pending update needs all64 distinct original trajectories')
    if any(row['policy_version'] != pending['policy_version'] for row in episodes):
        raise ValueError('Pending trajectories have mixed policy versions')
    expected = {row['episode_id']: row for row in training_specs(
        config, pending['policy_version'] + 1, permutation(config.seed))}
    keys = ('rank', 'board_seed', 'board_index', 'rollout_index', 'generation_seed')
    if any(row['episode_id'] not in expected or any(row[key] != expected[row['episode_id']][key] for key in keys)
           for row in episodes):
        raise ValueError('Pending batch board seeds or generation RNGs differ from the fixed plan')
    return pending, identity


async def resume(config, engine, monitor, path):
    path = Path(path)
    pending, identity = validate_manifest(config, path)
    update = pending['policy_version'] + 1
    plan = Path(config.output) / 'rollout-plans' / f'update-{update:04d}.json'
    collection_seconds = pending.get('collection_and_materialize_seconds')
    if collection_seconds is None:
        collection_seconds = path.stat().st_mtime - plan.stat().st_mtime
    if collection_seconds < 0:
        raise ValueError('Saved collection timing is invalid')
    started = time.perf_counter()
    initialized = await worker_rpc(engine, 'ps_train_initialize', kwargs=dict(config=dict(
        optimizer=dict(betas=[.9, .95], epsilon=1e-8), outcome=asdict(OutcomeConfig()), kl_coefficient=config.kl_coefficient)))
    if len(initialized) != config.gpus or any(row['world_size'] != config.gpus for row in initialized):
        raise ValueError('Initialized replica count differs from pending task gradients')
    output, metadata, version = Path(config.output), {}, 0
    latest = output / 'checkpoints/latest.pt'
    if latest.exists():
        restored = await worker_rpc(engine, 'ps_train_restore', kwargs=dict(path=str(latest), expected_metadata=identity))
        if len({row['policy_version'] for row in restored}) != 1:
            raise ValueError('Optimizer replicas restored different versions')
        version, metadata = restored[0]['policy_version'], restored[0]['metadata']
    if version != pending['policy_version']:
        raise ValueError('Pending batch is stale or its prior optimizer checkpoint is missing')
    update = version + 1
    restored = await worker_rpc(engine, 'ps_train_restore_pending', timeout=1800.,
        kwargs=dict(path=str(path.with_suffix('')), expected_metadata=identity))
    if sum(row['episodes'] for row in restored) != 64 or any(row['policy_version'] != version for row in restored):
        raise ValueError('Restored gradients do not cover the original fresh batch')
    parity = output / 'parity' / f'update-{update:04d}'
    if parity.exists():
        parity.rename(parity.with_name(parity.name + f'.before-pending-resume-{time.time_ns()}'))
    kl = await worker_rpc(engine, 'ps_train_reference_kl', timeout=43200., kwargs=dict(
        backend='experiments.ps_concentration.rng_overlap_loss.accumulate_kl', options=dict(
            reference_checkpoint=config.model, output=str(parity), temperature=config.temperature,
            allow_numerical_target_miss=config.allow_numerical_target_miss)))
    steps = await worker_rpc(engine, 'ps_train_step', timeout=1800.)
    metrics = aggregate_update(pending['reports'], pending['materialized'], kl, steps,
        update=update, seconds=collection_seconds + time.perf_counter() - started, gpus=config.gpus)
    metrics.update(cumulative_generated_tokens=metadata.get('cumulative_generated_tokens', 0) + metrics['generated_tokens'],
        current_layout_compute_gpu_hours=config.gpus * metrics['seconds'] / 3600,
        resumed_pending_task_gradient=True)
    receipt = dict(manifest=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_collection_policy_version=version, optimizer_update=update, new_trajectories_sampled=0,
        collection_seconds=collection_seconds,
        timing_scope='Saved collection/materialization plus this resume; excludes the failed reference-check attempt and allocation idle time, which require separate allocation accounting')
    write_json(output / 'update-metrics' / f'update-{update:04d}.json', dict(metrics=metrics,
        rollouts=pending['reports'], reductions=pending['reductions'], materialized=pending['materialized'],
        reference_kl=kl, optimizer=steps, pending_resume=receipt))
    metadata.update(identity, source_sha256=config.source_sha256,
        cumulative_generated_tokens=metrics['cumulative_generated_tokens'], pending_resume=receipt)
    await worker_rpc(engine, 'ps_train_save', timeout=1800., kwargs=dict(path=str(latest), metadata=metadata))
    write_json(output / 'state.json', dict(policy_version=update, **metadata))
    accepted = [json.loads(p.read_text())['metrics'] for p in sorted((output / 'update-metrics').glob('update-*.json'))]
    (output / 'updates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in accepted))
    log_metrics(monitor, 'train', metrics, update)
    write_json(output / f'pending-resume-{update:04d}.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pending', required=True)
    args = parser.parse_args()
    config = RunConfig(**json.loads(Path(args.config).read_text()))
    validate_manifest(config, args.pending)
    lock = Path(config.output).parent / 'training.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_run(config)
        monitor = setup_wandb(config)
        engine = make_engine(config, attention='ps')
        try:
            print(json.dumps(asyncio.run(resume(config, engine, monitor, args.pending))), flush=True)
        finally:
            engine.shutdown()
            monitor.close()


if __name__ == '__main__':
    main()
