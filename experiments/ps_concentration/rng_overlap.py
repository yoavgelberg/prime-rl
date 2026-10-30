"""Native vLLM games and synchronous PS GRPO with backward during generation."""

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from prime_rl.orchestrator.ps_games import GameRolloutRunner, worker_rpc
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig

from .control import handoff_requested
from .overlap_config import BoardProtocol, RunConfig, board_protocol, cohorts, episode_metrics, evaluation_specs, permutation, protocol_lock, training_specs


FACTORY = 'experiments.ps_concentration.rng_vllm.GameCohort'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def engine_options(config, *, attention='ps', checkpoint=None, evaluation=False):
    stream = config.stream()
    maximum = config.evaluation_concurrency
    hf = json.loads((Path(config.model) / 'config.json').read_text())
    head = hf.get('head_dim', hf['hidden_size'] // hf['num_attention_heads'])
    bytes_per_token = hf['num_hidden_layers'] * hf['num_key_value_heads'] * head * 4
    if attention == 'full':
        if maximum > 4:
            raise ValueError('Full-context evaluation requires at most four live games per B200')
        stream = replace(stream, max_requests=1)
        cache = bytes_per_token * (math.ceil(config.transcript_cap / 128) + 1) * 128 * maximum
    else:
        cache = max(2 * 1024**3, bytes_per_token * (512 + config.window + 256) * maximum)
    return dict(model=checkpoint or config.model, dtype='bfloat16', skip_tokenizer_init=True,
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=config.gpus,
        data_parallel_size_local=config.gpus,
        worker_cls='prime_rl.inference.vllm.ps_streaming_worker.PSStreamingWorker',
        max_model_len=config.transcript_cap, max_num_seqs=maximum, max_num_batched_tokens=1024,
        block_size=128, enable_chunked_prefill=True, enable_prefix_caching=False,
        kv_cache_memory_bytes=cache,
        hf_overrides=dict(use_sliding_window=attention == 'ps',
                          sliding_window=config.window if attention == 'ps' else None),
        additional_config=dict(ps_streaming=asdict(stream)),
        attention_backend='CUSTOM' if attention == 'ps' else 'FLASH_ATTN',
        cudagraph_capture_sizes=[size for size in (1, 2, 4, 8, 16, 32, 64) if size <= maximum])


def make_engine(config, *, attention='ps', checkpoint=None, evaluation=False):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from prime_rl.inference.vllm.ps_fa4 import register

    register()
    for name in ('VLLM_ATTENTION_BACKEND', 'VLLM_FLASH_ATTN_VERSION'):
        os.environ.pop(name, None)
    options = engine_options(config, attention=attention, checkpoint=checkpoint, evaluation=evaluation)
    write_json(Path(config.output) / f'engine-{attention}-{os.environ.get("SLURM_JOB_ID", "local")}.json', options)
    return AsyncLLM.from_engine_args(AsyncEngineArgs(**options))


def worker_options(config, specs, output, *, attention='ps', greedy=False, board=None):
    board = board or BoardProtocol()
    return dict(model=config.model, source=config.source, output=str(output), specs=specs,
        attention=attention, window=config.window, transcript_cap=config.transcript_cap,
        max_responses=board.max_responses, rows=board.rows, cols=board.cols, num_cards=board.num_cards,
        temperature=config.temperature, greedy=greedy,
        context_exhaustion_reward=-1.)


def reusable_episode(folder, spec):
    path = folder / 'result.json'
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if any(row.get(key) != value for key, value in spec.items() if key != 'rank'):
        raise ValueError(f'Evaluation identity changed: {folder}')
    record = Path(row['token_record'])
    if hashlib.sha256(record.read_bytes()).hexdigest() != row['token_record_sha256']:
        raise ValueError(f'Token record checksum failed: {record}')
    return row


async def evaluate_games(config, runner, *, split, output, label, attention='ps', policy_version=0,
                         policy_identity=None, count=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    board = board_protocol(split)
    if split == 'larger' and attention != 'ps':
        raise ValueError('Size generalization uses the selected PS window and greedy decoding')
    identity = dict(policy_version=policy_version, policy_identity=policy_identity or config.revision,
                    attention=attention, window=config.window, greedy=True, **board.identity())
    write_json(output / 'evaluation-protocol.json', dict(config=config.model_dump(), **board.identity(),
        transcript_cap=config.transcript_cap, prompt='Identical native system prompt; board size appears in observations',
        scope='Parameter-scaled RNG-Bench Matching Pairs' if split == 'larger' else 'Native RNG-Bench Matching Pairs 4x4'))
    specs = [dict(spec, **identity) for spec in evaluation_specs(config, split, label, count=count)]
    write_json(output / 'plan.json', specs)
    rows, pending = [], []
    for spec in specs:
        folder = output / spec['episode_id']
        row = reusable_episode(folder, spec)
        if row is not None:
            rows.append(row)
        else:
            if folder.exists():
                folder.rename(folder.with_name(folder.name + f'.interrupted-{time.time_ns()}'))
            pending.append(spec)
    started = time.perf_counter()
    for batch in cohorts(pending, config.gpus, config.evaluation_concurrency):
        accounting = output / 'cohorts' / f'cohort-{time.time_ns()}.json'
        started_wall = time.time()
        write_json(accounting, dict(status='started', wall_start=started_wall, gpus=config.gpus,
            episode_ids=[spec['episode_id'] for spec in batch], job=os.environ.get('SLURM_JOB_ID')))
        report = await runner.collect(FACTORY, worker_options(config, batch, output,
            attention=attention, greedy=True, board=board), policy_version=policy_version)
        write_json(accounting, dict(status='complete', wall_start=started_wall, wall_end=time.time(),
            gpus=config.gpus, job=os.environ.get('SLURM_JOB_ID'), report=report))
        rows.extend(report['episodes'])
        write_json(output / 'progress.json', dict(completed=len(rows), total=len(specs), **identity))
        await runner.release()
    rows.sort(key=lambda row: row['board_seed'])
    if len(rows) != len(specs) or len({row['board_seed'] for row in rows}) != len(specs):
        raise ValueError('Evaluation must contain every reserved board exactly once')
    metrics = dict(episode_metrics(rows), **identity, split=split,
                   current_process_seconds=time.perf_counter() - started)
    accounting = [json.loads(path.read_text()) for path in (output / 'cohorts').glob('*.json')]
    complete = [row for row in accounting if row['status'] == 'complete']
    metrics.update(completed_cohort_inference_seconds=sum(row['report']['seconds'] for row in complete),
        completed_cohort_gpu_hours=sum(row['report']['seconds'] * row['gpus'] / 3600 for row in complete),
        unfinished_cohort_accounting=len(accounting) - len(complete),
        peak_kv_pool_allocated_bytes=max((device['kv_pool_allocated_bytes'] for row in complete
            for device in row['report'].get('devices', [])), default=0),
        inference_cost_scope='Complete recorded cohorts, excluding setup; a lower bound when interrupted cohorts lack a complete receipt. Slurm allocation cost reported separately.')
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_json(output / 'summary.json', metrics)
    return metrics


def recover_uncommitted(output, version):
    recovery = output / f'recovered-{time.time_ns()}'
    for name in ('rollouts', 'rollout-plans', 'parity', 'update-metrics', 'validation'):
        for path in (output / name).glob('update-*'):
            number = int(path.name.removeprefix('update-').split('.')[0])
            if number > version:
                target = recovery / name
                target.mkdir(parents=True, exist_ok=True)
                path.rename(target / path.name)
    for path in (output / 'checkpoints').glob('policy-*'):
        if int(path.name.removeprefix('policy-').split('.')[0]) > version:
            recovery.mkdir(parents=True, exist_ok=True)
            path.rename(recovery / path.name)
    accepted = [json.loads(path.read_text())['metrics']
                for path in sorted((output / 'update-metrics').glob('update-*.json'))]
    (output / 'updates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in accepted))


def aggregate_update(reports, materialized, kl_reports, steps, *, update, seconds, gpus):
    episodes = [row for report in reports for row in report['episodes']]
    if len(episodes) != 64 or len({row['episode_id'] for row in episodes}) != 64:
        raise ValueError('One fresh global update requires exactly 64 distinct trajectories')
    if any(step['policy_version'] != update for step in steps):
        raise ValueError('Data-parallel replicas did not commit the same update')
    metrics = episode_metrics(episodes)
    tokens = metrics['generated_tokens']
    advantages = [value for row in materialized for value in row['advantages']]
    mean_advantage = sum(advantages) / len(advantages)
    metrics.update(update=update, training_episodes=update * 64, seconds=seconds,
        generated_tokens_per_second=tokens / seconds,
        generated_tokens_per_second_per_gpu=tokens / (seconds * gpus),
        learning_rate=steps[0]['learning_rate'], gradient_norm=steps[0]['gradient_norm'],
        weight_update_norm=steps[0]['weight_update_norm'], clip_fraction_at_origin=0.,
        gradient_clipped=steps[0]['gradient_clipped'],
        normalized_score_surrogate_loss=sum(row['normalized_score_surrogate_loss'] for row in materialized),
        clipped_grpo_surrogate_at_origin=sum(row['clipped_grpo_surrogate_at_origin'] for row in materialized),
        zero_advantage_groups=sum(row['zero_advantage_groups'] for row in materialized),
        advantage_mean=mean_advantage,
        advantage_std=math.sqrt(sum((value - mean_advantage)**2 for value in advantages) / len(advantages)),
        normalized_kl=sum(row['normalized_kl'] for row in kl_reports),
        normalized_total_score_surrogate_loss=sum(row['normalized_score_surrogate_loss'] for row in materialized)
            + .01 * sum(row['normalized_kl'] for row in kl_reports),
        kl_per_generated_token=sum(row['kl_sum'] for row in kl_reports) / tokens,
        parity_mean_abs=sum(row['parity']['mean_abs'] * row['generated_tokens'] for row in kl_reports) / tokens,
        parity_max_rank_p99=max(row['parity']['p99'] for row in kl_reports),
        strict_parity_go=all(row['parity']['status'] == 'GO' for row in kl_reports))
    backward = [rank for report in reports for rank in report['backward']]
    metrics['peak_allocated_bytes'] = max(row.get('peak_allocated_bytes', 0) for row in backward)
    metrics['peak_reserved_bytes'] = max(row.get('peak_reserved_bytes', 0) for row in backward)
    metrics['peak_kv_pool_allocated_bytes'] = max((device['kv_pool_allocated_bytes'] for report in reports
        for device in report.get('devices', [])), default=0)
    return metrics


def setup_wandb(config):
    from prime_rl.utils.config import WandbConfig
    from prime_rl.utils.logger import setup_logger
    from prime_rl.utils.monitor import setup_monitor
    import wandb

    os.environ['WANDB_ENTITY'] = config.wandb_entity
    os.environ['WANDB_MODE'] = 'online'
    setup_logger('INFO')
    monitor = setup_monitor(wandb_config=WandbConfig(project=config.wandb_project,
        name=f'Qwen3-4B-Thinking PS GRPO seed {config.seed}', id=config.wandb_id, offline=False),
        output_dir=Path(config.output), run_config=config)
    wandb.define_metric('train/update')
    wandb.define_metric('train/*', step_metric='train/update')
    wandb.define_metric('validation/*', step_metric='train/update')
    wandb.define_metric('reference_full/*', step_metric='train/update')
    write_json(Path(config.output) / 'wandb.json', dict(url=wandb.run.url, id=wandb.run.id))
    return monitor


def log_metrics(monitor, prefix, metrics, version):
    values = {f'{prefix}/{name}': value for name, value in metrics.items()
              if isinstance(value, (int, float, bool))}
    values['train/update'] = version
    values['train/episodes'] = version * 64
    monitor.log(values)
    print(json.dumps(dict(event=prefix, **metrics)), flush=True)


def pilot_gate(full, initial, trained):
    values = dict(full_matched_fraction=full['matched_fraction'], full_minimum=.8,
        initial_matched_fraction_gap=full['matched_fraction'] - initial['matched_fraction'], gap_minimum=.10,
        ps_return_improvement=trained['mean_return'] - initial['mean_return'], improvement_minimum=.05,
        trained_format_error_rate=trained['format_error_rate'], format_error_maximum=.01)
    passed = (values['full_matched_fraction'] >= .8 and values['initial_matched_fraction_gap'] >= .10
              and values['ps_return_improvement'] >= .05 and values['trained_format_error_rate'] < .01)
    return dict(status='GO' if passed else 'HOLD', **values,
        full_rl_skipped=True, interpretation='Numerical gate report; expansion remains an agent review of the completed pilot')


async def train(config, engine, monitor, competence):
    if config.window != 1024:
        raise ValueError('The training window is frozen at 1024; window sweeps are evaluation only')
    output, runner = Path(config.output), GameRolloutRunner(engine)
    initialized = await worker_rpc(engine, 'ps_train_initialize', kwargs=dict(config=dict(
        optimizer=dict(betas=[.9, .95], epsilon=1e-8), outcome=asdict(OutcomeConfig()),
        kl_coefficient=config.kl_coefficient)))
    if len(initialized) != config.gpus or any(row['world_size'] != config.gpus for row in initialized):
        raise ValueError('The initialized engine differs from the reserved DP layout')
    identity = config.scientific_identity()
    version, best_return, best_update, cumulative_tokens = 0, None, None, 0
    latest = output / 'checkpoints/latest.pt'
    if latest.exists():
        restored = await worker_rpc(engine, 'ps_train_restore', kwargs=dict(path=str(latest), expected_metadata=identity))
        version = restored[0]['policy_version']
        if any(row['policy_version'] != version for row in restored):
            raise ValueError('Restored DP policy versions differ')
        metadata = restored[0]['metadata']
        best_return, best_update = metadata.get('best_return'), metadata.get('best_update')
        cumulative_tokens = metadata.get('cumulative_generated_tokens', 0)
    recover_uncommitted(output, version)
    boards = permutation(config.seed)
    npy = output / 'training-board-permutation.npy'
    if not npy.exists():
        import numpy as np
        np.save(npy, boards)
    write_json(output / 'seed-plan.json', dict(training_seed=config.seed,
        permutation_sha256=hashlib.sha256(boards.tobytes()).hexdigest(),
        rng_derivation='SHA256 rng-matching-training-v1:{training_seed}:{update_zero_based}:{board_index}:{rollout_index}; little endian first 8 bytes modulo 2**63-1'))

    async def save(*, inference=False):
        metadata = dict(identity, best_return=best_return, best_update=best_update,
            cumulative_generated_tokens=cumulative_tokens, source_sha256=config.source_sha256,
            full_context_competence=competence)
        inference_path = output / 'checkpoints' / f'policy-{version:04d}'
        if inference and inference_path.exists() and not (inference_path / 'experiment.json').exists():
            inference_path.rename(inference_path.with_name(inference_path.name + f'.interrupted-{time.time_ns()}'))
        await worker_rpc(engine, 'ps_train_save', timeout=1800., kwargs=dict(path=str(latest), metadata=metadata,
            inference_path=str(inference_path) if inference and not inference_path.exists() else None,
            tokenizer_source=config.model))
        write_json(output / 'state.json', dict(policy_version=version, **metadata))

    async def validate():
        nonlocal best_return, best_update
        await save(inference=True)
        metrics = await evaluate_games(config, runner, split='validation',
            output=output / 'validation' / f'update-{version:04d}', label=f'val-u{version:04d}',
            policy_version=version, policy_identity=f'{config.wandb_id}:update-{version}')
        if version == 0:
            write_json(output / 'initial-comparison.json', dict(full_reference=competence, ps_reference=metrics,
                matched_fraction_gap=competence['matched_fraction'] - metrics['matched_fraction'],
                efficiency_return_gap=competence['mean_return'] - metrics['mean_return'],
                paired_board_split=competence.get('split') == 'validation' and competence.get('episodes') == 128))
        if best_return is None or metrics['mean_return'] > best_return:
            best_return, best_update = metrics['mean_return'], version
        log_metrics(monitor, 'validation', metrics, version)
        await save()
        write_json(output / 'selected-checkpoint.json', dict(update=best_update, validation_return=best_return,
            checkpoint=str(output / 'checkpoints' / f'policy-{best_update:04d}'),
            selection='Highest complete 128-board PS validation native efficiency return, earlier checkpoint on ties'))

    if version % config.validation_interval == 0:
        await validate()
    while version < config.updates:
        previous_metrics = output / 'update-metrics' / f'update-{version:04d}.json'
        previous_seconds = json.loads(previous_metrics.read_text())['metrics']['seconds'] if previous_metrics.exists() else 0.
        remaining = float(os.environ.get('PS_CONCENTRATION_END_EPOCH', 'inf')) - time.time()
        if handoff_requested() or remaining < max(1800, previous_seconds * 1.3 + 300):
            await save()
            write_json(output / 'handoff-ready.json', dict(policy_version=version, job=os.environ.get('SLURM_JOB_ID')))
            return
        update = version + 1
        started = time.perf_counter()
        specs = training_specs(config, update, boards)
        specs = [dict(spec, policy_version=version, policy_identity=f'{config.wandb_id}:update-{version}') for spec in specs]
        write_json(output / 'rollout-plans' / f'update-{update:04d}.json', specs)
        reports, reductions = [], []
        for batch in cohorts(specs, config.gpus, config.concurrency):
            report = await runner.collect(FACTORY, worker_options(config, batch,
                output / 'rollouts' / f'update-{update:04d}'), backward=True, policy_version=version)
            reports.append(report)
            reductions.append(await worker_rpc(engine, 'ps_train_reduce', timeout=1800.))
            await runner.release()
        materialized = await worker_rpc(engine, 'ps_train_materialize', timeout=1800.)
        kl_reports = await worker_rpc(engine, 'ps_train_reference_kl', timeout=43200., kwargs=dict(
            backend='experiments.ps_concentration.rng_overlap_loss.accumulate_kl', options=dict(
                reference_checkpoint=config.model, output=str(output / 'parity' / f'update-{update:04d}'),
                temperature=config.temperature, allow_numerical_target_miss=config.allow_numerical_target_miss)))
        steps = await worker_rpc(engine, 'ps_train_step', timeout=1800.)
        version = update
        metrics = aggregate_update(reports, materialized, kl_reports, steps, update=update,
            seconds=time.perf_counter() - started, gpus=config.gpus)
        cumulative_tokens += metrics['generated_tokens']
        metrics['cumulative_generated_tokens'] = cumulative_tokens
        metrics['current_layout_compute_gpu_hours'] = config.gpus * metrics['seconds'] / 3600
        write_json(output / 'update-metrics' / f'update-{update:04d}.json',
            dict(metrics=metrics, rollouts=reports, reductions=reductions, materialized=materialized,
                 reference_kl=kl_reports, optimizer=steps))
        await save()
        append_json(output / 'updates.jsonl', metrics)
        log_metrics(monitor, 'train', metrics, version)
        if version % config.validation_interval == 0 or version == config.updates:
            await validate()
    gate = None
    if version >= 32:
        initial = json.loads((output / 'validation/update-0000/summary.json').read_text())
        trained = json.loads((output / 'validation' / f'update-{version:04d}/summary.json').read_text())
        gate = pilot_gate(competence, initial, trained)
    write_json(output / 'pilot-complete.json', dict(updates=version, episodes=version * 64,
        best_update=best_update, best_return=best_return, automatic_matrix_expansion=False, pilot_gate=gate))


def lock_run(config):
    import yaml
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = protocol_lock(config)
    write_json(output / f'protocol-lock-{os.environ.get("SLURM_JOB_ID", "local")}.json', lock)
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config.model_dump(), sort_keys=True))
    packages = sorted((row.metadata['Name'], row.version) for row in importlib.metadata.distributions()
                      if row.metadata.get('Name'))
    write_json(output / f'runtime-dependencies-{os.environ.get("SLURM_JOB_ID", "local")}.json', packages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--attention', choices=['ps', 'full'], default='ps')
    parser.add_argument('--split', choices=['screening', 'validation', 'test', 'larger', 'intervention'], default='screening')
    parser.add_argument('--checkpoint')
    parser.add_argument('--competence')
    parser.add_argument('--window', type=int, help='Evaluation window sweep; training remains W=1024')
    parser.add_argument('--count', type=int, help='Evaluate the first N fixed split boards, e.g. 128 for window sweeps')
    args = parser.parse_args()
    config = RunConfig(**json.loads(Path(args.config).read_text()))
    if args.window is not None:
        config = replace(config, window=args.window)
    if args.mode == 'train' and args.count is not None:
        parser.error('--count is evaluation-only; training uses the complete fixed global batch')
    lock_run(config)
    competence = None
    if args.mode == 'train':
        if not args.competence:
            parser.error('Training requires a completed same-engine full-context competence screening')
        competence = json.loads(Path(args.competence).read_text())
        if (competence.get('attention') != 'full'
                or (competence.get('episodes'), competence.get('split')) not in ((64, 'screening'), (128, 'validation'))
                or not competence.get('greedy')
                or competence.get('policy_version') != 0 or competence.get('policy_identity') != config.revision):
            raise ValueError('The competence reference must be a complete greedy full-context screening or validation split')
        if competence['matched_fraction'] < .8:
            raise ValueError(f'HOLD: full-context matched fraction {competence["matched_fraction"]:.6f} < .8')
    monitor = setup_wandb(config) if args.mode == 'train' else None
    if monitor is not None:
        log_metrics(monitor, 'reference_full', competence, 0)
    engine = make_engine(config, attention=args.attention, checkpoint=args.checkpoint,
                         evaluation=args.mode == 'evaluate')
    try:
        if args.mode == 'train':
            if args.attention != 'ps' or args.checkpoint:
                raise ValueError('The accepted pilot trains PS only, restoring its exact optimizer snapshot automatically')
            asyncio.run(train(config, engine, monitor, competence))
        else:
            identity = config.revision
            version = 0
            if args.checkpoint:
                receipt = json.loads((Path(args.checkpoint) / 'experiment.json').read_text())
                version, identity = receipt['policy_version'], str(Path(args.checkpoint).resolve())
            metrics = asyncio.run(evaluate_games(config, GameRolloutRunner(engine), split=args.split,
                output=Path(config.output) / 'evaluation' / args.split, label=f'{args.attention}-u{version}',
                attention=args.attention, policy_version=version, policy_identity=identity, count=args.count))
            metrics['status'] = 'GO' if metrics['matched_fraction'] >= .8 else 'HOLD'
            metrics['competence_threshold'] = .8
            write_json(Path(config.output) / 'evaluation' / args.split / 'summary.json', metrics)
            print(json.dumps(metrics), flush=True)
    finally:
        engine.shutdown()
        if monitor is not None:
            monitor.close()


if __name__ == '__main__':
    main()
