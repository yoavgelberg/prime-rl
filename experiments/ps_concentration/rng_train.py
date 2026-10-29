"""Synchronous all-token outcome RL on the pinned native matching game."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .activation_offload import activation_offload
from .check_policy import accept_parity, teacher_logps
from .checkpoints import restore_adamw, retain_selected_and_latest, save_checkpoint
from .control import handoff_requested
from .objective import leave_one_out, sampled_kl, trajectory_loss
from .policy import load_model
from .rng_collect import NativeCollector
from .rng_training_config import MatchingConfig, training_rng


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gather(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def board_requests(permutation, config, update, rank, world):
    assert 8 % world == 0
    groups = range(rank * (8 // world), (rank + 1) * (8 // world))
    return [dict(board_seed=int(permutation[(update - 1) * 8 + group]), board_index=group,
                 rollout_index=rollout, generation_seed=training_rng(config.seed, update - 1, group, rollout),
                 episode_id=f'u{update:04d}-g{group}-r{rollout}')
            for group in groups for rollout in range(8)]


def record_metrics(records):
    attempts = sum(row['response_count'] for row in records)
    return dict(episodes=len(records), return_sum=sum(row['training_return'] for row in records),
        pairs_sum=sum(row['score'] for row in records), attempts_sum=attempts,
        completed=sum(bool(row['done']) for row in records),
        context_terminations=sum(bool(row.get('censored')) for row in records),
        invalid_actions=sum(row['invalid_actions'] for row in records),
        format_errors=sum(row['format_errors'] for row in records),
        generated_tokens=sum(row['generated_tokens'] for row in records),
        raw_transcript_tokens=sum(row['raw_transcript_tokens'] for row in records))


def combine_metrics(workers):
    result = {key: sum(row[key] for row in workers) for key in workers[0]}
    count = result['episodes']
    result.update(mean_return=result['return_sum'] / count, mean_pairs=result['pairs_sum'] / count,
        matched_fraction=result['pairs_sum'] / (8 * count),
        completion_rate=result['completed'] / count, mean_attempts=result['attempts_sum'] / count,
        format_error_rate=result['format_errors'] / max(result['attempts_sum'], 1))
    return result


def collect_batches(collector, specs, output, batch_size, *, greedy=False):
    records = []
    with torch.cuda.device(collector.policy.device):
        if collector.config.attention == 'full':
            free, _ = torch.cuda.mem_get_info()
            base = collector.policy
            kv = sum(4 * layer.self_attn.k_proj.weight.shape[0] * collector.config.transcript_cap
                     for layer in base.model.layers)
            casts = sum(p.numel() * 2 for p in base.parameters())
            admitted = (free - casts - 4 * 1024**3) // kv
            if admitted < 1:
                raise RuntimeError(f'No full-attention rollout slot: free={free}, kv={kv}, casts={casts}')
            batch_size = min(batch_size, int(admitted))
        for start in range(0, len(specs), batch_size):
            records.extend(collector.collect(specs[start:start + batch_size], output, greedy=greedy))
    return records


def validate(collector, output, config, update, rank, world):
    specs = [dict(board_seed=1100000 + index, board_index=index, rollout_index=0,
                  generation_seed=training_rng(0, 0, index, 0), episode_id=f'v{update:04d}-b{index}')
             for index in range(config.validation_boards) if index % world == rank]
    local = record_metrics(collect_batches(collector, specs, output / f'validation/update-{update:04d}',
                                           config.validation_batch_size, greedy=config.validation_greedy))
    result = combine_metrics(gather(local, world))
    result.update(update=update, training_episodes=update * 64, greedy=config.validation_greedy,
                  selection='Highest validation efficiency return under training attention; earlier wins ties')
    return result


def record_locks(output, model, source):
    source = Path(source)
    identity = json.loads((source / 'SOURCE.json').read_text())
    assert identity['commit'] == 'e3d195e857727dfa3d4a96561be202098d67da83'
    for row in identity['manifest']:
        assert hashlib.sha256((source / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    atomic_json(output / 'native-source-lock.json', identity)
    model = Path(model)
    names = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    atomic_json(output / 'model-tokenizer-lock.json', dict(model=str(model), revision=model.name,
        files={name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in names if (model / name).exists()}))
    distributions = sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata['Name']})
    (output / 'resolved-dependencies.txt').write_text('\n'.join(distributions) + '\n')


def save_resume(output, policy, optimizer, config, update, best_return, best_update, rank, world, devices):
    rng = dict(torch=torch.get_rng_state(), cuda={str(device): torch.cuda.get_rng_state(device).cpu() for device in devices})
    rngs = gather(rng, world)
    if rank == 0:
        temporary = output / 'latest_optimizer.pt.tmp'
        torch.save(dict(masters=[p.detach() for p in policy.parameters()], optimizer=optimizer.state_dict(),
            update=update, best_return=best_return, best_update=best_update, rngs=rngs,
            config=asdict(config), reference=config.model,
            scheduler=dict(kind='linear_warmup_then_constant', last_update=update, warmup=config.warmup)), temporary)
        temporary.replace(output / 'latest_optimizer.pt')
        atomic_json(output / 'latest_optimizer.json', dict(update=update, best_update=best_update))
    if world > 1:
        dist.barrier()


def restore_resume(path, policy, optimizer, config, rank, devices):
    state = torch.load(path, map_location=policy.device, weights_only=False)
    fixed = ('model', 'revision', 'attention', 'window', 'transcript_cap', 'max_responses', 'max_new_tokens',
             'seed', 'boards', 'rollouts', 'temperature', 'top_p', 'top_k', 'learning_rate', 'beta1', 'beta2',
             'epsilon', 'weight_decay', 'warmup', 'kl_coefficient', 'denominator', 'max_grad_norm',
             'context_exhaustion_reward', 'validation_greedy', 'validation_boards')
    assert state['reference'] == config.model
    assert all(state['config'][key] == getattr(config, key) for key in fixed), 'Resume changed the experiment'
    with torch.no_grad():
        for param, master in zip(policy.parameters(), state['masters'], strict=True):
            param.copy_(master)
    restore_adamw(optimizer, state['optimizer'])
    saved = state['rngs'][rank % len(state['rngs'])]
    torch.set_rng_state(saved['torch'].cpu())
    for device, value in zip(devices, saved['cuda'].values()):
        torch.cuda.set_rng_state(value.cpu(), device)
    return state['update'], state['best_return'], state['best_update']


def reconcile_resume(output, completed_update):
    recovery = output / f'recovered-{time.time_ns()}'
    for folder in ('rollouts', 'validation', 'update-metrics', 'validation-metrics', 'parity', 'rollout-plans'):
        for path in (output / folder).glob('update-*'):
            number = path.name.removeprefix('update-').split('-rank')[0].split('.')[0]
            if int(number) > completed_update:
                destination = recovery / folder
                destination.mkdir(parents=True, exist_ok=True)
                path.rename(destination / path.name)
    for path in output.glob('checkpoint-*'):
        if int(path.name.removeprefix('checkpoint-')) > completed_update:
            recovery.mkdir(parents=True, exist_ok=True)
            path.rename(recovery / path.name)
    rows = [json.loads(path.read_text()) for path in sorted((output / 'update-metrics').glob('update-*.json'))]
    (output / 'updates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def should_yield(update_seconds, rank, world, device):
    remaining = float(os.environ.get('PS_CONCENTRATION_END_EPOCH', 'inf')) - time.time()
    stop = handoff_requested() or remaining < max(900, update_seconds * 1.3 + 180)
    value = torch.tensor(int(stop) if rank == 0 else 0, device=device)
    if world > 1:
        dist.broadcast(value, src=0)
    return bool(value)


def train(args):
    from transformers import AutoTokenizer
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    assert world in (1, 2, 4, 8)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=4))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = MatchingConfig(model=args.model, source=args.source, attention=args.attention,
        seed=args.seed, updates=args.updates, window=args.window,
        context_exhaustion_reward=args.context_exhaustion_reward,
        rollout_batch_size=args.batch_size, validation_batch_size=args.validation_batch_size,
        separate_actors=args.separate_actors, training_workers=world,
        activation_cpu_offload=not args.no_activation_cpu_offload,
        allow_numerical_target_miss=args.allow_numerical_target_miss,
        source_sha256=os.environ.get('PS_CONCENTRATION_SOURCE_SHA', ''))
    assert config.context_exhaustion_reward is not None, 'Explicit resource-termination reward required for this pilot'
    assert Path(args.model).name == config.revision, 'Use the pinned starting snapshot as the reference'
    if rank == 0:
        config.save(output / 'resolved_config.yaml')
        record_locks(output, args.model, args.source)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = load_model(args.model, f'cuda:{rank}', train=True)
    reference = load_model(args.model, f'cuda:{rank}')
    actor_device = rank + world if args.separate_actors else rank
    if args.separate_actors:
        assert torch.cuda.device_count() >= 2 * world
    actor = policy if actor_device == rank else load_model(args.model, f'cuda:{actor_device}')
    collector = NativeCollector(actor, tokenizer, config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2),
                                 eps=config.epsilon, weight_decay=0, foreach=True)
    devices = list(dict.fromkeys((rank, actor_device)))
    permutation = np.random.default_rng(config.seed).permutation(1000000)
    if rank == 0:
        np.save(output / 'training_board_permutation.npy', permutation)
        (output / 'system-prompt.txt').write_text(collector.system_prompt)
    update, best_return, best_update = 0, -float('inf'), 0
    if args.resume:
        update, best_return, best_update = restore_resume(args.resume, policy, optimizer, config, rank, devices)
        if rank == 0:
            reconcile_resume(output, update)
    elif (output / 'latest_optimizer.pt').exists():
        raise RuntimeError('Existing optimizer state requires explicit --resume')
    elif rank == 0 and (output / 'checkpoint-0000').exists():
        (output / 'checkpoint-0000').rename(output / f'orphaned-{time.time_ns()}-checkpoint-0000')
    if world > 1:
        dist.barrier()
    if actor is not policy:
        with torch.no_grad():
            for target, source in zip(actor.parameters(), policy.parameters(), strict=True):
                target.copy_(source)
    if not args.resume:
        print(json.dumps(dict(event='initial_validation_start', rank=rank, greedy=config.validation_greedy,
                              boards=config.validation_boards)), flush=True)
        initial = validate(collector, output, config, 0, rank, world)
        best_return = initial['mean_return']
        if rank == 0:
            save_checkpoint(output / 'checkpoint-0000', policy, tokenizer,
                            dict(**initial, config=asdict(config), reference=args.model))
            atomic_json(output / 'validation-metrics/update-0000.json', initial)
        save_resume(output, policy, optimizer, config, 0, best_return, 0, rank, world, devices)
    if rank == 0:
        atomic_json(output / 'selected.json', dict(update=best_update, mean_return=best_return,
                    checkpoint=str(output / f'checkpoint-{best_update:04d}')))
    last_seconds = args.initial_update_seconds
    run_updates(policy, reference, actor, collector, optimizer, tokenizer, permutation,
                output, config, update, best_return, best_update, rank, world, devices, last_seconds)
    if world > 1:
        dist.destroy_process_group()


def run_updates(policy, reference, actor, collector, optimizer, tokenizer, permutation,
                output, config, completed_update, best_return, best_update, rank, world, devices, last_seconds):
    for update in range(completed_update + 1, config.updates + 1):
        if should_yield(last_seconds, rank, world, policy.device):
            save_resume(output, policy, optimizer, config, update - 1, best_return, best_update, rank, world, devices)
            if rank == 0:
                atomic_json(output / 'allocation-yield.json', dict(update=update - 1,
                    job=os.environ.get('SLURM_JOB_ID'), reason='Preserve completed state before allocation deadline'))
            return
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if actor is not policy:
            with torch.no_grad():
                for target, source in zip(actor.parameters(), policy.parameters(), strict=True):
                    target.copy_(source)
        specs = board_requests(permutation, config, update, rank, world)
        atomic_json(output / f'rollout-plans/update-{update:04d}-rank-{rank}.json', specs)
        print(json.dumps(dict(event='rollout_start', rank=rank, update=update, episodes=len(specs))), flush=True)
        rollout_started = time.perf_counter()
        records = collect_batches(collector, specs, output / f'rollouts/update-{update:04d}', config.rollout_batch_size)
        rollout_seconds = time.perf_counter() - rollout_started
        returns = np.array([row['training_return'] for row in records]).reshape(-1, 8)
        advantages = leave_one_out(returns).ravel()
        training_started = time.perf_counter()
        task_loss, kl_sum, tokens, differences = 0., 0., 0, []
        for index, (record, advantage) in enumerate(zip(records, advantages, strict=True)):
            with torch.no_grad():
                ref = teacher_logps(reference, record, temperature=config.temperature)
            context = activation_offload(policy.parameters()) if config.activation_cpu_offload else nullcontext()
            with context:
                logps = teacher_logps(policy, record, grad=True, temperature=config.temperature)
                loss = trajectory_loss(logps, ref, advantage, config)
            old = torch.tensor([value for value in record['old_tempered_logps'] if value is not None], device=policy.device)
            assert len(logps) == sum(record['loss_mask']) == len(old)
            differences.append((logps.detach() - old).abs().cpu().numpy())
            task_loss += float((-float(advantage) * logps.detach()).sum()) / (config.denominator * 64)
            kl_sum += float(sampled_kl(logps.detach(), ref).sum())
            tokens += len(logps)
            loss.backward()
            del logps, ref, old, loss
            print(json.dumps(dict(event='trajectory_backward', rank=rank, update=update,
                trajectory=index + 1, trajectories=len(records), sampled_tokens=record['generated_tokens'])), flush=True)
        differences = np.concatenate(differences)
        audit = dict(mean_abs=float(differences.mean()), p99=float(np.quantile(differences, .99)),
                     max_abs=float(differences.max()), mean_limit=.01, p99_limit=.05)
        audit['status'] = 'GO' if audit['mean_abs'] <= .01 and audit['p99'] <= .05 else 'HOLD'
        audit['exploratory_target_miss_allowed'] = config.allow_numerical_target_miss
        atomic_json(output / f'parity/update-{update:04d}-rank-{rank}.json', audit)
        accepted = torch.tensor(int(accept_parity(audit, config.allow_numerical_target_miss)), device=policy.device)
        if world > 1:
            dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
        if not bool(accepted):
            raise RuntimeError(f'Rollout/teacher consistency failed before optimizer step: {audit}')
        if world > 1:
            for parameter in policy.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True))
        lr = config.learning_rate * min(update / config.warmup, 1.)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if actor is not policy:
            with torch.no_grad():
                for target, source in zip(actor.parameters(), policy.parameters(), strict=True):
                    target.copy_(source)
        training_seconds = time.perf_counter() - training_started
        local = dict(task_loss=task_loss, kl_sum=kl_sum, generated_tokens=tokens,
            rollout_seconds=rollout_seconds, training_seconds=training_seconds, gradient_norm=norm,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(policy.device),
            zero_advantage_groups=int((np.abs(advantages.reshape(-1, 8)).sum(-1) == 0).sum()),
            mismatch_mean=audit['mean_abs'], mismatch_p99=audit['p99'])
        workers = gather(local, world)
        result = combine_metrics(gather(record_metrics(records), world))
        result.update(update=update, training_episodes=update * 64, learning_rate=lr, workers=workers,
            normalized_task_loss=sum(row['task_loss'] for row in workers),
            normalized_kl=sum(row['kl_sum'] for row in workers) / (config.denominator * 64),
            kl_per_generated_token=sum(row['kl_sum'] for row in workers) / result['generated_tokens'],
            rollout_seconds=max(row['rollout_seconds'] for row in workers),
            training_seconds=max(row['training_seconds'] for row in workers),
            rollout_generated_tok_s=result['generated_tokens'] / max(row['rollout_seconds'] for row in workers))
        result['normalized_objective'] = result['normalized_task_loss'] + config.kl_coefficient * result['normalized_kl']
        if rank == 0:
            atomic_json(output / f'update-metrics/update-{update:04d}.json', result)
            with (output / 'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(result) + '\n')
            print(json.dumps(dict(event='update', **result)), flush=True)
        del records, differences
        if update % config.validation_interval == 0 or update == config.updates:
            validation = validate(collector, output, config, update, rank, world)
            if validation['mean_return'] > best_return + 1e-12:
                best_return, best_update = validation['mean_return'], update
            if rank == 0:
                checkpoint = output / f'checkpoint-{update:04d}'
                if checkpoint.exists():
                    checkpoint.rename(checkpoint.with_name(f'orphaned-{time.time_ns()}-{checkpoint.name}'))
                save_checkpoint(checkpoint, policy, tokenizer,
                                dict(**validation, config=asdict(config), reference=config.model))
                atomic_json(output / f'validation-metrics/update-{update:04d}.json', validation)
                atomic_json(output / 'selected.json', dict(update=best_update, mean_return=best_return,
                            checkpoint=str(output / f'checkpoint-{best_update:04d}')))
        save_resume(output, policy, optimizer, config, update, best_return, best_update, rank, world, devices)
        if rank == 0 and (update % config.validation_interval == 0 or update == config.updates):
            retain_selected_and_latest(output, best_update, update)
        last_seconds = max(gather(time.perf_counter() - started, world))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attention', choices=['ps', 'full'], default='ps')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--window', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--validation-batch-size', type=int, default=32)
    parser.add_argument('--context-exhaustion-reward', type=float)
    parser.add_argument('--resume')
    parser.add_argument('--separate-actors', action='store_true')
    parser.add_argument('--no-activation-cpu-offload', action='store_true')
    parser.add_argument('--allow-numerical-target-miss', action='store_true')
    parser.add_argument('--initial-update-seconds', type=float, default=7200)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
