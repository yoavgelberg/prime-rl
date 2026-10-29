import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.distributed as dist

from .check_policy import accept_parity, parity, teacher_logps
from .checkpoints import retain_selected_and_latest, save_checkpoint
from .config import Config, SPLITS, generation_seed
from .control import handoff_requested
from .metrics import summarize
from .objective import leave_one_out, sampled_kl, trajectory_loss
from .policy import load_model
from .rollout import run_batch, save_episodes


def rank_info():
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=1))
    torch.cuda.set_device(rank)
    return rank, world


def gather(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def validate(actor, tokenizer, config, output, update, rank, world, batch_size):
    records = []
    seeds = list(SPLITS['validation'])[rank::world]
    for start in range(0, len(seeds), batch_size):
        batch = seeds[start:start+batch_size]
        records.extend(run_batch(actor, tokenizer, batch, batch, config, greedy=True))
    save_episodes(output / f'validation/update-{update:04d}-rank-{rank}.jsonl.gz', records)
    local = summarize(records)
    all_metrics = gather(local, world)
    result = {key: sum(m[key] * m['episodes'] for m in all_metrics)/128
              for key in ('native_return', 'matched_fraction', 'completion_rate', 'format_error_rate')}
    result.update(update=update, episodes=update * 64, validation_boards=128)
    return result


def train(args):
    from transformers import AutoTokenizer
    from .audit import token_audit
    rank, world = rank_info()
    assert world in (1, 2, 4, 8) and 8 % world == 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    output = Path(args.output or f'runs/concentration/{args.attention}-seed{args.seed}')
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cap = token_audit(tokenizer)['52']['required_cap']
    config = Config(model=args.model, attention=args.attention, window=args.window, seed=args.seed,
                    updates=args.updates, transcript_cap=cap, revision=args.revision)
    config.allow_numerical_target_miss = args.allow_numerical_target_miss
    config.decode_backend = 'cuda_graph' if os.environ.get('PS_CONCENTRATION_GRAPHS') == '1' else 'eager'
    config.gpu_layout = 'separate_actors' if args.separate_actors else 'colocated_data_parallel'
    config.rollout_batch_size = args.batch_size
    config.training_workers = world
    config.source_sha256 = os.environ.get('PS_CONCENTRATION_SOURCE_SHA', '')
    if not config.revision and len(Path(args.model).name) == 40:
        config.revision = Path(args.model).name
    if rank == 0:
        config.save(output / 'resolved_config.yaml')
    actor_device = rank + world if args.separate_actors else rank
    if args.separate_actors:
        assert torch.cuda.device_count() >= 2*world
    policy = load_model(args.model, f'cuda:{rank}')
    policy.requires_grad_(True)
    reference = load_model(args.model, f'cuda:{rank}')
    actor = load_model(args.model, f'cuda:{actor_device}') if actor_device != rank else policy
    masters = list(policy.parameters())
    optimizer = torch.optim.AdamW(masters, lr=config.learning_rate, betas=(config.beta1, config.beta2),
                                 eps=config.epsilon, weight_decay=0, foreach=True)
    permutation = np.random.default_rng(args.seed).permutation(1_000_000)
    if rank == 0:
        np.save(output / 'training_board_permutation.npy', permutation)
    start_update = 0
    best_return, best_update = -float('inf'), 0
    if args.resume:
        state = torch.load(args.resume, map_location=f'cuda:{rank}', weights_only=False)
        assert state['reference'] == args.model, 'Resume must preserve the frozen starting checkpoint'
        restored_config=state['config']
        fixed=('attention','seed','cards','limit','max_new_tokens','temperature','top_p','denominator',
               'learning_rate','warmup','kl_coefficient','beta1','beta2','epsilon','weight_decay','max_grad_norm')
        assert all(restored_config[key]==getattr(config,key) for key in fixed), 'Resume changed the experiment definition'
        assert config.attention=='full' or restored_config['window']==config.window, 'PS window is frozen'
        for master, tensor, param in zip(masters, state['masters'], policy.parameters(), strict=True):
            master.data.copy_(tensor)
            param.data.copy_(tensor)
        optimizer.load_state_dict(state['optimizer'])
        start_update, best_return, best_update = state['update'], state['best_return'], state['best_update']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        del state
        if rank == 0:
            recovery = output/f'recovery-{time.time_ns()}'
            for filename in ('updates.jsonl','validation.jsonl'):
                path = output/filename
                if path.exists():
                    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    if any(row['update'] > start_update for row in lines):
                        recovery.mkdir(exist_ok=True)
                        path.rename(recovery/filename)
                        path.write_text(''.join(json.dumps(row)+'\n' for row in lines if row['update'] <= start_update))
            for folder in ('rollouts','validation'):
                for path in (output/folder).glob('update-*-rank*.jsonl.gz'):
                    match=re.match(r'update-(\d+)-rank',path.name)
                    if match and int(match[1])>start_update:
                        destination=recovery/folder
                        destination.mkdir(parents=True,exist_ok=True)
                        path.rename(destination/path.name)
            (output/'selected.json').write_text(json.dumps(dict(update=best_update,native_return=best_return,
                                                               checkpoint=str(output/f'checkpoint-{best_update:04d}'))))
            (output/'latest_optimizer.json').write_text(json.dumps(dict(update=start_update)))
        if world > 1:
            dist.barrier()
    else:
        if rank == 0:
            initial_checkpoint = output/'checkpoint-0000'
            if initial_checkpoint.exists():
                initial_checkpoint.rename(output/f'orphaned-{time.time_ns()}-checkpoint-0000')
            recovery = output/f'recovery-{time.time_ns()}'
            for folder in ('rollouts','validation'):
                if (output/folder).exists():
                    recovery.mkdir(exist_ok=True)
                    (output/folder).rename(recovery/folder)
            for name in ('updates.jsonl','validation.jsonl'):
                if (output/name).exists():
                    recovery.mkdir(exist_ok=True)
                    (output/name).rename(recovery/name)
        if world > 1:
            dist.barrier()
        initial = validate(actor, tokenizer, config, output, 0, rank, world, args.batch_size)
        best_return = initial['native_return']
        if rank == 0:
            save_checkpoint(output / 'checkpoint-0000', policy, tokenizer,
                            dict(**initial, config=asdict(config), reference=args.model))
            with (output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(initial) + '\n')

    def save_resume(update):
        temporary = output/'latest_optimizer.tmp'
        torch.save(dict(masters=[p.detach() for p in masters], optimizer=optimizer.state_dict(), update=update,
                        best_return=best_return,best_update=best_update,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),config=asdict(config),reference=args.model),temporary)
        temporary.replace(output/'latest_optimizer.pt')
        receipt=output/'latest_optimizer.json.tmp'
        receipt.write_text(json.dumps(dict(update=update)))
        receipt.replace(output/'latest_optimizer.json')

    def yield_allocation(update):
        request=torch.tensor(int(handoff_requested()) if rank==0 else 0,device=policy.device)
        if world>1:
            dist.broadcast(request,src=0)
        if not bool(request):
            return False
        if rank==0:
            receipt=output/'latest_optimizer.json'
            if not receipt.exists() or json.loads(receipt.read_text())['update']!=update:
                save_resume(update)
                retain_selected_and_latest(output,best_update,update)
            print(json.dumps(dict(event='allocation_handoff',update=update)),flush=True)
        if world>1:
            dist.barrier()
            dist.destroy_process_group()
        return True

    if yield_allocation(start_update):
        return
    for update in range(start_update + 1, args.updates + 1):
        started = time.perf_counter()
        if actor is not policy:
            with torch.no_grad():
                for target, source in zip(actor.parameters(), policy.parameters(), strict=True):
                    target.copy_(source)
        group_indices = list(range(rank * (8//world), (rank+1) * (8//world)))
        boards = [int(permutation[(update-1)*8 + i]) for i in group_indices]
        seeds = [board for board in boards for _ in range(8)]
        rngs = [generation_seed(args.seed, update-1, i, k) for i in group_indices for k in range(8)]
        records = []
        rollout_start = time.perf_counter()
        for start in range(0, len(seeds), args.batch_size):
            records.extend(run_batch(actor, tokenizer, seeds[start:start+args.batch_size],
                                     rngs[start:start+args.batch_size], config))
        rollout_seconds = time.perf_counter() - rollout_start
        rollout_path = output/f'rollouts/update-{update:04d}-rank-{rank}.jsonl.gz'
        if rollout_path.exists():
            rollout_path.rename(rollout_path.with_name(f'recovered-{time.time_ns()}-{rollout_path.name}'))
        save_episodes(rollout_path, records)
        if update == 1:
            audit = parity(policy, records[:1])
            audit['exploratory_target_miss_allowed'] = args.allow_numerical_target_miss
            (output / f'parity-rank-{rank}.json').write_text(json.dumps(audit, indent=2))
            if not accept_parity(audit, args.allow_numerical_target_miss):
                raise RuntimeError(f'Rollout/training mismatch: {audit}')
        returns = np.array([r['native_return'] for r in records]).reshape(-1, 8)
        advantages = leave_one_out(returns).ravel()
        optimizer.zero_grad(set_to_none=True)
        task_loss, kl_sum, tokens, mismatch = 0., 0., 0, []
        training_start = time.perf_counter()
        for record, advantage in zip(records, advantages, strict=True):
            with torch.no_grad():
                ref = teacher_logps(reference, record, temperature=config.temperature)
            logps = teacher_logps(policy, record, grad=True, temperature=config.temperature)
            kl = sampled_kl(logps, ref)
            loss = trajectory_loss(logps, ref, advantage, config)
            loss.backward()
            task_loss += float((-float(advantage)*logps.detach()).sum()) / (config.denominator*64)
            kl_sum += float(kl.detach().sum())
            tokens += len(logps)
            old = torch.tensor([p for p in record['old_tempered_logps'] if p is not None], device=policy.device)
            mismatch.extend((logps.detach()-old).abs().cpu().tolist())
            del logps, ref, kl, loss
        if world > 1:
            for master in masters:
                dist.all_reduce(master.grad, op=dist.ReduceOp.SUM)
        norm = float(torch.nn.utils.clip_grad_norm_(masters, config.max_grad_norm, error_if_nonfinite=True))
        lr = config.learning_rate * min(update/config.warmup, 1.)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if actor is not policy:
                for target, source in zip(actor.parameters(), policy.parameters(), strict=True):
                    target.copy_(source)
        train_seconds = time.perf_counter() - training_start
        metric = dict(update=update, episodes=update*64, rank=rank, native_return=float(returns.mean()),
                      matched_fraction=float(np.mean([r['matched_fraction'] for r in records])),
                      normalized_task_loss=task_loss, kl_sum=kl_sum, kl_per_generated_token=kl_sum/tokens,
                      normalized_kl=kl_sum/(config.denominator*64), generated_tokens=tokens,
                      zero_advantage_groups=int((np.abs(advantages.reshape(-1, 8)).sum(-1) == 0).sum()),
                      raw_transcript_tokens=sum(r['raw_transcript_tokens'] for r in records),
                      learning_rate=lr, gradient_norm=norm, rollout_seconds=rollout_seconds,
                      training_seconds=train_seconds, seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                      mismatch_mean=float(np.mean(mismatch)), mismatch_p99=float(np.quantile(mismatch, .99)))
        all_metrics = gather(metric, world)
        if rank == 0:
            global_tokens = sum(m['generated_tokens'] for m in all_metrics)
            global_kl = sum(m['normalized_kl'] for m in all_metrics)
            global_metric = dict(update=update,episodes=update*64,workers=all_metrics,
                native_return=float(np.mean([m['native_return'] for m in all_metrics])),
                matched_fraction=float(np.mean([m['matched_fraction'] for m in all_metrics])),
                normalized_objective=sum(m['normalized_task_loss'] for m in all_metrics)+config.kl_coefficient*global_kl,
                normalized_kl=global_kl,kl_per_generated_token=sum(m['kl_sum'] for m in all_metrics)/global_tokens,
                generated_tokens=global_tokens,raw_transcript_tokens=sum(m['raw_transcript_tokens'] for m in all_metrics),
                rollout_seconds=max(m['rollout_seconds'] for m in all_metrics),
                training_seconds=max(m['training_seconds'] for m in all_metrics),
                generated_tokens_per_second=global_tokens/max(m['rollout_seconds'] for m in all_metrics))
            with (output / 'updates.jsonl').open('a') as handle:
                handle.write(json.dumps(global_metric) + '\n')
            print(json.dumps(dict(event='update', **metric)), flush=True)
        if update % 16 == 0 or update == args.updates:
            validation = validate(actor, tokenizer, config, output, update, rank, world, args.batch_size)
            if validation['native_return'] > best_return+1e-12:
                best_return, best_update = validation['native_return'], update
            if rank == 0:
                checkpoint = output / f'checkpoint-{update:04d}'
                if checkpoint.exists():
                    checkpoint.rename(checkpoint.with_name(f'orphaned-{time.time_ns()}-{checkpoint.name}'))
                save_checkpoint(checkpoint, policy, tokenizer,
                                dict(**validation, config=asdict(config), reference=args.model))
                (output / 'selected.json').write_text(json.dumps(dict(update=best_update, native_return=best_return,
                                                                     checkpoint=str(output/f'checkpoint-{best_update:04d}')), indent=2))
                with (output / 'validation.jsonl').open('a') as handle:
                    handle.write(json.dumps(validation) + '\n')
                save_resume(update)
                retain_selected_and_latest(output,best_update,update)
            if world > 1:
                dist.barrier()
        if yield_allocation(update):
            return
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', choices=['ps', 'full'], required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--revision', default='')
    parser.add_argument('--window', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--separate-actors', action='store_true')
    parser.add_argument('--allow-numerical-target-miss', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
