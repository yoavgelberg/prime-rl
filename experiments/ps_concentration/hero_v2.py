"""Independent vLLM actors and pipelined synchronous FA4-PS GRPO trainers."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


@dataclass(frozen=True)
class HeroConfig:
    model: str
    source: str
    output: str
    node_dir: str
    source_sha256: str
    deadline: float
    gpus: int = 8
    seed: int = 11
    updates: int = 128
    boards: int = 16
    group_size: int = 8
    denominator: int = 26624
    window: int = 1024
    transcript_cap: int = 262144
    max_responses: int = 96
    thinking_token_limit: int = 128
    action_token_limit: int = 32
    temperature: float = .6
    learning_rate: float = 2e-6
    beta1: float = .9
    beta2: float = .95
    adam_epsilon: float = 1e-8
    max_gradient_norm: float = 1.
    warmup_updates: int = 8
    schedule_updates: int | None = None
    minimum_lr_fraction: float = .2
    continuation_start_update: int | None = None
    continuation_warmup_updates: int = 8
    continuation_initial_lr: float = 4e-7
    clip_ratio: float = .2
    kl_coefficient: float = .01
    entropy_coefficient: float = .001
    format_penalty: float = .02
    invalid_penalty: float = .02
    head_block_tokens: int = 128
    checkpoint_interval: int = 8
    checkpoint_io_lock: str | None = None
    export_interval: int = 16
    wandb_id: str = 'rng-4b-ps-grpo-s11-hero-v2-a1'
    revision: str = '768f209d9ea81521153ed38c47d515654e938aea'
    defer_native_logs: bool = False
    accepted_resume_sources: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'accepted_resume_sources', tuple(self.accepted_resume_sources))
        if self.gpus not in (4, 8) or self.boards != 16 or self.group_size != 8:
            raise ValueError('Hero uses16boards×8rollouts with4or8allocatedGPUs')
        if Path(self.model).name != self.revision or len(self.source_sha256) != 64:
            raise ValueError('Pin the original model revision and captured implementation')
        if self.window not in (64, 128, 256, 512, 1024, 2048, 4096) or self.thinking_token_limit != 128 or self.max_responses != 96:
            raise ValueError('Keep the accepted memory task and thinking budget')
        if self.schedule_updates is not None and self.schedule_updates <= self.warmup_updates:
            raise ValueError('The fixed learning-rate horizon must exceed warmup')
        if self.continuation_start_update is not None and (self.continuation_start_update < 0
                or self.continuation_warmup_updates < 1 or not 0 < self.continuation_initial_lr <= self.learning_rate):
            raise ValueError('Continuation warmup needs a positive duration and an initial rate below its target')
        if self.boards % self.trainers or self.group_size < 2 or self.action_token_limit < 1:
            raise ValueError('Every trainer needs complete groups and a finite answer budget')
        if min(self.learning_rate,self.kl_coefficient,self.entropy_coefficient,self.denominator) <= 0:
            raise ValueError('Learning rate,KL,entropy and denominator must be positive')
        if type(self.defer_native_logs) is not bool or any(len(value)!=64 for value in self.accepted_resume_sources):
            raise ValueError('Logging mode must be boolean and reviewed resume sources must be SHA256 digests')

    @property
    def batch_size(self):
        return self.boards * self.group_size

    @property
    def actors(self):
        return self.gpus // 2

    @property
    def trainers(self):
        return self.gpus // 2

    @property
    def live_games(self):
        return self.batch_size // self.actors


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def checkpoint_write_lock(path):
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def reusable_episode(path, spec, config):
    path = Path(path)
    if not path.exists():
        return None
    row = read(path)
    if any(row.get(key) != value for key, value in spec.items()):
        raise ValueError('Saved episode differs from the frozen seed/policy plan')
    for key in ('thinking_token_limit', 'action_token_limit'):
        if row.get(key) != getattr(config, key):
            raise ValueError('Saved episode uses a different generation controller')
    tokens = Path(row['token_record'])
    if hashlib.sha256(tokens.read_bytes()).hexdigest() != row['token_record_sha256']:
        raise ValueError('Saved rollout tokens changed')
    return row


def wait(path, *, deadline=None):
    path = Path(path)
    while not path.exists():
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.1)
    return read(path)


def identity(config):
    result = dict(task='RNGBench native textual Matching Pairs4x4', algorithm='GRPO',
        variant='hero-v2', model_revision=config.revision, reference_checkpoint=config.model,
        source_sha256=config.source_sha256, attention='ps', window=config.window,
        group_size=config.group_size, global_batch_size=config.batch_size,
        kl_coefficient=config.kl_coefficient, entropy_coefficient=config.entropy_coefficient,
        format_penalty=config.format_penalty, invalid_penalty=config.invalid_penalty,
        denominator=config.denominator, thinking_token_limit=config.thinking_token_limit,
        action_token_limit=config.action_token_limit,
        learning_rate=config.learning_rate, optimizer_betas=[config.beta1,config.beta2],
        adam_epsilon=config.adam_epsilon, gradient_clip=config.max_gradient_norm,
        warmup_updates=config.warmup_updates, minimum_lr_fraction=config.minimum_lr_fraction,
        temperature=config.temperature, top_p=1., top_k=-1,
        backend='FA4-PS native-effective BF16 teacher; independent vLLM actors',
        rollout_importance_correction='One fresh epoch,tokenwise PPO ratio clipping using actual native rollout logps',
        native_teacher_strict_parity_proven=False)
    if config.schedule_updates is not None:
        result['lr_schedule_updates'] = config.schedule_updates
    if config.continuation_start_update is not None:
        result['continuation_lr_schedule'] = dict(start_update=config.continuation_start_update,
            warmup_updates=config.continuation_warmup_updates, initial_lr=config.continuation_initial_lr,
            target_lr=config.learning_rate, after_warmup='constant')
    return result


def learning_rate(config, update):
    if config.continuation_start_update is not None:
        fraction = min(1., max(0., (update-config.continuation_start_update)/config.continuation_warmup_updates))
        return config.continuation_initial_lr + fraction*(config.learning_rate-config.continuation_initial_lr)
    horizon = config.schedule_updates or config.updates
    progress = min(1., max(0., (update-config.warmup_updates)/max(1, horizon-config.warmup_updates)))
    decay = config.minimum_lr_fraction+(1-config.minimum_lr_fraction)*(1+math.cos(math.pi*progress))/2
    return config.learning_rate * min(update/config.warmup_updates, 1.) * decay


def validate_resume_identity(saved, config):
    expected = identity(config)
    if saved == expected:
        return
    if saved.get('source_sha256') not in config.accepted_resume_sources:
        raise ValueError('Checkpoint source differs without an explicitly reviewed source migration')
    if saved | dict(source_sha256=expected['source_sha256']) != expected:
        raise ValueError('Source migration cannot change any model/controller/objective/optimizer identity field')


def specs(config, update, permutation):
    from .rng_training_config import training_rng

    return [dict(rank=0, actor_rank=board % config.actors, board_index=board,
        board_seed=int(permutation[(update - 1) * config.boards + board]), rollout_index=k,
        generation_seed=training_rng(config.seed, update - 1, board, k),
        episode_id=f'u{update:04d}-g{board}-r{k}', policy_version=update - 1,
        policy_identity=f'{config.wandb_id}:update-{update-1}')
        for board in range(config.boards) for k in range(config.group_size)]


async def actor_main(config, rank):
    from dataclasses import asdict
    from prime_rl.trainer.ps_backward.config import StreamConfig
    from prime_rl.inference.vllm.ps_fa4 import register
    from prime_rl.orchestrator.ps_games import GameRolloutRunner, worker_rpc
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    register()
    root, node = Path(config.output), Path(config.node_dir)
    hf = read(Path(config.model) / 'config.json')
    kv_bytes = hf['num_hidden_layers'] * hf['num_key_value_heads'] * hf.get('head_dim', 128) * 4
    options = dict(model=config.model, dtype='bfloat16', skip_tokenizer_init=True,
        tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
        data_parallel_size_local=1, worker_cls='experiments.ps_concentration.hero_inference.HeroInferenceWorker',
        max_model_len=config.transcript_cap, max_num_seqs=config.live_games,
        max_num_batched_tokens=1024, block_size=128, enable_chunked_prefill=True,
        enable_prefix_caching=False, attention_backend='CUSTOM',
        kv_cache_memory_bytes=max(8*1024**3, kv_bytes*(512+config.window+256)*config.live_games),
        hf_overrides=dict(use_sliding_window=True, sliding_window=config.window),
        additional_config=dict(ps_streaming=asdict(StreamConfig(window=config.window,
            max_requests=1, chunk_tokens=1024, ring_tokens=1024))),
        cudagraph_capture_sizes=[n for n in (1,2,4,8,16,32,64) if n <= config.live_games])
    write(root / f'actor-{rank}-engine.json', options)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
    runner = GameRolloutRunner(engine)
    write(node / f'actor-{rank}/ready.json', dict(utc=utc(), pid=os.getpid(), options=options))
    version = 0
    update = 1
    resume = root / 'checkpoints/latest.pt.json'
    if resume.exists():
        update = read(resume)['policy_version'] + 1
    while True:
        command = node / f'actor-{rank}/update-{update:04d}.json'
        while not command.exists():
            if (node / 'stop').exists():
                engine.shutdown()
                return
            await asyncio.sleep(.1)
        request = read(command)
        if request['policy_version'] != version:
            result = await worker_rpc(engine, 'ps_hero_load_weights', timeout=300.,
                kwargs=dict(path=request['weights'], policy_version=request['policy_version']))
            assert len(result) == 1 and result[0]['policy_version'] == request['policy_version']
            version = request['policy_version']
        batch = request['specs']
        assert len(batch) == config.batch_size // config.actors and all(x['rank'] == 0 for x in batch)
        output = root / f'rollouts/update-{update:04d}/actor-{rank}'
        reused, fresh = [], []
        for spec in batch:
            saved = reusable_episode(output / spec['episode_id'] / 'result.json', spec, config)
            if saved is None:
                folder = output / spec['episode_id']
                if folder.exists():
                    folder.rename(folder.with_name(folder.name + f'.abandoned-{time.time_ns()}'))
                fresh.append(spec)
            else:
                reused.append(saved)
        report = dict(episodes=[], seconds=0.)
        if fresh:
            report = await runner.collect('experiments.ps_concentration.rng_vllm.GameCohort',
            dict(model=config.model, source=config.source, output=str(output), specs=fresh,
                attention='ps', window=config.window, transcript_cap=config.transcript_cap,
                max_responses=config.max_responses, temperature=config.temperature,
                thinking_token_limit=config.thinking_token_limit,
                action_token_limit=config.action_token_limit,
                thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n',
                greedy=False, context_exhaustion_reward=-1., compress_native=True,
                defer_native_logs=config.defer_native_logs),
            backward=False, policy_version=version)
            await runner.release()
        report['episodes'].extend(reused)
        report['reused_episodes'] = [row['episode_id'] for row in reused]
        write(node / f'actor-{rank}/finished-{update:04d}.json', report)
        update += 1
    engine.shutdown()


def trainer_main(config, rank, port):
    import torch
    import torch.distributed as dist
    from .policy import load_model
    from .native_teacher import NativeTeacher
    from .hero_objective import advantages, shaped_return, trajectory_backward
    from prime_rl.trainer.ps_backward.checkpoint import model_state, cpu_tree

    torch.set_num_threads(2)
    torch.manual_seed(config.seed + rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=config.trainers)
    root, node = Path(config.output), Path(config.node_dir)
    policy = load_model(config.model, train=True)
    reference = load_model(config.model)
    if not torch.equal(policy.model.rotary_emb.inv_freq, reference.model.rotary_emb.inv_freq):
        raise ValueError('Policy and reference must preserve the same original RoPE frequencies')
    policy.train(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate,
        betas=(config.beta1,config.beta2), eps=config.adam_epsilon, weight_decay=0., fused=True)
    version = 0
    latest = root / 'checkpoints/latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        validate_resume_identity(state['identity'], config)
        if state['identity'] != identity(config):
            write(root / f'source-migration-{os.environ["SLURM_JOB_ID"]}-{rank}.json', dict(
                policy_version=state['policy_version'], old_identity=state['identity'],
                new_identity=identity(config), approved_sources=config.accepted_resume_sources,
                scope='Resume identical model/optimizer/RNG; only reviewed implementation source changes'))
        policy.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            if 'step' in values:
                values['step'] = values['step'].to(policy.device)
        version = state['policy_version']
        rngs = state.get('rngs_by_rank', [state['rngs']])
        local_rng = rngs[rank % len(rngs)]
        torch.set_rng_state(local_rng['torch'])
        torch.cuda.set_rng_state(local_rng['cuda'])
        del state
    actor_adapter = NativeTeacher(policy, config.window, backend='fa4', checkpoint_layers=True)
    ref_adapter = NativeTeacher(reference, config.window, backend='fa4')
    writer = ThreadPoolExecutor(max_workers=1) if rank == 0 else None
    pending = None

    def save_rng(update):
        path = node / f'rng-{update:04d}-{rank}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()), temporary)
        temporary.replace(path)

    def weights(update):
        path = node / f'weights-{update:04d}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(dict(policy_version=update, model=model_state(policy, dtype=torch.bfloat16)), temporary)
        temporary.replace(path)
        write(node / f'weights-{update:04d}.json', dict(policy_version=update, path=str(path), bytes=path.stat().st_size))

    def snapshot(update):
        states = []
        for other in range(config.trainers):
            path = node / f'rng-{update:04d}-{other}.pt'
            while not path.exists():
                time.sleep(.05)
            states.append(torch.load(path, map_location='cpu', weights_only=True))
        saved = dict(format='ps-hero-v2-adamw-v1', policy_version=update,
            model=model_state(policy), optimizer=cpu_tree(optimizer.state_dict()),
            rngs=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()),
            rngs_by_rank=states,
            identity=identity(config), resolved_config=asdict(config))

        def commit_unlocked():
            directory = root / 'checkpoints'
            directory.mkdir(exist_ok=True)
            path = directory / 'latest.pt'
            temporary = directory / f'latest.pt.tmp-{update}'
            torch.save(saved, temporary)
            temporary.replace(path)
            metadata = dict(policy_version=update, bytes=path.stat().st_size,
                identity=identity(config), checkpoint_path=str(path.resolve()), utc=utc())
            write(path.with_suffix('.pt.json'), metadata)
            if update % config.export_interval == 0 or update == config.updates:
                export = Path(config.node_dir).parent / 'candidate-policies' / f'policy-{update:04d}'
                export.mkdir(parents=True, exist_ok=True)
                policy.config.save_pretrained(export)
                from safetensors.torch import save_file
                tensors = {}
                aliases = {}
                for name, value in saved['model'].items():
                    key = value.untyped_storage().data_ptr(), value.shape, value.stride()
                    if key in aliases:
                        continue
                    aliases[key] = name
                    tensors[name] = value.bfloat16().contiguous()
                save_file(tensors, str(export / 'model.safetensors'))
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(config.model).save_pretrained(export)
                write(export / 'experiment.json', dict(identity(config), policy_version=update,
                    storage=f'BF16 effective weights; FP32 optimizer state at {path.resolve()}; '
                            'durability is recorded in checkpoint/archive receipts'))
            return metadata

        def commit():
            with checkpoint_write_lock(config.checkpoint_io_lock):
                return commit_unlocked()

        return writer.submit(commit)

    if rank == 0 and version:
        weights(version)
    write(node / f'trainer-{rank}/ready.json', dict(utc=utc(), pid=os.getpid(), policy_version=version))
    for update in range(version + 1, config.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        completed = set()
        expected = config.boards // config.trainers
        statistics = []
        started = time.perf_counter()
        while len(completed) < expected:
            if (node / 'stop').exists():
                save_rng(update-1)
                if pending is not None:
                    pending.result()
                metadata = root / 'checkpoints/latest.pt.json'
                if rank == 0 and (not metadata.exists() or read(metadata)['policy_version'] != update-1):
                    snapshot(update-1).result()
                dist.barrier()
                dist.destroy_process_group()
                return
            commands = sorted((node / f'trainer-{rank}').glob(f'group-{update:04d}-*.json'))
            ready = [path for path in commands if path.name not in completed]
            if not ready:
                time.sleep(.1)
                continue
            for path in ready:
                command = read(path)
                rows = command['rows']
                if len(rows) != config.group_size or {r['policy_version'] for r in rows} != {update-1}:
                    raise ValueError('Group must have8freshrollouts at the fixed policy')
                if len({r['board_identity_sha256'] for r in rows}) != 1:
                    raise ValueError('Board-group rollouts did not reset identically')
                returns = [shaped_return(r, format_penalty=config.format_penalty,
                    invalid_penalty=config.invalid_penalty) for r in rows]
                adv, std = advantages(returns)
                metrics = []
                for row, advantage in zip(rows, adv, strict=True):
                    with gzip.open(row['token_record'], 'rt') as handle:
                        record = json.load(handle)
                    if sum(record['loss_mask']) != row['generated_tokens']:
                        raise ValueError('Every sampled token must contribute exactly once')
                    values = trajectory_backward(policy, reference, record, advantage, config,
                        adapters=(actor_adapter, ref_adapter))
                    metrics.append(values)
                    write(node / f'trainer-{rank}/progress.json', dict(update=update,
                        board=command['board'], episode=row['episode_id'], metrics=values, utc=utc()))
                statistics.append(dict(board=command['board'], reward_std=std,
                    advantages=adv, shaped_returns=returns, trajectories=metrics))
                completed.add(path.name)
        write(node / f'trainer-{rank}/gradients-{update:04d}.json', dict(groups=statistics,
            seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        wait(node / f'step-{update:04d}.json', deadline=config.deadline)
        step_started = time.perf_counter()
        for parameter in policy.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm, error_if_nonfinite=True)
        lr = learning_rate(config, update)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        actor_adapter.projections.casts.clear()
        policy._ps_update_version = update
        checkpoint = update % config.checkpoint_interval == 0 or update == config.updates or (
            node / f'checkpoint-final-{update:04d}').exists()
        if checkpoint:
            save_rng(update)
        if rank == 0:
            weights(update)
            if pending is not None and pending.done():
                pending.result()
            if checkpoint:
                if pending is not None:
                    pending.result()
                pending = snapshot(update)
        write(node / f'trainer-{rank}/stepped-{update:04d}.json', dict(update=update,
            gradient_norm=float(norm), gradient_clipped=float(norm)>config.max_gradient_norm, learning_rate=lr,
            optimizer_and_exchange_seconds=time.perf_counter()-step_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), utc=utc()))
        if (node / f'checkpoint-final-{update:04d}').exists():
            if rank == 0:
                if pending is not None:
                    pending.result()
                metadata = root / 'checkpoints/latest.pt.json'
                if not metadata.exists() or read(metadata)['policy_version'] != update:
                    snapshot(update).result()
            dist.barrier()
            while not (node / 'stop').exists():
                time.sleep(.1)
            break
    if pending is not None:
        pending.result()
    if writer is not None:
        writer.shutdown()
    dist.destroy_process_group()


def aggregate(config, rows):
    from .overlap_config import episode_metrics
    from .hero_objective import shaped_return

    metrics = episode_metrics(rows)
    shaped = [shaped_return(r, format_penalty=config.format_penalty,
        invalid_penalty=config.invalid_penalty) for r in rows]
    metrics['mean_native_return'] = metrics['mean_return']
    metrics['native_reward_std'] = metrics['reward_std']
    metrics['mean_shaped_return'] = math.fsum(shaped)/len(shaped)
    metrics['reward_std'] = math.sqrt(math.fsum((r-metrics['mean_shaped_return'])**2 for r in shaped)/len(shaped))
    metrics['mean_return'] = metrics['mean_shaped_return']
    return metrics


def coordinator(config, config_path):
    import numpy as np
    import wandb
    from .overlap_config import permutation, protocol_lock

    root, node = Path(config.output), Path(config.node_dir)
    root.mkdir(parents=True, exist_ok=True)
    node.mkdir(parents=True, exist_ok=False)
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices) != config.gpus:
        raise ValueError('Use exactly the assigned4or8GPUallocation')
    model_lock = json.loads((Path(config.source)/'SOURCE.json').read_text())
    assert model_lock['commit'] == 'e3d195e857727dfa3d4a96561be202098d67da83'
    write(root / 'resolved_config.json', asdict(config))
    write(root / 'identity.json', identity(config))
    write(root / 'dependency-revision-lock.json', protocol_lock(config))
    boards = permutation(config.seed)
    np.save(root / 'training-board-permutation.npy', boards)
    sock = socket.socket()
    sock.bind(('127.0.0.1',0))
    port = sock.getsockname()[1]
    sock.close()
    children, logs = [], []
    for role, count in [('actor',config.actors),('trainer',config.trainers)]:
        for rank in range(count):
            control = node / f'{role}-{rank}'
            control.mkdir()
            physical = rank if role == 'actor' else config.actors+rank
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[physical], OMP_NUM_THREADS='2')
            for name in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
                env.pop(name,None)
            handle = (root/f'{role}-{rank}-{os.environ["SLURM_JOB_ID"]}.log').open('a')
            command = [sys.executable,'-u','-m','experiments.ps_concentration.hero_v2',
                '--role',role,'--rank',str(rank),'--port',str(port),'--config',str(config_path)]
            child = subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((role,rank,child))
            logs.append(handle)
    write(root / f'processes-{os.environ["SLURM_JOB_ID"]}.json', dict(utc=utc(), coordinator=os.getpid(),
        children=[dict(role=r,rank=n,pid=p.pid) for r,n,p in children], devices=devices,
        node_dir=str(node), policy_identity=identity(config)))
    monitor = wandb.init(entity='tech-ml', project='ps-memory', id=config.wandb_id,
        resume='allow', name=f'Qwen3-4B PS-GRPO W{config.window} seed{config.seed}'
            + (' continuation' if config.schedule_updates else ' hero v2'),
        config=asdict(config), allow_val_change=True,
        tags=['hero-v2','fa4-ps','kl','independent-actors'])
    wandb.define_metric('train/update')
    wandb.define_metric('train/*', step_metric='train/update')
    write(root/'wandb.json',dict(url=monitor.url,id=monitor.id))

    def healthy():
        for role, rank, process in children:
            if process.poll() is not None:
                raise RuntimeError(f'{role}{rank} exited{process.returncode}; inspect its preserved log')

    latest = root/'checkpoints/latest.pt.json'
    version = read(latest)['policy_version'] if latest.exists() else 0
    fork = root / 'fork-receipt.json'
    parent_version = read(fork)['parent_policy_version'] if fork.exists() else 0
    try:
        while not all((node/f'{r}-{n}/ready.json').exists() for r,n,_ in children):
            healthy()
            write(root/'live.json',dict(utc=utc(),phase='initializing',policy_version=version,
                ready=[f'{r}-{n}' for r,n,_ in children if (node/f'{r}-{n}/ready.json').exists()]))
            time.sleep(1.)
        prior_seconds = 300.
        for update in range(version+1,config.updates+1):
            healthy()
            if config.deadline-time.time() < max(900.,prior_seconds*1.3+300):
                write(root/'allocation-handoff.json',dict(policy_version=version,utc=utc(),job=os.environ['SLURM_JOB_ID']))
                break
            started=time.perf_counter()
            batch=specs(config,update,boards)
            write(root/f'rollout-plans/update-{update:04d}.json',batch)
            for rank in range(config.actors):
                write(node/f'actor-{rank}/update-{update:04d}.json',dict(policy_version=update-1,
                    weights=str(node/f'weights-{update-1:04d}.pt'),
                    specs=[s for s in batch if s['actor_rank']==rank]))
            dispatched=set()
            assignments=[0]*config.trainers
            results={}
            while len(dispatched)<config.boards:
                healthy()
                for spec in batch:
                    if spec['episode_id'] in results:
                        continue
                    path=root/f'rollouts/update-{update:04d}/actor-{spec["actor_rank"]}/{spec["episode_id"]}/result.json'
                    if path.exists():
                        row=read(path)
                        if any(row[k]!=spec[k] for k in spec):
                            raise ValueError('A fresh actor result differs from its recorded seed plan')
                        results[spec['episode_id']]=row
                for board in range(config.boards):
                    if board in dispatched:
                        continue
                    selected=[s['episode_id'] for s in batch if s['board_index']==board]
                    if all(s in results for s in selected):
                        trainer=min(range(config.trainers),key=lambda r:(assignments[r],r))
                        if assignments[trainer] >= config.boards//config.trainers:
                            raise ValueError('Completed group dispatch exceeds trainer capacity')
                        write(node/f'trainer-{trainer}/group-{update:04d}-{board:02d}.json',
                            dict(board=board,rows=[results[s] for s in selected]))
                        dispatched.add(board)
                        assignments[trainer]+=1
                values=aggregate(config,list(results.values())) if results else {}
                write(root/'live.json',dict(utc=utc(),phase='rollout-and-training',update=update,
                    policy_version=version,completed=len(results),total=config.batch_size,
                    dispatched_groups=len(dispatched),partial_metrics=values,
                    seconds=time.perf_counter()-started))
                time.sleep(.5)
            while not all((node/f'actor-{r}/finished-{update:04d}.json').exists() for r in range(config.actors)):
                healthy()
                time.sleep(.2)
            while not all((node/f'trainer-{r}/gradients-{update:04d}.json').exists() for r in range(config.trainers)):
                healthy()
                write(root/'live.json',dict(utc=utc(),phase='training-drain',update=update,
                    policy_version=version,completed=len(results),total=config.batch_size,
                    partial_metrics=aggregate(config,list(results.values())),seconds=time.perf_counter()-started))
                time.sleep(.5)
            final=update==config.updates or config.deadline-time.time()<max(900.,prior_seconds*1.3+300)
            if final:
                (node/f'checkpoint-final-{update:04d}').touch()
            write(node/f'step-{update:04d}.json',dict(update=update))
            while not all((node/f'trainer-{r}/stepped-{update:04d}.json').exists() for r in range(config.trainers)):
                healthy()
                time.sleep(.1)
            weight=wait(node/f'weights-{update:04d}.json',deadline=config.deadline)
            train=[read(node/f'trainer-{r}/gradients-{update:04d}.json') for r in range(config.trainers)]
            steps=[read(node/f'trainer-{r}/stepped-{update:04d}.json') for r in range(config.trainers)]
            trajectories=[t for rank in train for group in rank['groups'] for t in group['trajectories']]
            assert len(trajectories)==config.batch_size
            tokens=sum(t['sampled_tokens'] for t in trajectories)
            rows=[results[s['episode_id']] for s in batch]
            metrics=aggregate(config,rows)
            seconds=time.perf_counter()-started
            metrics.update(update=update,training_episodes=update*config.batch_size,
                new_training_episodes=(update-parent_version)*config.batch_size,
                kl_per_generated_token=sum(t['kl_sum'] for t in trajectories)/tokens,
                entropy_per_generated_token=sum(t['entropy_sum'] for t in trajectories)/tokens,
                normalized_policy_loss=sum(t['policy_sum'] for t in trajectories)/(config.denominator*config.batch_size),
                normalized_kl=sum(t['kl_sum'] for t in trajectories)/(config.denominator*config.batch_size),
                normalized_entropy=sum(t['entropy_sum'] for t in trajectories)/(config.denominator*config.batch_size),
                normalized_total_loss=sum(t['normalized_loss'] for t in trajectories),
                clip_fraction=sum(t['clipped_tokens'] for t in trajectories)/tokens,
                rollout_teacher_mean_abs_logp=sum(t['rollout_teacher_abs_logp_sum'] for t in trajectories)/tokens,
                rollout_teacher_max_abs_logp=max(t['rollout_teacher_max_abs_logp'] for t in trajectories),
                gradient_norm=steps[0]['gradient_norm'],learning_rate=steps[0]['learning_rate'],
                seconds=seconds,generated_tokens_per_second=metrics['generated_tokens']/seconds,
                zero_advantage_groups=sum(g['reward_std']==0 for t in train for g in t['groups']),
                peak_training_allocated_bytes=max(t['peak_allocated_bytes'] for t in train),
                actor_concurrency_per_gpu=config.live_games,actor_queue_per_gpu=config.batch_size//config.actors,
                reference_kl_enabled=True)
            write(root/f'update-metrics/update-{update:04d}.json',dict(metrics=metrics,trainer=train,steps=steps,
                actor_weights=weight,episode_ids=[r['episode_id'] for r in rows]))
            with (root/'episodes.jsonl').open('a') as handle:
                for row in rows:
                    handle.write(json.dumps(row)+'\n')
            write(root/'state.json',dict(policy_version=update,identity=identity(config),
                training_episodes=update*config.batch_size,utc=utc()))
            wandb.log({**{f'train/{k}':v for k,v in metrics.items() if isinstance(v,(int,float,bool))},
                'train/avg_reward':metrics['mean_return'],'train/avg_native_reward':metrics['mean_native_return'],
                'train/avg_unshaped_efficiency_reward':metrics['mean_native_return'],
                'train/update':update})
            print(json.dumps(dict(event='train',**metrics)),flush=True)
            old=node/f'weights-{update-2:04d}.pt'
            old.unlink(missing_ok=True)
            version=update
            prior_seconds=seconds
            if final:
                break
        write(node/'stop',dict(policy_version=version,utc=utc()))
        for _,_,child in children:
            child.wait(timeout=300.)
        write(root/'training-complete.json',dict(policy_version=version,target=config.updates,
            reached_target=version==config.updates,utc=utc(),checkpoint=str(latest)))
    finally:
        write(node/'stop',dict(policy_version=version,utc=utc()))
        for _,_,child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for _,_,child in children:
            try:
                child.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10.)
        monitor.finish()
        for handle in logs:
            handle.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--role',choices=['coordinator','actor','trainer'],default='coordinator')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    config=HeroConfig(**read(args.config))
    if args.role=='actor':
        asyncio.run(actor_main(config,args.rank))
    elif args.role=='trainer':
        trainer_main(config,args.rank,args.port)
    else:
        def interrupted(signum, frame):
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,interrupted)
        coordinator(config,args.config)


if __name__=='__main__':
    main()
