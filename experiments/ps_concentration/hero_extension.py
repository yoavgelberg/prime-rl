"""Exploratory matched continuations from the competent W1024 update192 policy."""

import argparse
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .hero_continue import state_fingerprint, training_state
from .hero_exports import file_hash
from .hero_finish import require_idle_devices
from .hero_pair import device_groups, observe, saved_version, validate_resume
from .hero_v2 import HeroConfig, identity, read, utc, write


def extension_config(parent, *, window, output, node_dir, source_sha256, deadline):
    if parent.window != 1024 or parent.seed != 11 or parent.updates != 192:
        raise ValueError('Resume the completed seed11 W1024 update192 branch')
    if window not in (1024, 512):
        raise ValueError('The approved extension compares W1024 and W512')
    return replace(parent, window=window, output=str(output), node_dir=str(node_dir),
        source_sha256=source_sha256, deadline=deadline, gpus=4, updates=320,
        learning_rate=1e-6, continuation_start_update=192, continuation_warmup_updates=8,
        continuation_initial_lr=4e-7, checkpoint_interval=16, export_interval=16,
        checkpoint_io_lock=str(Path(output).parents[1] / 'checkpoint-io.lock'),
        accepted_resume_sources=(), wandb_id=f'rng-4b-ps-grpo-s11-hero-ext-w{window}-a1')


def prepare(parent_run, output, source_sha256, node_base):
    import torch

    parent_run, output, node_base = Path(parent_run), Path(output), Path(node_base)
    parent = HeroConfig(**read(parent_run / 'resolved_config.json'))
    path = parent_run / 'checkpoints/latest.pt'
    sidecar = read(path.with_suffix('.pt.json'))
    if sidecar['policy_version'] != 192 or sidecar['identity'] != identity(parent):
        raise ValueError('Require the immutable original W1024 full192 snapshot')
    output.mkdir(parents=True, exist_ok=True)
    configs = [extension_config(parent, window=w, output=output / f'arms/w{w}',
        node_dir=node_base / f'w{w}/prepare-control', source_sha256=source_sha256,
        deadline=0.) for w in (1024, 512)]
    plan = dict(format='hero-exploratory-extension-v1', parent_run=str(parent_run),
        parent_policy_version=192, source_sha256=source_sha256, output=str(output),
        target_update=320, first_milestone=256, gpu_hours_reserved=192,
        learning_rate_schedule=dict(initial=4e-7, target=1e-6, linear_updates=8, constant_after=200),
        branches=[asdict(c) for c in configs],
        reference_policy=str(parent_run.parents[1] / 'policies/parent-0128'),
        scope='User-authorized exploratory extension after partial test inspection; old192 study/selection remains frozen',
        storage='Active full states in node RAM; original192 remains durable; versioned pins require verified off-node backup',
        ml='Both arms fork the same W1024 model/Adam/RNG192, same ordered boards and generation RNGs, global128/G8; only W differs')
    plan = json.loads(json.dumps(plan))
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError('Preserve the immutable extension plan')
    else:
        write(plan_path, plan)
    pending = [c for c in configs if not (Path(c.output) / 'fork-receipt.json').exists()]
    if not pending:
        for config in configs:
            validate_resume(config)
        return plan
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state['policy_version'] != 192 or state['identity'] != identity(parent):
        raise ValueError('Full parent state differs from its identity')
    if len(state['model']) not in (398, 399) or len(state['optimizer']['state']) != 398:
        raise ValueError('Preserve all model and Adam tensors')
    if any(int(value['step']) != 192 for value in state['optimizer']['state'].values()):
        raise ValueError('Adam steps must remain192')
    fingerprint = state_fingerprint(training_state(state))
    for config in pending:
        root = Path(config.output)
        root.mkdir(parents=True, exist_ok=True)
        storage = node_base / f'w{config.window}/checkpoints'
        storage.mkdir(parents=True, exist_ok=True)
        link = root / 'checkpoints'
        if not link.is_symlink():
            if link.exists():
                raise ValueError('Do not replace checkpoint storage')
            link.symlink_to(storage, target_is_directory=True)
        if link.resolve() != storage.resolve():
            raise ValueError('Checkpoint storage differs from the plan')
        target = storage / 'latest.pt'
        if target.exists():
            raise ValueError('An incomplete fork needs explicit inspection before resume')
        saved = state | dict(identity=identity(config), resolved_config=asdict(config))
        temporary = storage / 'latest.pt.preparing'
        torch.save(saved, temporary)
        checked = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
        if state_fingerprint(training_state(checked)) != fingerprint:
            raise ValueError('The fork changed model/Adam/RNG state')
        del checked
        temporary.replace(target)
        write(target.with_suffix('.pt.json'), dict(policy_version=192, bytes=target.stat().st_size,
            identity=identity(config), checkpoint_path=str(target), utc=utc()))
        write(root / 'resolved_config.json', asdict(config))
        write(root / 'fork-receipt.json', dict(utc=utc(), parent_policy_version=192,
            parent_checkpoint=str(path), parent_identity=identity(parent), identity=identity(config),
            training_state_sha256=fingerprint, all_model_adam_rng_values_preserved=True,
            checkpoint_storage='node RAM; original parent192 preserved on shared storage',
            plan_sha256=file_hash(plan_path)))
        print(json.dumps(dict(event='FORK_VERIFIED', window=config.window, fingerprint=fingerprint)), flush=True)
    return plan


def pin_checkpoint(config):
    root = Path(config.output)
    version = saved_version(root)
    if version <= 192 or not (root / f'export-catalog/policy-{version:04d}.json').exists():
        return
    metadata = root / f'backup-pins/update-{version:04d}.json'
    if metadata.exists():
        return
    catalog = read(root / f'export-catalog/policy-{version:04d}.json')
    source = root / 'checkpoints/latest.pt'
    before = source.stat()
    expected = catalog['snapshot_stamp']
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (expected['inode'], expected['bytes'], expected['mtime_ns']):
        raise ValueError('Pin only the verified complete snapshot')
    target = source.parent / f'update-{version:04d}.pinned.pt'
    if not target.exists():
        os.link(source, target)
    after = target.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError('Checkpoint changed during pinning')
    write(metadata, dict(utc=utc(), window=config.window, policy_version=version,
        path=str(target.resolve()), bytes=after.st_size, inode=after.st_ino,
        mtime_ns=after.st_mtime_ns, identity=identity(config), durability='NODE_ONLY_BACKUP_PENDING'))


def run(plan_path, gpus, deadline):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    output = Path(plan['output'])
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices) != gpus or os.environ.get('SLURM_JOB_PARTITION') != 'yejin':
        raise ValueError('Use assigned regular yejin GPUs only')
    groups = device_groups(devices)
    session = f'{os.environ["SLURM_JOB_ID"]}-{time.time_ns()}'
    runtime = output / 'runtime' / session
    runtime.mkdir(parents=True)
    configs = [HeroConfig(**row) for row in plan['branches']]
    require_idle_devices(gpus)
    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running, handles = [], []
        try:
            while running or time.time() < deadline - 1200:
                pending = sorted([c for c in configs
                    if c.window not in {item['config'].window for item in running}
                    and time.time() < deadline - 1200 and validate_resume(c) < plan['target_update']],
                    key=lambda c: (saved_version(c.output), -c.window))
                for slot in range(len(groups)):
                    if not pending or any(item['slot'] == slot for item in running):
                        continue
                    config = pending.pop(0)
                    version = saved_version(config.output)
                    target = plan['target_update'] if gpus == 8 else min(plan['target_update'], version + 16)
                    config = replace(config, updates=target, deadline=deadline,
                        node_dir=str(Path(config.node_dir).parent / f'control-{session}-{version}'))
                    path = runtime / f'w{config.window}-from{version}.json'
                    write(path, asdict(config))
                    handle = (Path(config.output) / f'coordinator-{session}.log').open('a')
                    handles.append(handle)
                    process = subprocess.Popen([sys.executable, '-u', '-m',
                        'experiments.ps_concentration.hero_v2', '--config', str(path)],
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(groups[slot])),
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    running.append(dict(config=config, process=process, slot=slot))
                write(output / 'processes.json', dict(utc=utc(), coordinator=os.getpid(),
                    job_id=os.environ['SLURM_JOB_ID'], plan_sha256=file_hash(plan_path),
                    children=[dict(pid=x['process'].pid, window=x['config'].window,
                        devices=groups[x['slot']], config=asdict(x['config'])) for x in running]))
                snapshots = []
                for item in list(running):
                    config, process = item['config'], item['process']
                    observe(config, output, Path(plan['reference_policy']))
                    pin_checkpoint(config)
                    live = Path(config.output) / 'live.json'
                    snapshots.append(dict(window=config.window, pid=process.pid,
                        live=read(live) if live.exists() else None))
                    if process.poll() is not None:
                        if process.returncode:
                            raise RuntimeError(f'W{config.window} exited {process.returncode}; inspect its preserved log')
                        if validate_resume(config) < config.updates:
                            write(output / 'status.json', dict(status='ALLOCATION_HANDOFF', utc=utc()))
                            return
                        running.remove(item)
                write(output / 'live.json', dict(utc=utc(), phase='EXPLORATORY_TRAINING', branches=snapshots,
                    versions={str(c.window): saved_version(c.output) for c in configs}))
                if not running and (time.time() >= deadline - 1200 or
                        all(validate_resume(c) == plan['target_update'] for c in configs)):
                    break
                time.sleep(5)
        finally:
            for item in running:
                if item['process'].poll() is None:
                    os.killpg(item['process'].pid, signal.SIGTERM)
            for item in running:
                item['process'].wait(timeout=330)
            for handle in handles:
                handle.close()
    write(output / 'status.json', dict(utc=utc(), status='EXTENSION_TRAINING_STOPPED',
        versions={str(c.window): saved_version(c.output) for c in configs},
        scope='Exploratory training only; old final evaluation and memory interventions remain pending'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('parent-run', 'output', 'source-sha256', 'node-base'):
        prepare_parser.add_argument('--' + name, required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--plan', required=True)
    run_parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    run_parser.add_argument('--deadline', type=float, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'prepare':
        prepare(**args)
    else:
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupted)
        run(args['plan'], args['gpus'], args['deadline'])


if __name__ == '__main__':
    main()
