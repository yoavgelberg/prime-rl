"""Run the frozen parent sweep and matched four-GPU continuations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .hero_continue import branch_config, fork_checkpoint, parent_checkpoint
from .hero_exports import file_hash, verify as verify_export
from .hero_finish import execute_conditions, require_idle_devices
from .hero_policy_archive import preserve as preserve_policy
from .hero_telemetry import preserve as preserve_telemetry
from .hero_v2 import HeroConfig, identity, read, utc, write, validate_resume_identity


def device_groups(devices):
    if len(set(devices)) != len(devices):
        raise ValueError('Assigned GPU identities must be distinct')
    if len(devices) == 8:
        return [devices[:4], devices[4:]]
    if len(devices) == 4:
        return [devices]
    raise ValueError('The paired experiment uses an assigned four-or-eight-B200 allocation')


def saved_version(run):
    path = Path(run) / 'checkpoints/latest.pt.json'
    return read(path)['policy_version'] if path.exists() else 0


def validate_resume(config):
    run = Path(config.output)
    sidecar = read(run / 'checkpoints/latest.pt.json')
    if sidecar['identity'] != identity(config):
        validate_resume_identity(sidecar['identity'], config, policy_version=sidecar['policy_version'])
    if (run / 'state.json').exists() and read(run / 'state.json')['policy_version'] != sidecar['policy_version']:
        raise ValueError('Latest committed branch update lacks its exact optimizer snapshot; preserve and diagnose')
    if (run / 'checkpoints/latest.pt').stat().st_size != sidecar['bytes']:
        raise ValueError('Committed branch snapshot has a different size')
    return sidecar['policy_version']


def observe(config, output, reference):
    run = Path(config.output)
    if (run / 'state.json').exists():
        telemetry = run / 'actor-telemetry/manifest.json'
        if not telemetry.exists() or read(telemetry)['committed_update'] != read(run / 'state.json')['policy_version']:
            preserve_telemetry(run)
    checkpoint = run / 'checkpoints/latest.pt'
    sidecar = checkpoint.with_suffix('.pt.json')
    if not sidecar.exists():
        return
    version = read(sidecar)['policy_version']
    if version <= 128:
        return
    export = Path(config.node_dir).parent / 'candidate-policies' / f'policy-{version:04d}'
    experiment = export / 'experiment.json'
    if not experiment.exists() or read(experiment)['policy_version'] != version:
        return
    catalog = run / f'export-catalog/policy-{version:04d}.json'
    if not catalog.exists():
        verify_export(checkpoint, export, catalog)
    receipt = read(catalog)
    if receipt['identity'] != identity(config):
        raise ValueError('Inference export differs from its branch')
    target = Path(output) / 'policy-archives' / f'w{config.window}-{version:04d}'
    if not target.exists():
        preserve_policy(receipt, target, reference=reference)


def runtime_configs(parent, plan, node_base, deadline):
    return [branch_config(parent, window=row['window'], output=row['output'],
        node_dir=Path(node_base) / row['name'] / 'control',
        source_sha256=plan['source_sha256'], deadline=deadline) for row in plan['branches']]


def wait_for_storage(output, required_gib, deadline):
    while shutil.disk_usage(output).free < required_gib * 1024**3:
        write(Path(output) / 'live.json', dict(utc=utc(), phase='AWAITING_SHARED_STORAGE',
            required_gib=required_gib, available_gib=shutil.disk_usage(output).free / 1024**3))
        if time.time() > deadline - 1200:
            raise TimeoutError('Insufficient durable checkpoint storage before allocation handoff')
        time.sleep(10)


def required_fork_storage(plan, configs):
    output = Path(plan['output'])
    archived = sum(path.stat().st_size for folder in (output / 'policies', output / 'policy-archives')
        if folder.exists() for path in folder.rglob('*') if path.is_file())
    initialized = sum((Path(config.output) / 'checkpoints/latest.pt').stat().st_size for config in configs
        if (Path(config.output) / 'fork-receipt.json').exists()
        and (Path(config.output) / 'checkpoints/latest.pt').is_file())
    return max(0., plan['storage']['required_shared_headroom_gib'] - (archived + initialized) / 1024**3)


def wait_for_export(parent_run, deadline):
    catalog = Path(parent_run) / 'export-catalog/policy-0128.json'
    started = time.monotonic()
    while not catalog.exists():
        if time.monotonic() - started > 600 or time.time() > deadline - 1200:
            raise TimeoutError('Parent128 export verifier has not completed; inspect its existing handle')
        time.sleep(2)


def preserve_parent(plan, reference):
    run, output = Path(plan['parent_run']), Path(plan['output'])
    for version in range(16, 128, 16):
        target = output / 'policy-archives' / f'parent-{version:04d}'
        if not target.exists():
            preserve_policy(read(run / f'export-catalog/policy-{version:04d}.json'), target, reference=reference)


def run(plan_path, *, gpus, node_base, deadline):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    output, parent_run = Path(plan['output']), Path(plan['parent_run'])
    output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value.strip()]
    if len(devices) != gpus or os.environ.get('SLURM_JOB_PARTITION') != 'yejin':
        raise ValueError('Use only the assigned regular-yejin B200s')
    groups = device_groups(devices)
    with (parent_run.parent / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_for_export(parent_run, deadline)
        _, parent = parent_checkpoint(parent_run)
        if identity(parent) != plan['parent_identity']:
            raise ValueError('Parent differs from the frozen paired plan')
        require_idle_devices(gpus)
        reference = Path(plan['policy_directory']) / 'parent-0128'
        preserve_policy(read(parent_run / 'export-catalog/policy-0128.json'), reference)
        configs = runtime_configs(parent, plan, node_base, deadline)
        session_path = output / 'session.json'
        if session_path.exists():
            session = read(session_path)
            if session['plan_sha256'] != file_hash(plan_path):
                raise ValueError('Resume the same frozen paired plan')
        else:
            session = dict(plan=str(plan_path), plan_sha256=file_hash(plan_path), sweep_layout=gpus)
            write(session_path, session)
        with ThreadPoolExecutor(max_workers=2) as workers:
            archive_future = workers.submit(preserve_parent, plan, reference)
            missing = [config for config in configs if not (Path(config.output) / 'fork-receipt.json').exists()]
            fork_future = None
            if missing and shutil.disk_usage(output).free >= required_fork_storage(plan, configs) * 1024**3:
                fork_future = workers.submit(fork_checkpoint, parent_run, configs)
            execute_conditions(plan['initial_sweep'][str(session['sweep_layout'])],
                output / 'initial-execution', gpus=gpus)
            archive_future.result()
            if missing:
                if fork_future is None:
                    wait_for_storage(output, required_fork_storage(plan, configs), deadline)
                    fork_future = workers.submit(fork_checkpoint, parent_run, configs)
                fork_future.result()
        require_idle_devices(gpus)
        pending = [config for config in configs if validate_resume(config) < 192]
        running, completed, handles = [], [], []
        runtime = output / 'runtime' / f'{os.environ["SLURM_JOB_ID"]}-{time.time_ns()}'
        runtime.mkdir(parents=True)
        try:
            while pending or running:
                while pending and len(running) < len(groups):
                    config = pending.pop(0)
                    used = {item['slot'] for item in running}
                    slot = next(index for index in range(len(groups)) if index not in used)
                    path = runtime / f'w{config.window}.json'
                    write(path, asdict(config))
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(groups[slot]))
                    handle = (Path(config.output) / f'coordinator-{os.environ["SLURM_JOB_ID"]}.log').open('a')
                    handles.append(handle)
                    argv = [sys.executable, '-u', '-m', 'experiments.ps_concentration.hero_v2', '--config', str(path)]
                    process = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    running.append(dict(config=config, process=process, slot=slot))
                    write(output / 'processes.json', dict(utc=utc(), coordinator=os.getpid(),
                        children=[dict(pid=item['process'].pid, window=item['config'].window,
                            config=asdict(item['config']), devices=groups[item['slot']]) for item in running]))
                snapshots = []
                for item in list(running):
                    config, process = item['config'], item['process']
                    observe(config, output, reference)
                    live = Path(config.output) / 'live.json'
                    snapshots.append(dict(window=config.window, pid=process.pid, exit_code=process.poll(),
                        live=read(live) if live.exists() else None))
                    if process.poll() is not None:
                        observe(config, output, reference)
                        completed.append(dict(window=config.window, exit_code=process.returncode,
                            version=saved_version(config.output)))
                        running.remove(item)
                write(output / 'live.json', dict(utc=utc(), phase='PAIRED_TRAINING', branches=snapshots,
                    pending_windows=[config.window for config in pending], completed=completed))
                if running:
                    time.sleep(5)
        finally:
            for handle in handles:
                handle.close()
        failed = [row for row in completed if row['exit_code'] != 0]
        versions = {str(config.window): saved_version(config.output) for config in configs}
        status = ('FAILED_BRANCH_PRESERVED' if failed else
                  'CONTINUATIONS_COMPLETE_FINAL_EVAL_PENDING' if all(v == 192 for v in versions.values())
                  else 'CONTINUATION_HANDOFF_PENDING')
        result = dict(utc=utc(), status=status, versions=versions, completed=completed,
            plan_sha256=file_hash(plan_path), scope='Training and archive stage only; final comparisons/interventions remain pending')
        write(output / 'status.json', result)
        if failed:
            raise RuntimeError('A continuation failed; inspect preserved branch logs and exact live handles')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpus', type=int, choices=[4, 8], required=True)
    parser.add_argument('--node-base', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, gpus=args.gpus, node_base=args.node_base, deadline=args.deadline), indent=2))


if __name__ == '__main__':
    main()
