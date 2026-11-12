"""Prepare matched continuations without resetting the trained optimizer or RNG."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

from .hero_v2 import HeroConfig, identity, read, write


def branch_config(parent, *, window, output, node_dir, source_sha256, deadline):
    if parent.updates != 128 or parent.window != 1024 or parent.seed != 11:
        raise ValueError('Continue the approved seed11 W1024 parent at update128')
    if window not in (1024, 256):
        raise ValueError('The paired continuation windows are frozen at1024and256')
    result = replace(parent, window=window, gpus=4, updates=192, schedule_updates=128,
        output=str(output), node_dir=str(node_dir), source_sha256=source_sha256,
        deadline=deadline, checkpoint_interval=16, export_interval=16,
        checkpoint_io_lock=str(Path(output).parent / 'checkpoint-io.lock'),
        accepted_resume_sources=(), wandb_id=f'rng-4b-ps-grpo-s11-hero-v2-cont-w{window}-a1')
    before, after = identity(parent), identity(result)
    after.pop('lr_schedule_updates')
    after.update(source_sha256=before['source_sha256'], window=before['window'])
    if after != before or result.seed != parent.seed:
        raise ValueError('Continuation changes more than window and explicit schedule provenance')
    return result


def state_fingerprint(value):
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b'[')
            for element in item:
                visit(element)
            digest.update(b']')
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update((type(item).__name__ + ':' + repr(item) + ';').encode())
        else:
            raise TypeError(f'Unrecognized checkpoint value: {type(item)}')

    visit(value)
    return digest.hexdigest()


def training_state(state):
    return {key: value for key, value in state.items() if key not in ('identity', 'resolved_config')}


def parent_checkpoint(run, *, expected_parameters=4022468096):
    from .hero_finish import ready

    run = Path(run)
    config = HeroConfig(**read(run / 'resolved_config.json'))
    if config.updates != 128 or config.window != 1024:
        raise ValueError('Parent must finish its original128-update W1024 budget')
    ready(run, dict(training_barrier=128, training_identity=identity(config)))
    catalog = read(run / 'export-catalog/policy-0128.json')
    checkpoint = run / 'checkpoints/latest.pt'
    stat = checkpoint.stat()
    expected = catalog['snapshot_stamp']
    if (catalog['status'] != 'VERIFIED' or catalog['policy_version'] != 128
            or catalog['identity'] != identity(config) or catalog['unique_parameters'] != expected_parameters
            or (stat.st_ino, stat.st_size, stat.st_mtime_ns) !=
               (expected['inode'], expected['bytes'], expected['mtime_ns'])):
        raise ValueError('Parent requires the matching full checkpoint and exact BF16 export proof')
    return checkpoint, config


def fork_checkpoint(parent_run, configs, *, expected_parameters=4022468096, checkpoint_storage='shared'):
    import torch

    if len(configs) != 2 or {config.window for config in configs} != {1024, 256}:
        raise ValueError('Fork exactly the paired W1024andW256 branches')
    if checkpoint_storage not in ('shared', 'node'):
        raise ValueError('Choose shared or node-local checkpoint storage explicitly')
    if len({Path(config.output).resolve() for config in configs}) != 2 or len({Path(config.node_dir).resolve() for config in configs}) != 2:
        raise ValueError('Paired branches need disjoint outputs and node directories')
    checkpoint, parent = parent_checkpoint(parent_run, expected_parameters=expected_parameters)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state['policy_version'] != 128 or state['identity'] != identity(parent)
            or HeroConfig(**state['resolved_config']) != parent):
        raise ValueError('Parent state changed after readiness verification')
    fingerprint = state_fingerprint(training_state(state))
    receipts = []
    for config in configs:
        expected = branch_config(parent, window=config.window, output=config.output,
            node_dir=config.node_dir, source_sha256=config.source_sha256, deadline=config.deadline)
        if asdict(config) != asdict(expected):
            raise ValueError('Branch configuration differs from the paired continuation protocol')
        run, node = Path(config.output), Path(config.node_dir).parent
        run.mkdir(parents=True, exist_ok=True)
        link = run / 'checkpoints'
        directory = link if checkpoint_storage == 'shared' else node / 'checkpoints'
        if checkpoint_storage == 'shared':
            if link.is_symlink():
                raise ValueError('Do not silently change existing checkpoint durability')
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != directory.resolve():
                    raise ValueError('Do not replace an existing checkpoint directory')
            else:
                link.symlink_to(directory, target_is_directory=True)
        path = directory / 'latest.pt'
        if path.exists():
            existing = torch.load(path, map_location='cpu', weights_only=True)
            if (existing['policy_version'] != 128 or existing['identity'] != identity(config)
                    or state_fingerprint(training_state(existing)) != fingerprint):
                raise ValueError('Preserve an existing branch state; resume it instead of forking again')
            receipts.append(read(run / 'fork-receipt.json'))
            del existing
            continue
        saved = state | dict(identity=identity(config), resolved_config=asdict(config))
        temporary = path.with_suffix('.pt.preparing')
        torch.save(saved, temporary)
        verified = torch.load(temporary, map_location='cpu', weights_only=True)
        if state_fingerprint(training_state(verified)) != fingerprint:
            raise ValueError('Fork changed model,optimizer,RNG,step or checkpoint structure')
        if verified['identity'] != identity(config):
            raise ValueError('Fork identity differs from its declared continuation')
        del verified
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        write(path.with_suffix('.pt.json'), dict(policy_version=128, bytes=path.stat().st_size,
            identity=identity(config), storage=checkpoint_storage))
        write(run / 'continuation-config.json', asdict(config))
        receipt = dict(parent_checkpoint=str(checkpoint), parent_policy_version=128,
            parent_identity=identity(parent), identity=identity(config), training_state_sha256=fingerprint,
            checkpoint=str(path), bytes=path.stat().st_size,
            checkpoint_storage=checkpoint_storage,
            scope='Exact model,Adam,RNG and step preserved; only declared identity/configuration changes. '
                  'Two learner ranks restore parent RNG ranks0and1; rollout RNG derives from paired episode seeds.')
        write(run / 'fork-receipt.json', receipt)
        receipts.append(receipt)
    if len({row['training_state_sha256'] for row in receipts}) != 1:
        raise ValueError('Paired branches did not receive identical starting training states')
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', required=True)
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--checkpoint-storage', choices=['shared', 'node'], default='shared')
    args = parser.parse_args()
    configs = [HeroConfig(**json.loads(Path(path).read_text())) for path in args.config]
    result = fork_checkpoint(args.parent_run, configs, checkpoint_storage=args.checkpoint_storage)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
