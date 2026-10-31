"""Prespecify deferred validation without starting any model or GPU work."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity


def prepare(run, output, policy_dir, *, source_root, source_sha256):
    run, output, policy_dir = map(Path, (run, output, policy_dir))
    if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('Pin the captured evaluation source archive SHA256')
    hero = HeroConfig(**json.loads((run / 'resolved_config.json').read_text()))
    versions = sorted({*range(hero.export_interval, hero.updates + 1, hero.export_interval), hero.updates})
    files, layouts = {}, {}
    for gpus in (4, 8):
        entries = []
        for attention, version in [('full', 0), ('ps', 0), *[('ps', value) for value in versions]]:
            name = 'base-full' if attention == 'full' else ('base-ps' if not version else f'ps-{version:04d}')
            folder = output / f'{gpus}gpu' / name
            config = replace(inference_config(hero, output=folder, gpus=gpus),
                source_sha256=source_sha256, evaluation_concurrency=4 if attention == 'full' else 128 // gpus)
            config_path = folder / 'config.json'
            data = (json.dumps(config.model_dump(), indent=2, allow_nan=False) + '\n').encode()
            files[config_path] = data
            argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
                '--config', str(config_path), '--attention', attention, '--split', 'validation', '--count', '128']
            if version:
                argv.extend(['--checkpoint', str(policy_dir / f'policy-{version:04d}')])
            entries.append(dict(condition=name, attention=attention, policy_version=version, count=128,
                argv=argv, config_sha256=hashlib.sha256(data).hexdigest(),
                summary=str(folder / 'evaluation/validation/summary.json')))
        layouts[str(gpus)] = entries
    plan = dict(status='PREPARED_NOT_EXECUTED', run_dir=str(run), training_barrier=hero.updates,
        training_identity=identity(hero), training_batch_size=hero.batch_size,
        source_root=str(source_root), source_sha256=source_sha256,
        exports=dict(versions=versions, directory=str(policy_dir)), layouts=layouts,
        checkpoint_selection='Highest complete greedy PS validation efficiency reward, earlier update on ties',
        evaluation_reward='pairs/8-attempted_responses/96; training error penalties reported separately',
        guard='Defer all GPU evaluation until training completes and its workers exit. '
            'Verify policy export hashes and preserve candidates before releasing the training allocation.',
        scope='One PS hero seed. Matching full-context and PS base controls use the identical thinking/action controller. '
            'This plan launches no model. Final test seeds cannot select a checkpoint.')
    files[output / 'plan.json'] = (json.dumps(plan, indent=2, allow_nan=False) + '\n').encode()
    if any(path.exists() and path.read_bytes() != data for path, data in files.items()):
        raise ValueError('Preserve the existing validation plan; use a new directory for a revised plan')
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy-dir', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    plan = prepare(args.run_dir, args.output, args.policy_dir,
        source_root=args.source_root, source_sha256=args.source_sha256)
    print(json.dumps(dict(status=plan['status'], barrier=plan['training_barrier'],
        policy_versions=plan['exports']['versions'], layouts=sorted(plan['layouts']))))


if __name__ == '__main__':
    main()
