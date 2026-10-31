"""Preserve completed actor reports from node-local storage without a model pass."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def preserve(run):
    run = Path(run)
    state = read(run / 'state.json')
    output = run / 'actor-telemetry'
    updates = []
    for update in range(1, state['policy_version'] + 1):
        metric_path = run / f'update-metrics/update-{update:04d}.json'
        metric_data = metric_path.read_bytes()
        point = json.loads(metric_data)
        metrics = point['metrics']
        if metrics['update'] != update or len(set(point['episode_ids'])) != metrics['episodes']:
            raise ValueError('Committed update has inconsistent episode identities')
        node = Path(point['actor_weights']['path']).parent
        directory = output / f'update-{update:04d}'
        files, rows, devices, durations = [], [], [], []
        for rank in range(len(point['steps'])):
            source = node / f'actor-{rank}/finished-{update:04d}.json'
            target = directory / f'actor-{rank}.json'
            if source.exists():
                data = source.read_bytes()
                if target.exists() and target.read_bytes() != data:
                    raise ValueError('Preserved actor report differs from its node-local source')
            else:
                data = target.read_bytes()
            report = json.loads(data)
            episodes = report['episodes']
            if any(row['policy_version'] != update - 1 or row['actor_rank'] != rank for row in episodes):
                raise ValueError('Actor report belongs to another collection policy or actor')
            rows.extend(episodes)
            devices.extend(dict(actor_rank=rank, **device) for device in report.get('devices', []))
            durations.append(dict(actor_rank=rank, seconds=report['seconds'],
                reused_episodes=report.get('reused_episodes', [])))
            files.append(dict(source=str(source), path=str(target), sha256=digest(data), bytes=len(data)))
            directory.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix('.json.tmp')
                temporary.write_bytes(data)
                temporary.replace(target)
        if len(rows) != metrics['episodes'] or sorted(row['episode_id'] for row in rows) != sorted(point['episode_ids']):
            raise ValueError('Actor reports do not cover the committed batch exactly once')
        for field in ('generated_tokens', 'raw_transcript_tokens'):
            if sum(row[field] for row in rows) != metrics[field]:
                raise ValueError(f'Actor {field} disagree with committed update metrics')
        updates.append(dict(update=update, episodes=len(rows), devices=devices, actor_durations=durations,
            metric_path=str(metric_path), metric_sha256=digest(metric_data), files=files))
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), committed_update=state['policy_version'],
        updates=updates, scope='Complete actor reports only. KV pool bytes are shared allocated capacity, '
        'not per-game retained KV or a measurement of peak live logical KV.')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    temporary.replace(output / 'manifest.json')
    return dict(committed_update=state['policy_version'], actor_reports=sum(len(x['files']) for x in updates),
        episodes=sum(x['episodes'] for x in updates), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(preserve(args.run_dir)))


if __name__ == '__main__':
    main()
