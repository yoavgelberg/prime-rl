import gzip
import json

import numpy as np


def read_episodes(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize an empty evaluation')
    completed = [r['flips'] for r in records if r['completed']]
    turns = sum(r['flips'] for r in records)
    age_bins = {name: [0, 0] for name in ('<=W', '(W,2W]', '>2W')}
    for rec in records:
        for decision in rec.get('known_match', []):
            # Bin a decision by its most recently directly revealed available partner.
            age = min(decision['partners'].values())
            name = '<=W' if age <= rec['window'] else '(W,2W]' if age <= 2*rec['window'] else '>2W'
            age_bins[name][0] += int(decision['correct'])
            age_bins[name][1] += 1
    return dict(episodes=len(records), native_return=float(np.mean([r['native_return'] for r in records])),
                matched_fraction=float(np.mean([r['matched_fraction'] for r in records])),
                completion_rate=float(np.mean([r['completed'] for r in records])),
                flips=float(np.mean([r['flips'] if r['completed'] else 2*(104 if r.get('cards') == 104 else 52)
                                     for r in records])),
                completed_only_flips=float(np.mean(completed)) if completed else None,
                completed_only_denominator=len(completed),
                format_error_rate=sum(r['format_errors'] for r in records)/turns,
                native_invalid_actions=sum(r['invalid_actions'] for r in records),
                generated_tokens=sum(r['generated_tokens'] for r in records),
                raw_transcript_tokens=sum(r['raw_transcript_tokens'] for r in records),
                peak_kv_bytes=max(r['peak_kv_bytes'] for r in records),
                total_inference_seconds=sum(r['inference_seconds'] for r in records),
                known_match={name: dict(correct=v[0], opportunities=v[1], utilization=v[0]/v[1] if v[1] else None)
                             for name, v in age_bins.items()})


def paired_bootstrap(left, right, seed=0, resamples=10000):
    a = {r['board_seed']: r['native_return'] for r in left}
    b = {r['board_seed']: r['native_return'] for r in right}
    assert a.keys() == b.keys()
    differences = np.array([a[s] - b[s] for s in sorted(a)])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(resamples, len(differences)))].mean(-1)
    return dict(difference=float(differences.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                uncertainty='paired uncertainty over boards within one training seed', resamples=resamples)
