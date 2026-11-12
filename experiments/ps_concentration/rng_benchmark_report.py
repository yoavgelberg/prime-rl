"""Report paired base-model results, preserving resource-censored endpoints."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics

from .rng_metrics import score_prefix


def summarize(root):
    root = Path(root)
    conditions, episodes = {}, []
    for attention in ('full', 'ps'):
        rows = []
        for path in sorted((root / attention).glob('seed-*-result.json')):
            row = json.loads(path.read_text())
            native = json.loads((root / attention / f'seed-{row["seed"]}/game.json').read_text())
            scored = score_prefix(native, 96)
            if not row.get('censored'):
                assert scored['score'] == row['score']
                assert scored['response_count'] == row['response_count']
            row.update(scored)
            row['attention'] = attention
            row['endpoint_type'] = 'context_limit' if row.get('censored') else ('completed' if scored['done'] else 'attempt_limit')
            row['pairs_observed'] = scored['score']
            rows.append(row)
        assert len(rows) == 64 and len({row['seed'] for row in rows}) == 64
        endpoints = [row for row in rows if not row.get('censored')]
        batches = [json.loads(p.read_text()) for p in (root / attention).glob('batch-*-result.json')]
        active_seconds = sum(b['seconds'] for b in batches)
        conditions[attention] = dict(boards=len(rows), censored=len(rows) - len(endpoints),
            mean_pairs_observed=statistics.mean(r['pairs_observed'] for r in rows),
            mean_pairs_uncensored=statistics.mean(r['pairs_observed'] for r in endpoints),
            mean_pairs_96_endpoint_bounds=[sum(r['pairs_observed'] for r in rows) / len(rows),
                (sum(r['pairs_observed'] for r in endpoints) + 8 * (len(rows) - len(endpoints))) / len(rows)],
            mean_attempts_observed=statistics.mean(r['response_count'] for r in rows),
            mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in rows),
            median_per_game_tok_s=statistics.median(r['generated_tokens'] / r['seconds'] for r in rows),
            invalid_actions=sum(r['invalid_actions'] for r in rows), format_errors=sum(r['format_errors'] for r in rows),
            a29_batch_generated_tok_s_per_active_gpu=sum(b['generated_tokens'] for b in batches) / active_seconds,
            a29_active_batch_gpu_hours=active_seconds / 3600,
            a29_batches=len(batches))
        episodes.extend(rows)
    full = {r['seed']: r['pairs_observed'] for r in episodes if r['attention'] == 'full'}
    ps = {r['seed']: r['pairs_observed'] for r in episodes if r['attention'] == 'ps'}
    assert full.keys() == ps.keys()
    differences = [full[seed] - ps[seed] for seed in sorted(full)]
    rng = random.Random(0)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    comparison = dict(mean_difference=statistics.mean(differences), ci95=[draws[250], draws[9750]],
        resamples=10000, estimand='Full minus PS pairs at actual execution stop, includes resource-censored stops',
        uncertainty='Paired board uncertainty for one fixed base model and sampled trajectories; not training-seed variability')
    return dict(conditions=conditions, paired_observed_difference=comparison, episodes=episodes,
                scope='Untrained native-thinking RNG-Bench baseline, not POPGym or an RL result',
                generation='Temperature0.6 topP0.95 topK20; all native thinking retained; 96 attempted responses; common262144 transcript cap')


def write_report(result, output, calibration=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'evaluation-episodes.jsonl').open('w') as handle:
        for row in result['episodes']:
            handle.write(json.dumps(row) + '\n')
    fields = ['attention', 'seed', 'pairs_observed', 'response_count', 'endpoint_type', 'generated_tokens', 'raw_transcript_tokens', 'seconds', 'invalid_actions', 'format_errors']
    with (output / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['episodes'])
    text = ['# Native-thinking Matching Pairs baseline', '', result['scope'], '', result['generation'], '',
            '| Attention | Boards | Observed pairs /8 | Uncensored endpoints | Context-limited | Mean generated tokens |',
            '|---|---:|---:|---:|---:|---:|']
    for arm, m in result['conditions'].items():
        text.append(f'| {arm} | {m["boards"]} | {m["mean_pairs_observed"]:.4f} | {m["boards"]-m["censored"]} | {m["censored"]} | {m["mean_generated_tokens"]:.0f} |')
    ps = result['conditions']['ps']
    text += ['', f'PS uncensored endpoints average {ps["mean_pairs_uncensored"]:.4f}/8. Its missing-endpoint bounds at 96 attempts are {ps["mean_pairs_96_endpoint_bounds"]}; these are bounds, not confidence intervals.',
             '', 'Context-limited runs retain their observed native pairs and are not presented as completed 96-attempt endpoints. No RL or textual-memory conclusion follows from this baseline.',
             '', 'Completed-batch throughput includes batch tails but excludes model setup and old migrated serial episodes. GPU concurrency and generated lengths differ between arms.', '']
    if calibration:
        result['calibration'] = json.loads(Path(calibration).read_text())
        text += ['CPU calibration on 1,000 separate boards:', '']
        for name, m in result['calibration']['policies'].items():
            text.append(f'- {name}: {m["mean_pairs"]:.3f}/8 pairs, {m["mean_attempts"]:.3f} attempted flips on average.')
    (output / 'README.md').write_text('\n'.join(text) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    means = [result['conditions'][arm]['mean_pairs_observed'] for arm in ('full', 'ps')]
    ax.bar(['Full', 'PS W=1024'], means, color=['#2474ad', '#d8752a'])
    low, high = ps['mean_pairs_96_endpoint_bounds']
    ax.vlines(1, low, high, color='black', linestyle='--')
    if calibration:
        ax.axhline(result['calibration']['policies']['random']['mean_pairs'], color='gray', linestyle=':', label='Random legal play')
        ax.legend(frameon=False)
    ax.set(ylim=(0, 8.4), ylabel='Mean pairs found /8', title='Untrained native-thinking model, 64 paired boards')
    fig.text(.5, .01, f'PS: {ps["censored"]} context-limited runs; dashed range bounds their missing endpoints.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'baseline-pairs.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration')
    args = parser.parse_args()
    result = summarize(args.run_dir)
    write_report(result, args.output, args.calibration)
    print(json.dumps({key: value for key, value in result.items() if key not in ('episodes', 'calibration')}, indent=2))


if __name__ == '__main__':
    main()
