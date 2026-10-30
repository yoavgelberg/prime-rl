"""Produce native Matching Pairs tables and figures from saved episode records."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .overlap_config import episode_metrics
from .rng_memory_metrics import ANALYSIS_VERSION, known_match_decisions, summarize
from .rng_training_config import outcome_reward


def paired_bootstrap(treatment, control, *, seed=0, resamples=10000):
    left = {row['board_seed']: row['training_return'] for row in treatment}
    right = {row['board_seed']: row['training_return'] for row in control}
    if len(left) != len(treatment) or len(right) != len(control) or left.keys() != right.keys():
        raise ValueError('A paired comparison requires exactly one outcome on each identical board seed')
    differences = np.array([left[board] - right[board] for board in sorted(left)], dtype=np.float64)
    if not len(differences):
        raise ValueError('Cannot bootstrap an empty board set')
    rng = np.random.default_rng(seed)
    means = np.concatenate([rng.choice(differences, (min(256, resamples - start), len(differences)))
                            .mean(axis=1) for start in range(0, resamples, 256)])
    return dict(mean_difference=float(differences.mean()),
        ci95=[float(value) for value in np.quantile(means, [.025, .975])],
        boards=len(differences), resamples=resamples,
        uncertainty='Paired uncertainty over board seeds within one fixed training seed; not training variability')


def seed_summary(rows):
    seeds = [row['training_seed'] for row in rows]
    if len(set(seeds)) != len(seeds) or any(seed is None for seed in seeds):
        raise ValueError('Training variability requires distinct, identified independent training runs')
    means = np.array([row['mean_return'] for row in rows])
    return dict(training_seeds=seeds, independent_training_runs=len(seeds),
        mean_return=float(means.mean()),
        training_seed_std=float(means.std(ddof=1)) if len(means) > 1 else None,
        scope='Equal weighting of per-training-seed means; boards are not independent training runs')


def load_condition(entry, *, allow_partial=False):
    rows = [json.loads(line) for line in Path(entry['episodes']).read_text().splitlines() if line]
    seeds = [row['board_seed'] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f'Duplicate board in {entry["name"]}')
    expected = entry['expected_boards']
    if not rows or len(rows) > expected or (not allow_partial and len(rows) != expected):
        raise ValueError(f'{entry["name"]}: {len(rows)} episodes, expected {expected}; partial reports require --allow-partial')
    if entry.get('greedy') is not True or any(row.get('greedy') is not True for row in rows):
        raise ValueError('The final comparison requires greedy evaluation, never sampled training trajectories')
    for row in rows:
        actual = outcome_reward(row, row.get('max_responses', 96), -1.)
        if abs(actual - row['training_return']) > 1e-10:
            raise ValueError(f'Recorded efficiency reward differs from pairs/8-attempts/96: {row["episode_id"]}')
        if row.get('attention') != entry['attention'] or row.get('window') != entry['window']:
            raise ValueError('An evaluation condition mixes attention settings')
    metrics = episode_metrics(rows)
    metrics['mean_attempts_to_completion'] = float(np.mean([
        row['response_count'] if row['done'] else row.get('max_responses', 96) for row in rows]))
    return dict(entry, **metrics, episodes_file=entry['episodes'], complete_split=len(rows) == expected, rows=rows)


def analyze_condition(condition, tokenizer, *, examples=None):
    decisions = []
    example_count = 0
    for row in sorted(condition['rows'], key=lambda value: value['board_seed']):
        token_path = Path(row['token_record'])
        if hashlib.sha256(token_path.read_bytes()).hexdigest() != row['token_record_sha256']:
            raise ValueError(f'Token checksum failed: {token_path}')
        cached = token_path.parent / 'memory-metrics.json'
        memory = None
        if cached.exists():
            memory = json.loads(cached.read_text())
            if memory.get('token_record_sha256') != row['token_record_sha256'] or memory.get('analysis_version') != ANALYSIS_VERSION:
                memory = None
        if memory is None:
            with gzip.open(token_path, 'rt') as handle:
                record = json.load(handle)
            game_path = record['native_game_json']
            game = json.loads(Path(game_path).read_text())
            values = known_match_decisions(record, game, tokenizer)
            memory = dict(token_record_sha256=row['token_record_sha256'], analysis_version=ANALYSIS_VERSION, decisions=values,
                known_match=summarize(values), scope='Latest direct environment reveal at original action-query position; notes do not reset age')
            cached.write_text(json.dumps(memory, indent=2) + '\n')
        decisions.extend(memory['decisions'])
        if examples is not None and example_count < 20:
            from .rng_trace import export
            with gzip.open(token_path, 'rt') as handle:
                game_path = json.load(handle)['native_game_json']
            export(token_path, game_path, tokenizer, Path(examples) / f'board-{row["board_seed"]}')
            example_count += 1
    return dict(known_match=summarize(decisions), example_transcripts=example_count,
        decision_scope='Completed native responses, including invalid attempts; unfinished resource-censored responses excluded')


def summarize_manifest(manifest, *, allow_partial=False):
    conditions = [load_condition(entry, allow_partial=allow_partial) for entry in manifest['conditions']]
    named = {condition['name']: condition for condition in conditions}
    if len(named) != len(conditions):
        raise ValueError('Condition names must be unique')
    comparisons = []
    for entry in manifest.get('comparisons', []):
        trained, sliding, full = [named[entry[key]] for key in ('treatment', 'sliding_control', 'full_control')]
        if trained['attention'] != 'ps' or sliding['attention'] != 'ps' or full['attention'] != 'full':
            raise ValueError('Memory recovery compares two PS evaluations against a full-context reference')
        if trained['window'] != sliding['window']:
            raise ValueError('Treatment and sliding control must have the same window')
        paired_bootstrap(full['rows'], sliding['rows'])
        result = paired_bootstrap(trained['rows'], sliding['rows'])
        gap = full['mean_return'] - sliding['mean_return']
        result.update(entry, training_seed=trained['training_seed'],
            treatment_return=trained['mean_return'], sliding_control_return=sliding['mean_return'],
            full_control_return=full['mean_return'], full_context_gap=gap,
            recovery=result['mean_difference'] / gap if gap >= .10 else None,
            control_policy=sliding['policy'],
            full_rl_control_skipped=sliding['policy'] == 'base')
        comparisons.append(result)
    groups = {}
    for condition in conditions:
        if condition['training_seed'] is not None:
            key = (condition['policy'], condition['attention'], condition['window'], condition['split'])
            groups.setdefault(key, []).append(condition)
    aggregate = [dict(policy=key[0], attention=key[1], window=key[2], split=key[3], **seed_summary(rows))
                 for key, rows in groups.items()]
    return dict(task='Native RNG-Bench textual Matching Pairs 4x4; efficiency reward pairs/8-attempts/96',
        conditions=conditions, paired_comparisons=comparisons, training_seed_summaries=aggregate,
        full_rl_was_required=manifest.get('full_rl_was_required', False),
        interpretation='Base-control comparisons use an untrained checkpoint and do not claim a trained Full RL comparison')


def save_tables(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ['name', 'policy', 'training_seed', 'split', 'attention', 'window', 'episodes', 'complete_split',
        'mean_return', 'mean_pairs', 'matched_fraction', 'completion_rate', 'mean_attempts_to_completion',
        'completed_only_mean_attempts', 'completed_only_denominator', 'mean_attempts', 'format_errors',
        'native_invalid_actions', 'resource_terminations', 'generated_tokens', 'raw_transcript_tokens',
        'mean_per_game_tokens_per_second']
    with (output / 'main-comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['conditions'])
    clean = dict(result, conditions=[{key: value for key, value in row.items() if key != 'rows'}
                                    for row in result['conditions']])
    (output / 'summary.json').write_text(json.dumps(clean, indent=2, allow_nan=False) + '\n')
    lines = ['# Matching Pairs memory experiment', '', result['task'], '',
        '| Condition | Training seed | Attention | Boards | Pairs /8 | Efficiency return | Attempts | Context limited |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in result['conditions']:
        attention = f'PS W={row["window"]}' if row['attention'] == 'ps' else 'Full'
        boards = str(row['episodes']) + (' (partial)' if not row['complete_split'] else '')
        lines.append(f'| {row["name"]} | {row["training_seed"]} | {attention} | {boards} | '
            f'{row["mean_pairs"]:.4f} | {row["mean_return"]:.4f} | {row["mean_attempts"]:.2f} | {row["resource_terminations"]} |')
    lines += ['', 'Resource-limited episodes retain observed pairs but receive the separately recorded -1 training reward. Their unobserved 96-attempt endpoints are not imputed as measured outcomes.', '',
              result['interpretation'], '',
              'Paired intervals below use 10,000 resamples over board seeds within a fixed training seed. Training-seed standard deviations describe independent training runs; episode counts are not training replication.', '']
    for row in result['paired_comparisons']:
        lines.append(f'- {row["treatment"]} minus {row["sliding_control"]}: {row["mean_difference"]:.4f}, '
            f'95% board-bootstrap interval {row["ci95"]}; full-context gap {row["full_context_gap"]:.4f}; recovery {row["recovery"]}.')
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def plots(result, manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    if manifest.get('training_runs'):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for run in manifest['training_runs']:
            root = Path(run['run_dir'])
            points = [json.loads(path.read_text()) for path in sorted((root / 'validation').glob('update-*/summary.json'))]
            if points:
                ax.plot([point['policy_version'] * 64 for point in points], [point['mean_return'] for point in points],
                    marker='o', label=f'PS GRPO seed {run["training_seed"]}')
                plotted = True
        if plotted:
            ax.set(xlabel='Training episodes', ylabel='Greedy validation efficiency return')
            ax.legend(frameon=False)
            fig.tight_layout(); fig.savefig(output / 'return-vs-training-episodes.png', dpi=180)
        plt.close(fig)
    sliding = [row for row in result['conditions'] if row['attention'] == 'ps' and row['split'] == 'test']
    series = {}
    for row in sliding:
        series.setdefault((row['policy'], row['training_seed']), []).append(row)
    if sliding:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (policy, seed), rows in series.items():
            rows.sort(key=lambda row: row['window'])
            ax.plot([row['window'] for row in rows], [row['mean_return'] for row in rows], marker='o',
                label=policy if seed is None else f'{policy} seed {seed}')
        ax.set(xlabel='Rolling window W (tokens)', ylabel='Greedy test efficiency return', xscale='log')
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output / 'final-return-vs-window.png', dpi=180); plt.close(fig)
    memory = [row for row in result['conditions'] if row.get('known_match') is not None]
    if memory:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = ['<=W', '(W,2W]', '>2W']
        for row in memory:
            ax.plot(bins, [row['known_match'][name]['utilization'] for name in bins], marker='o', label=row['name'])
        ax.set(ylim=(0, 1.05), xlabel='Age of latest direct environment reveal', ylabel='Known-match utilization')
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output / 'known-match-utilization-vs-age.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--tokenizer', help='Add offline reveal-age diagnostics with the original pinned tokenizer')
    parser.add_argument('--examples-condition', help='Export the first 20 fixed-seed-order trajectories of this PS RL condition')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    result = summarize_manifest(manifest, allow_partial=args.allow_partial)
    if args.examples_condition and not args.tokenizer:
        parser.error('--examples-condition requires --tokenizer')
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for condition in result['conditions']:
            examples = Path(args.output) / 'examples' if condition['name'] == args.examples_condition else None
            if examples is not None and (condition['attention'] != 'ps' or condition['training_seed'] is None):
                raise ValueError('The 20 memory examples must be from a trained PS policy')
            condition.update(analyze_condition(condition, tokenizer, examples=examples))
    save_tables(result, args.output)
    plots(result, manifest, args.output)
    print(json.dumps(dict(conditions=len(result['conditions']), comparisons=len(result['paired_comparisons']),
                         output=args.output)), flush=True)


if __name__ == '__main__':
    main()
