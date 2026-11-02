"""Produce native Matching Pairs tables and figures from saved episode records."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .overlap_config import BoardProtocol, RunConfig, SPLITS, board_protocol, episode_metrics
from .rng_memory_metrics import ANALYSIS_VERSION, known_match_decisions, summarize
from .rng_training_config import outcome_reward
from .hero_protocol import batch_size, is_hero, training_identity


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
    metrics = {}
    names = ('mean_return', 'mean_pairs', 'matched_fraction', 'completion_rate', 'mean_attempts_to_completion',
        'completed_only_mean_attempts', 'completed_only_denominator', 'mean_attempts', 'format_errors',
        'format_error_rate', 'native_invalid_actions', 'resource_terminations', 'generated_tokens',
        'raw_transcript_tokens', 'mean_per_game_tokens_per_second', 'peak_kv_pool_allocated_bytes',
        'completed_cohort_inference_seconds', 'completed_cohort_gpu_hours',
        'known_match_recent_utilization', 'known_match_middle_utilization', 'known_match_old_utilization',
        'known_match_recent_opportunities', 'known_match_middle_opportunities', 'known_match_old_opportunities')
    for name in names:
        values = np.array([row[name] for row in rows if row.get(name) is not None], dtype=float)
        if any(name in row for row in rows):
            metrics[name] = dict(mean=float(values.mean()) if len(values) else None,
                sample_std=float(values.std(ddof=1)) if len(values) > 1 else None,
                defined_training_runs=len(values))
    return dict(training_seeds=seeds, independent_training_runs=len(seeds),
        mean_return=float(means.mean()),
        training_seed_std=float(means.std(ddof=1)) if len(means) > 1 else None,
        metrics=metrics,
        scope='Equal weighting of per-training-seed means; boards are not independent training runs')


def load_condition(entry, *, allow_partial=False):
    rows = [json.loads(line) for line in Path(entry['episodes']).read_text().splitlines() if line]
    seeds = [row['board_seed'] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f'Duplicate board in {entry["name"]}')
    expected = entry['expected_boards']
    if not rows or len(rows) > expected or (not allow_partial and len(rows) != expected):
        raise ValueError(f'{entry["name"]}: {len(rows)} episodes, expected {expected}; partial reports require --allow-partial')
    start, limit = SPLITS[entry['split']]
    reserved = set(range(start, start + expected))
    if expected > limit or not set(seeds) <= reserved or (len(rows) == expected and set(seeds) != reserved):
        raise ValueError('An evaluation condition must use its fixed reserved board seeds')
    if entry.get('greedy') is not True or any(row.get('greedy') is not True for row in rows):
        raise ValueError('The final comparison requires greedy evaluation, never sampled training trajectories')
    board, standard = board_protocol(entry['split']).identity(), BoardProtocol().identity()
    for row in rows:
        if any(key in entry and row.get(key) != entry[key] for key in ('policy_version', 'policy_identity')):
            raise ValueError('An evaluation condition contains a different checkpoint identity')
        if any(key not in row or row[key] != value for key, value in entry.get('controller', {}).items()):
            raise ValueError('An evaluation condition contains a different generation controller')
        if any(row.get(key, standard[key]) != value for key, value in board.items()):
            raise ValueError('An evaluation condition mixes board sizes or attempt limits')
        actual = outcome_reward(row, row.get('max_responses', 96), -1.)
        if abs(actual - row['training_return']) > 1e-10:
            raise ValueError(f'Recorded efficiency reward differs from pairs/total_pairs-attempts/attempt_cap: {row["episode_id"]}')
        if row.get('attention') != entry['attention'] or row.get('window') != entry['window']:
            raise ValueError('An evaluation condition mixes attention settings')
    metrics = episode_metrics(rows)
    metrics['mean_attempts_to_completion'] = float(np.mean([
        row['response_count'] if row['done'] else row.get('max_responses', 96) for row in rows]))
    if entry.get('summary'):
        summary = json.loads(Path(entry['summary']).read_text())
        if any(key in entry and summary.get(key) != entry[key] for key in ('policy_version', 'policy_identity')):
            raise ValueError('An evaluation summary belongs to a different checkpoint identity')
        if (summary['episodes'] != len(rows) or summary['attention'] != entry['attention']
                or summary['window'] != entry['window']
                or any(summary.get(key, standard[key]) != value for key, value in board.items())):
            raise ValueError('System-metric summary belongs to a different evaluation condition')
        for name in ('peak_kv_pool_allocated_bytes', 'completed_cohort_inference_seconds',
                     'completed_cohort_gpu_hours', 'unfinished_cohort_accounting'):
            metrics[name] = summary.get(name)
    return dict(entry, **metrics, **{key: value for key, value in board.items() if key not in metrics},
        board_seeds_sha256=hashlib.sha256(json.dumps(sorted(seeds)).encode()).hexdigest(),
        evaluation_scope='window_sweep' if entry['split'] in ('test', 'extension_test') and expected == 128 else entry['split'],
        episodes_file=entry['episodes'], complete_split=len(rows) == expected, rows=rows)


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
            with (gzip.open if Path(game_path).suffix == '.gz' else open)(Path(game_path), 'rt') as handle:
                game = json.load(handle)
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
    known = summarize(decisions)
    flattened = {}
    for label, age in [('recent', '<=W'), ('middle', '(W,2W]'), ('old', '>2W')]:
        flattened[f'known_match_{label}_utilization'] = known[age]['utilization']
        flattened[f'known_match_{label}_opportunities'] = known[age]['opportunities']
    return dict(known_match=known, **flattened, example_transcripts=example_count,
        decision_scope='Completed native responses, including invalid attempts; unfinished resource-censored responses excluded')


def aggregate_conditions(conditions):
    groups = {}
    for condition in conditions:
        if condition['training_seed'] is not None:
            key = (condition['policy'], condition['attention'], condition['window'], condition['split'],
                   condition['expected_boards'], condition['board_seeds_sha256'])
            groups.setdefault(key, []).append(condition)
    return [dict(policy=key[0], attention=key[1], window=key[2], split=key[3],
                 boards_per_training_seed=key[4], board_seeds_sha256=key[5],
                 complete_split=all(row['complete_split'] for row in rows), **seed_summary(rows))
            for key, rows in groups.items()]


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
        if any(condition['split'] != 'test' for condition in (trained, sliding, full)):
            raise ValueError('The primary memory-recovery comparison uses the held-out standard board test split')
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
    aggregate = aggregate_conditions(conditions)
    interventions = []
    for entry in manifest.get('interventions', []):
        summary = json.loads(Path(entry['summary']).read_text())
        interventions.append(dict(entry, result=summary))
    accounting = json.loads(Path(manifest['allocation_accounting']).read_text()) if manifest.get('allocation_accounting') else None
    qualification = None
    if manifest.get('checkpoint_selection'):
        source = manifest['checkpoint_selection']
        data = Path(source['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('The validation selection receipt changed after final evaluation preparation')
        selected = json.loads(data)
        if selected['split'] != 'validation' or selected['attention'] != 'ps' or selected['episodes'] != 128:
            raise ValueError('Deferred qualification requires the complete PS validation selection')
        full = selected['full_context_matched_fraction']
        gap = selected['full_minus_base_ps_matched_fraction']
        status = 'GO' if full >= .8 and gap >= .1 else 'HOLD'
        if selected['competence_and_forgetting_status'] != status:
            raise ValueError('Deferred qualification status disagrees with its exact gate values')
        qualification = dict(status=status, full_context_matched_fraction=full,
            full_minus_base_ps_matched_fraction=gap, competence_threshold=.8, forgetting_threshold=.1,
            selected_update=selected['update'], selected_validation_return=selected['validation_return'],
            episodes=128, split='validation', selection_receipt=source,
            scope='Deferred original-policy competence and forgetting qualification; not a final-test success claim')
    return dict(task='Native RNG-Bench textual Matching Pairs 4x4; efficiency reward pairs/8-attempts/96. '
        'Size generalization: parameter-scaled 4x8, eight ranks with four copies each, reward pairs/16-attempts/192.',
        conditions=conditions, paired_comparisons=comparisons, training_seed_summaries=aggregate,
        interventions=interventions, allocation_accounting=accounting, deferred_qualification=qualification,
        accounting_scope=manifest.get('accounting_scope', 'Explicit allocations supplied in the accounting artifact'),
        full_rl_was_required=manifest.get('full_rl_was_required', False),
        interpretation='Base-control comparisons use an untrained checkpoint and do not claim a trained Full RL comparison')


def save_tables(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ['name', 'policy', 'training_seed', 'split', 'evaluation_scope', 'attention', 'window', 'board_rows', 'board_cols',
        'distinct_ranks', 'total_pairs', 'max_responses', 'episodes', 'expected_boards', 'board_seeds_sha256', 'complete_split',
        'mean_return', 'mean_pairs', 'matched_fraction', 'completion_rate', 'mean_attempts_to_completion',
        'completed_only_mean_attempts', 'completed_only_denominator', 'mean_attempts', 'format_errors',
        'native_invalid_actions', 'resource_terminations', 'generated_tokens', 'raw_transcript_tokens',
        'mean_per_game_tokens_per_second', 'peak_kv_pool_allocated_bytes', 'completed_cohort_inference_seconds',
        'completed_cohort_gpu_hours', 'unfinished_cohort_accounting',
        'known_match_recent_utilization', 'known_match_middle_utilization', 'known_match_old_utilization',
        'known_match_recent_opportunities', 'known_match_middle_opportunities', 'known_match_old_opportunities']
    tables = [('main-comparison.csv', lambda row:row['evaluation_scope'] not in ('larger', 'window_sweep')),
        ('size-generalization.csv', lambda row:row['evaluation_scope'] == 'larger'),
        ('window-sweep.csv', lambda row:row['evaluation_scope'] == 'window_sweep')]
    for filename, include in tables:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(row for row in result['conditions'] if include(row))
    summary_fields = ['policy', 'attention', 'window', 'split', 'boards_per_training_seed',
        'board_seeds_sha256', 'complete_split', 'training_seeds', 'independent_training_runs',
        'metric', 'mean', 'sample_std', 'defined_training_runs']
    with (output / 'training-seed-summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction='ignore')
        writer.writeheader()
        for group in result['training_seed_summaries']:
            for name, values in group['metrics'].items():
                writer.writerow(dict(group, metric=name, **values))
    clean = dict(result, conditions=[{key: value for key, value in row.items() if key != 'rows'}
                                    for row in result['conditions']])
    (output / 'summary.json').write_text(json.dumps(clean, indent=2, allow_nan=False) + '\n')
    lines = ['# Matching Pairs memory experiment', '', result['task'], '',
        '| Condition | Training seed | Attention | Boards | Pairs found / total | Efficiency return | Attempts | Context limited |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in result['conditions']:
        attention = f'PS W={row["window"]}' if row['attention'] == 'ps' else 'Full'
        boards = str(row['episodes']) + (' (partial)' if not row['complete_split'] else '')
        lines.append(f'| {row["name"]} | {row["training_seed"]} | {attention} | {boards} | '
            f'{row["mean_pairs"]:.4f}/{row["total_pairs"]} | {row["mean_return"]:.4f} | {row["mean_attempts"]:.2f} | {row["resource_terminations"]} |')
    lines += ['', 'Resource-limited episodes retain observed pairs but receive the separately recorded -1 training reward. Their unobserved attempt-cap endpoints are not imputed as measured outcomes.', '',
              'Size generalization is a parameter-scaled RNG-Bench board, reported separately in size-generalization.csv. More copies per rank can make some decisions easier; this is not a claim of strictly greater difficulty.', '',
              'training-seed-summary.csv reports means and sample standard deviations across independent training seeds for each metric. Undefined metrics retain their number of defined training runs. Different board sets, including the 512-board test and 128-board window sweep, remain separate.', '',
              result['interpretation'], '',
              'Paired intervals below use 10,000 resamples over board seeds within a fixed training seed. Training-seed standard deviations describe independent training runs; episode counts are not training replication.', '']
    if result.get('deferred_qualification') is not None:
        gate = result['deferred_qualification']
        lines += [f'Deferred qualification: **{gate["status"]}** on {gate["episodes"]} validation boards. '
            f'Original full-context matched fraction {gate["full_context_matched_fraction"]:.6f} '
            f'(threshold ≥{gate["competence_threshold"]:.2f}); full minus original PS matched-fraction gap '
            f'{gate["full_minus_base_ps_matched_fraction"]:.6f} (threshold ≥{gate["forgetting_threshold"]:.2f}). '
            f'Selected update {gate["selected_update"]}, validation return {gate["selected_validation_return"]:.6f}. '
            'This qualification is separate from final-test results.', '']
    for row in result['paired_comparisons']:
        lines.append(f'- {row["treatment"]} minus {row["sliding_control"]}: {row["mean_difference"]:.4f}, '
            f'95% board-bootstrap interval {row["ci95"]}; full-context gap {row["full_context_gap"]:.4f}; recovery {row["recovery"]}.')
    if result.get('allocation_accounting') is not None:
        accounting = result['allocation_accounting']
        lines += ['', f'Allocation accounting: {accounting["total_allocated_gpu_hours"]:.6f} GPU-hours. {result["accounting_scope"]}.',
            '', f'By hardware type: {accounting["gpu_hours_by_type"]}. Pending maximum reservations: {accounting["pending_maximum_gpu_hours"]} GPU-hours, charged zero so far.',
            '', 'Allocation time includes setup, idle, failure and cleanup. Evaluation cohort time excludes setup and is a lower bound when interrupted cohort receipts are incomplete. KV bytes describe the shared physical pool per GPU, including available capacity and block padding.']
        from .allocation_accounting import save
        save(accounting, output / 'allocation-accounting')
    intervention_rows = []
    for entry in result.get('interventions', []):
        summary = entry['result']
        for subset in ('branches', 'conditional_unedited_selects_p'):
            values = summary[subset]
            for branch in ('unedited', 'swap', 'sham'):
                intervention_rows.append(dict(name=entry['name'], training_seed=entry['training_seed'],
                    subset=subset, branch=branch, denominator=values['denominator'],
                    qualified_states=summary['qualified_states'], executed_states=summary['executed_states'],
                    untestable_states=summary['untestable_states'], pending_states=summary['pending_states'],
                    causal_textual_memory_established=summary['causal_textual_memory_established'], **values[branch]))
    if intervention_rows:
        with (output / 'intervention-table.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(intervention_rows[0]))
            writer.writeheader(); writer.writerows(intervention_rows)
        lines += ['', 'Intervention results are in intervention-table.csv, with all qualified-state counts and the unedited-selects-partner subset reported separately. Any unexecuted or untestable states remain visible. No textual-memory conclusion follows from collecting states alone.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def window_series(conditions):
    series = {}
    for row in conditions:
        if row['attention'] == 'ps' and row['split'] == 'test':
            key = (row['policy'], row['training_seed'], row['board_seeds_sha256'])
            series.setdefault(key, []).append(row)
    for rows in series.values():
        if len({row['window'] for row in rows}) != len(rows):
            raise ValueError('A window curve requires one outcome per window on an identical board set')
        rows.sort(key=lambda row: row['window'])
    return series


def training_curve_rows(runs):
    rows = []
    for run in runs:
        root = Path(run['run_dir'])
        state = root / 'state.json'
        state = json.loads(state.read_text()) if state.exists() else {}
        committed = state.get('policy_version', 0)
        batch = batch_size(state)
        for path in sorted((root / 'update-metrics').glob('update-*.json')):
            point = json.loads(path.read_text())['metrics']
            update = point['update']
            if update > committed:
                continue
            if update < 1 or point['episodes'] != batch or point['training_episodes'] != update * batch:
                raise ValueError(f'A training curve requires complete {batch}-trajectory optimizer updates')
            rows.append(dict(training_seed=run['training_seed'], kind='sampled_training', update=update,
                training_episodes=point['training_episodes'],
                mean_return=point['mean_native_return'] if is_hero(state) else point['mean_return']))
        explicit = run.get('validation_summaries')
        paths = list((root / 'validation').glob('update-*/summary.json')) if explicit is None else list(map(Path, explicit))
        if explicit is not None and not paths:
            raise ValueError('Deferred validation summaries cannot be empty')
        versions = set()
        points = [(path, json.loads(path.read_text())) for path in paths]
        for path, point in sorted(points, key=lambda value: value[1]['policy_version']):
            update = point['policy_version']
            if update > committed:
                if explicit is not None:
                    raise ValueError('Deferred validation cannot include an uncommitted checkpoint')
                continue
            if point['episodes'] != 128 or point.get('greedy') is not True:
                raise ValueError('A validation curve requires complete greedy 128-board evaluations')
            if update < 0 or update in versions:
                raise ValueError('A validation curve requires one result per checkpoint version')
            versions.add(update)
            if explicit is not None:
                protocol = json.loads((path.parent / 'evaluation-protocol.json').read_text())
                config = RunConfig(**protocol['config'])
                training_identity(root, state, config)
                if point.get('split') != 'validation' or point.get('attention') != 'ps':
                    raise ValueError('Deferred learning curves use only PS validation results')
                condition = load_condition(dict(name=f'validation-{update}', policy='PS RL',
                    training_seed=run['training_seed'], split='validation', attention='ps',
                    window=config.window, greedy=True, expected_boards=128,
                    episodes=str(path.parent / 'episodes.jsonl'), summary=str(path)))
                if any(row.get('policy_version') != update or row.get('policy_identity') != point.get('policy_identity')
                       for row in condition['rows']):
                    raise ValueError('Deferred validation records belong to another checkpoint')
                if any(row.get(key) != getattr(config, key) for row in condition['rows']
                       for key in ('thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')):
                    raise ValueError('Deferred validation records used another generation controller')
                if config.seed != run['training_seed'] or not np.isclose(condition['mean_return'], point['mean_return'], rtol=0, atol=1e-10):
                    raise ValueError('Deferred validation return or training seed differs from its records')
            rows.append(dict(training_seed=run['training_seed'], kind='greedy_validation', update=update,
                training_episodes=update * batch, mean_return=point['mean_return']))
    if any(not np.isfinite(row['mean_return']) for row in rows):
        raise ValueError('A return curve cannot contain nonfinite measurements')
    return rows


def plots(result, manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    curve = training_curve_rows(manifest.get('training_runs', []))
    kinds = [kind for kind in ('sampled_training', 'greedy_validation') if any(row['kind'] == kind for row in curve)]
    if kinds:
        with (output / 'training-return-curve.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        fig, axes = plt.subplots(1, len(kinds), figsize=(7 * len(kinds), 4.5), squeeze=False)
        for ax, kind in zip(axes[0], kinds, strict=True):
            for seed in sorted({row['training_seed'] for row in curve if row['kind'] == kind}):
                points = sorted((row for row in curve if row['kind'] == kind and row['training_seed'] == seed),
                    key=lambda row: row['update'])
                ax.plot([point['training_episodes'] for point in points], [point['mean_return'] for point in points],
                    marker='o', label=f'PS GRPO seed {seed}')
            label = 'Sampled training efficiency return' if kind == 'sampled_training' else 'Greedy validation efficiency return'
            ax.set(xlabel='Training episodes', ylabel=label)
            ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(output / 'return-vs-training-episodes.png', dpi=180)
        plt.close(fig)
    series = window_series(result['conditions'])
    if series:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (policy, seed, _), rows in series.items():
            label = policy if seed is None else f'{policy} seed {seed}'
            label += f' ({rows[0]["episodes"]} boards)'
            ax.plot([row['window'] for row in rows], [row['mean_return'] for row in rows], marker='o',
                label=label)
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
        result['training_seed_summaries'] = aggregate_conditions(result['conditions'])
    save_tables(result, args.output)
    plots(result, manifest, args.output)
    print(json.dumps(dict(conditions=len(result['conditions']), comparisons=len(result['paired_comparisons']),
                         output=args.output)), flush=True)


if __name__ == '__main__':
    main()
