"""Report the extension from committed batches and fixed fresh-board evaluations."""

import argparse
import csv
from pathlib import Path

from .hero_exports import file_hash
from .hero_pair_final import condition_result
from .hero_v2 import read, utc, write
from .overlap_config import episode_metrics
from .rng_report import analyze_condition, paired_bootstrap


def training_rows(plan):
    rows, sources = [], {}
    for branch in plan['branches']:
        root = Path(branch['output'])
        version = read(root / 'state.json')['policy_version']
        for update in range(193, version+1):
            path = root / f'update-metrics/update-{update:04d}.json'
            metrics = read(path)['metrics']
            if (metrics['update'] != update or metrics['episodes'] != 128
                    or metrics['training_episodes'] != update*128):
                raise ValueError('Plot only complete committed128-trajectory extension batches')
            if update >= 215 and metrics.get('exact_categorical_kl_active') is not True:
                raise ValueError('Do not mislabel the approved exact-KL schedule')
            sources[str(path)] = file_hash(path)
            rows.append(dict(arm=f'w{branch["window"]}', update=update,
                new_training_episodes=(update-192)*128, lineage_episodes=update*128,
                shaped_reward=metrics['mean_return'], efficiency_reward=metrics['mean_native_return'],
                pairs=metrics['mean_pairs'], responses=metrics['mean_attempts'],
                seconds=metrics['seconds'], tokens_per_game_second=metrics['mean_per_game_tokens_per_second'],
                generated_tokens=metrics['generated_tokens'], gradient_norm=metrics['gradient_norm'],
                kl_per_generated_token=metrics['kl_per_generated_token'],
                exact_kl=bool(metrics.get('exact_categorical_kl_active', False))))
    return rows, sources


def window_rows(conditions):
    rows = []
    for condition in conditions:
        if (condition['arm'] not in ('w512', 'w1024') or condition['split'] != 'extension_test'
                or condition['attention'] != 'ps'):
            continue
        boards = sorted(condition['rows'], key=lambda row: row['board_seed'])[:128]
        if len(boards) != 128 or [r['board_seed'] for r in boards] != list(range(5000000, 5000128)):
            raise ValueError('All sweep points use the same first128 fresh boards')
        rows.append(dict(arm=condition['arm'], window=condition['window'],
            source_condition=condition['name'], derived_subset=condition['expected_boards'] != 128,
            **episode_metrics(boards)))
    return sorted(rows, key=lambda row: (row['arm'], row['window']))


def comparison(conditions, final):
    named = {row['name']: row for row in conditions}
    required = [*final['primary'], 'w1024-full-w1024-extension_test']
    if not all(name in named for name in required):
        return None
    treatment, control, full = [named[name] for name in required]
    result = paired_bootstrap(treatment['rows'], control['rows'])
    paired_bootstrap(full['rows'], control['rows'])
    gap = full['mean_return']-control['mean_return']
    return dict(result, treatment=treatment['name'], control=control['name'], full_reference=full['name'],
        treatment_return=treatment['mean_return'], control_return=control['mean_return'],
        full_reference_return=full['mean_return'], full_reference_gap=gap,
        recovery=result['mean_difference']/gap if gap >= .10 else None,
        treatment_pairs=treatment['mean_pairs'], control_pairs=control['mean_pairs'],
        independent_training_seeds=1, training_seed_std=None)


def table(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def figures(curve, windows, conditions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for arm in ('w1024', 'w512'):
        points = [r for r in curve if r['arm'] == arm]
        for ax, metric in zip(axes, ('shaped_reward', 'efficiency_reward'), strict=True):
            ax.plot([r['new_training_episodes'] for r in points], [r[metric] for r in points], label=arm)
    for ax in axes:
        ax.axvline((215-192)*128, color='grey', linestyle=':', alpha=.6)
        ax.set_xlabel('Fresh episodes since common update192 checkpoint')
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Sampled training reward including error penalties')
    axes[1].set_ylabel('Sampled efficiency reward before error penalties')
    fig.savefig(output/'return-vs-training-episodes.png', dpi=180)
    fig.savefig(output/'return-vs-training-episodes.pdf')
    plt.close(fig)
    if windows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for arm in ('w1024', 'w512'):
            points = [r for r in windows if r['arm'] == arm]
            for ax, metric in zip(axes, ('mean_return', 'mean_pairs'), strict=True):
                ax.plot([r['window'] for r in points], [r[metric] for r in points], marker='o', label=arm)
        for ax in axes:
            ax.set(xscale='log', xlabel='Rolling window W (tokens)')
            ax.grid(alpha=.25)
            ax.legend(frameon=False)
        axes[0].set_ylabel('Greedy fresh-board efficiency reward')
        axes[1].set(ylabel='Pairs found /8', ylim=(0, 8.1))
        fig.savefig(output/'final-return-vs-window.png', dpi=180)
        fig.savefig(output/'final-return-vs-window.pdf')
        plt.close(fig)
    known = [r for r in conditions if r.get('known_match') and r['window'] == 512]
    if known:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        bins = ['<=W', '(W,2W]', '>2W']
        for row in known:
            ax.plot(bins, [row['known_match'][age]['utilization'] for age in bins], marker='o', label=row['arm'])
        ax.set(ylim=(0, 1.05), xlabel='Latest direct environment reveal age (W512)', ylabel='Known-match utilization')
        ax.legend(frameon=False)
        fig.savefig(output/'known-match-utilization-vs-age.png', dpi=180)
        fig.savefig(output/'known-match-utilization-vs-age.pdf')
        plt.close(fig)


def report(final_path, output, *, allow_partial=False):
    final_path, output = Path(final_path), Path(output)
    final = read(final_path)
    if file_hash(final['plan']) != final['plan_sha256']:
        raise ValueError('Use the frozen extension training plan')
    plan = read(final['plan'])
    output.mkdir(parents=True, exist_ok=True)
    curve, sources = training_rows(plan)
    available, missing = [], []
    for entry in final['conditions']:
        if not Path(entry['summary']).exists():
            missing.append(entry['condition'])
        else:
            available.append(condition_result(entry) | dict(arm=entry['arm']))
    memories, intervention_rows, cache_rows, missing_memory = {}, [], [], []
    for label in ('w512-w512', 'w1024-w512', 'w1024-w1024'):
        path = final_path.parent/'memory'/label/'summary.json'
        if not path.exists():
            missing_memory.append(label)
            continue
        summary = read(path)
        if summary['pending_states']:
            raise ValueError('An intervention summary still has pending qualified states')
        memories[label] = dict(path=str(path), sha256=file_hash(path), result=summary)
        if 'streaming_vs_unedited_cold' in summary:
            cache_rows.append(dict(policy=label, **summary['streaming_vs_unedited_cold'],
                native_teacher_mean_abs_logp=summary['streaming_teacher_parity']['mean_abs'],
                native_teacher_p99_abs_logp=summary['streaming_teacher_parity']['p99'],
                native_teacher_parity_status=summary['streaming_teacher_parity']['status']))
        for subset in ('branches', 'conditional_unedited_selects_p'):
            for branch in ('unedited', 'swap', 'sham'):
                intervention_rows.append(dict(policy=label, subset=subset, branch=branch,
                    qualified_states=summary['qualified_states'],
                    executed_states=summary['executed_states'], untestable_states=summary['untestable_states'],
                    pending_states=summary['pending_states'],
                    causal_textual_memory_established=summary['causal_textual_memory_established'],
                    denominator=summary[subset]['denominator'], **summary[subset][branch]))
    if not allow_partial and (missing or missing_memory):
        raise ValueError('Final results require every frozen evaluation and all three memory analyses')
    targets = [c for c in available if c['split']=='extension_test' and c['attention']=='ps'
        and c['window']==512 and c['expected_boards']==512 and c['arm'] in ('w512', 'w1024', 'parent')]
    if targets:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan['branches'][0]['model'], local_files_only=True)
        for condition in targets:
            condition.update(analyze_condition(condition, tokenizer,
                examples=output/'examples'/condition['arm'] if condition['arm'] != 'parent' else None))
    configs = {f'w{c["window"]}':c for c in plan['branches']}
    for condition in available:
        config = configs.get(condition['arm'], plan['branches'][0])
        condition['mean_shaped_reward'] = sum(r['training_return']-config['format_penalty']*r['format_errors']
            -config['invalid_penalty']*r['invalid_actions'] for r in condition['rows'])/len(condition['rows'])
    windows = window_rows(available)
    fields = ['name', 'policy_version', 'attention', 'window', 'episodes', 'total_pairs', 'max_responses', 'mean_return', 'mean_shaped_reward',
        'mean_pairs', 'matched_fraction', 'completion_rate', 'mean_attempts', 'mean_attempts_to_completion',
        'completed_only_mean_attempts', 'completed_only_denominator', 'format_errors', 'native_invalid_actions',
        'generated_tokens', 'raw_transcript_tokens', 'mean_per_game_tokens_per_second',
        'peak_kv_pool_allocated_bytes', 'completed_cohort_inference_seconds', 'completed_cohort_gpu_hours']
    table(output/'main-comparison.csv', [r for r in available if r['split']=='extension_test' and r['expected_boards']==512], fields)
    table(output/'size-generalization.csv', [r for r in available if r['split']=='extension_larger'], fields)
    table(output/'window-sweep.csv', windows, ['arm', 'window', 'episodes', 'mean_return', 'mean_pairs', 'source_condition', 'derived_subset'])
    table(output/'intervention-table.csv', intervention_rows, ['policy', 'subset', 'branch', 'qualified_states',
        'executed_states', 'untestable_states', 'pending_states', 'causal_textual_memory_established',
        'denominator', 'greedy_correct', 'greedy_follows_false',
        'probability_correct_partner', 'probability_following_false'])
    table(output/'intervention-cache-control.csv', cache_rows, ['policy', 'denominator', 'streaming_greedy_correct',
        'cold_greedy_correct', 'cold_minus_streaming', 'native_teacher_mean_abs_logp',
        'native_teacher_p99_abs_logp', 'native_teacher_parity_status'])
    table(output/'training-curve.csv', curve, list(curve[0]) if curve else ['arm', 'update'])
    figures(curve, windows, available, output)
    result = dict(utc=utc(), status='PARTIAL' if missing or missing_memory else 'COMPLETE',
        final_plan=dict(path=str(final_path), sha256=file_hash(final_path)), missing_conditions=missing,
        missing_memory=missing_memory, training_metric_sources=sources, paired_comparison=comparison(available, final),
        conditions=[{k:v for k,v in row.items() if k != 'rows'} for row in available],
        interventions=memories, independent_training_seeds=1, training_seed_std=None,
        analysis_implementation={name:file_hash(Path(__file__).with_name(name)) for name in
            ('hero_extension_report.py', 'rng_report.py', 'rng_memory_metrics.py', 'rng_trace.py', 'rng_intervention_summary.py')},
        scope=final['scope'], limitations=['No Full-RL training control or independent training-seed replication',
            'Strict native/learner numerical parity remains unresolved',
            'Good PS scores alone do not establish causally used readable textual memory',
            'Sampled training curves and changing visited-history populations are not held-out causal comparisons'])
    write(output/'results.json', result)
    lines = ['# Compressed-memory extension results', '', f'Status: {result["status"]}. '+final['scope'], '',
        '| Policy | Attention | Pairs /8 | Responses | Efficiency reward | Reward with error penalties |',
        '|---|---|---:|---:|---:|---:|']
    for row in available:
        if row['split']=='extension_test' and row['expected_boards']==512:
            lines.append(f'| {row["arm"]} v{row["policy_version"]} | {row["attention"]} W{row["window"]} | '
                f'{row["mean_pairs"]:.3f} | {row["mean_attempts"]:.2f} | {row["mean_return"]:+.4f} | {row["mean_shaped_reward"]:+.4f} |')
    lines += ['', 'Efficiency is pairs/8 − responses/96. Training also subtracts 0.02 per format error and native invalid action.', '',
        'Training curves are sampled batches; final tables use greedy decoding on fixed fresh boards. '
        'Bootstrap intervals measure board uncertainty within the single common training lineage.', '',
        'Readable memory requires at least 32 qualified intervention states and a swap effect beyond sham, '
        'with the cold-rebuild versus streaming control reported separately.', '',
        '[Metrics and provenance](results.json) · [Training curves](return-vs-training-episodes.png) · '
        '[Size generalization](size-generalization.csv) · [Interventions](intervention-table.csv)']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report(args.final_plan, args.output, allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
