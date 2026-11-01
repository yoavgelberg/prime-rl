"""Report paired continuations without treating their shared parent as replication."""

import argparse
import csv
import json
import math
from pathlib import Path

from .hero_exports import file_hash
from .hero_pair_final import condition_result
from .hero_v2 import read, utc, write
from .rng_report import analyze_condition, paired_bootstrap


def reward_columns(condition, *, format_penalty, invalid_penalty):
    rows = condition['rows']
    return dict(mean_unshaped_efficiency_reward=condition['mean_return'],
        mean_shaped_reward=math.fsum(row['training_return'] - format_penalty * row['format_errors']
            - invalid_penalty * row['invalid_actions'] for row in rows) / len(rows))


def training_rows(plan):
    rows, sources = [], {}
    roots = [('shared-parent', Path(plan['parent_run']), 1, 128)]
    roots.extend((entry['name'], Path(entry['output']), 129, 192) for entry in plan['branches'])
    for arm, root, first, last in roots:
        if read(root / 'state.json')['policy_version'] != last:
            raise ValueError('Report requires the completed parent and both full continuations')
        for update in range(first, last + 1):
            path = root / f'update-metrics/update-{update:04d}.json'
            metrics = read(path)['metrics']
            if (metrics['update'] != update or metrics['episodes'] != 128
                    or metrics['training_episodes'] != update * 128):
                raise ValueError('Every plotted point requires its complete committed training batch')
            sources[str(path)] = file_hash(path)
            rows.append(dict(arm=arm, update=update, lineage_episodes=update * 128,
                new_episodes=(update if arm == 'shared-parent' else update - 128) * 128,
                shaped_reward=metrics['mean_return'],
                unshaped_efficiency_reward=metrics['mean_native_return'],
                pairs=metrics['mean_pairs'], responses=metrics['mean_attempts'],
                seconds=metrics['seconds'], tok_s_game=metrics['mean_per_game_tokens_per_second']))
    return rows, sources


def comparison(conditions):
    named = {row['name']: row for row in conditions}
    treatment, control, full = [named[name] for name in
        ('w256-ps-w256-test', 'w1024-ps-w256-test', 'w1024-full-test')]
    result = paired_bootstrap(treatment['rows'], control['rows'])
    paired_bootstrap(full['rows'], control['rows'])
    gap = full['mean_return'] - control['mean_return']
    return dict(result, treatment=treatment['name'], control=control['name'],
        treatment_return=treatment['mean_return'], control_return=control['mean_return'],
        full_reference_return=full['mean_return'], full_reference_gap=gap,
        recovery=result['mean_difference'] / gap if gap >= .10 else None,
        branch_training_seed=11, independent_training_seeds=1,
        scope='Matched W256 versus W1024 transfer continuations from one shared128-update parent. '
              'The control was trained under W1024 prefix sliding, not full attention. '
              'Intervals describe board uncertainty; training variability is unmeasured.')


def table(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plots(conditions, curve, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm in ('shared-parent', 'w1024', 'w256'):
        points = [row for row in curve if row['arm'] == arm]
        if arm != 'shared-parent':
            points = [next(row for row in curve if row['arm'] == 'shared-parent' and row['update'] == 128), *points]
        for axis, field in zip(axes, ('shaped_reward', 'unshaped_efficiency_reward'), strict=True):
            axis.plot([row['lineage_episodes'] for row in points], [row[field] for row in points], label=arm)
            axis.set_xlabel('Episodes along policy lineage (first16,384 shared)')
            axis.grid(alpha=.25)
    axes[0].set_ylabel('Sampled training reward, with error penalties')
    axes[1].set_ylabel('Sampled efficiency reward, before error penalties')
    axes[0].legend(frameon=False)
    fig.savefig(output / 'return-vs-training-episodes.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm in ('w1024', 'w256'):
        points = sorted((row for row in conditions if row['name'].startswith(f'{arm}-window-')),
            key=lambda row: row['window'])
        for axis, field in zip(axes, ('mean_return', 'mean_pairs'), strict=True):
            axis.plot([row['window'] for row in points], [row[field] for row in points], marker='o', label=arm)
            axis.set(xscale='log', xlabel='Rolling window W (tokens)')
    axes[0].set_ylabel('Greedy held-out efficiency reward (128boards)')
    axes[1].set_ylabel('Pairs found /8 (same128boards)')
    axes[1].set_ylim(0, 8)
    axes[0].legend(frameon=False)
    fig.savefig(output / 'final-return-vs-window.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = ['<=W', '(W,2W]', '>2W']
    for row in conditions:
        if row.get('known_match') is not None:
            ax.plot(bins, [row['known_match'][age]['utilization'] for age in bins], marker='o', label=row['name'])
    ax.set(ylim=(0, 1.05), xlabel='Age of latest direct environment reveal (W256)',
        ylabel='Known-match utilization')
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / 'known-match-utilization-vs-age.png', dpi=180)
    plt.close(fig)


def report(final_path, output, memory):
    final, output, memory = read(final_path), Path(output), Path(memory)
    plan = read(final['plan'])
    if file_hash(final['plan']) != final['plan_sha256']:
        raise ValueError('Report requires the frozen paired experiment plan')
    output.mkdir(parents=True, exist_ok=True)
    conditions = [condition_result(entry) for entry in final['conditions'] if entry['split'] != 'intervention']
    training = read(Path(plan['parent_run']) / 'resolved_config.json')
    for condition in conditions:
        condition.update(reward_columns(condition, format_penalty=training['format_penalty'],
            invalid_penalty=training['invalid_penalty']))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(read(Path(plan['parent_run']) / 'resolved_config.json')['model'])
    for condition in conditions:
        if condition['name'] in {f'{label}-ps-w256-test' for label in ('base', 'parent', 'w1024', 'w256')}:
            examples = output / 'examples' / condition['policy'] if condition['name'].startswith(('w1024-', 'w256-')) else None
            condition.update(analyze_condition(condition, tokenizer, examples=examples))
    delta = comparison(conditions)
    curve, sources = training_rows(plan)
    table(output / 'training-curve.csv', curve, list(curve[0]))
    fields = ['name', 'policy_version', 'attention', 'window', 'episodes', 'mean_return',
        'mean_unshaped_efficiency_reward', 'mean_shaped_reward', 'mean_pairs',
        'matched_fraction', 'completion_rate', 'mean_attempts', 'mean_attempts_to_completion',
        'completed_only_mean_attempts', 'completed_only_denominator', 'format_errors', 'format_error_rate',
        'native_invalid_actions', 'generated_tokens', 'raw_transcript_tokens',
        'mean_per_game_tokens_per_second', 'peak_kv_pool_allocated_bytes',
        'completed_cohort_inference_seconds', 'completed_cohort_gpu_hours', 'resource_terminations',
        'known_match_recent_utilization', 'known_match_middle_utilization', 'known_match_old_utilization',
        'known_match_recent_opportunities', 'known_match_middle_opportunities', 'known_match_old_opportunities']
    table(output / 'main-comparison.csv', [row for row in conditions if row['split'] == 'test'
        and row['expected_boards'] == 512], fields)
    table(output / 'size-generalization.csv', [row for row in conditions if row['split'] == 'larger'], fields)
    table(output / 'window-sweep.csv', [row for row in conditions if '-window-' in row['name']], fields)
    interventions, intervention_rows = {}, []
    for arm in ('w1024', 'w256'):
        summary_path = memory / arm / 'summary.json'
        summary = read(summary_path)
        interventions[arm] = dict(path=str(summary_path), sha256=file_hash(summary_path), result=summary)
        for subset in ('branches', 'conditional_unedited_selects_p'):
            for branch in ('unedited', 'swap', 'sham'):
                intervention_rows.append(dict(arm=arm, subset=subset, branch=branch,
                    **{key: summary[key] for key in ('qualified_states', 'executed_states', 'untestable_states',
                        'pending_states', 'causal_textual_memory_established')},
                    denominator=summary[subset]['denominator'], **summary[subset][branch]))
    table(output / 'intervention-table.csv', intervention_rows, list(intervention_rows[0]))
    plots(conditions, curve, output)
    accounting = Path(plan['parent_run']).parent / 'whole-goal-accounting/allocation-costs.json'
    result = dict(utc=utc(), task='Native RNG-Bench Matching Pairs4x4;96responsecap;128thinking+32answer token caps',
        selection=final['selection'], paired_comparison=delta,
        conditions=[{key: value for key, value in row.items() if key != 'rows'} for row in conditions],
        interventions=interventions, training_metric_sources=sources,
        allocation_accounting=dict(path=str(accounting), sha256=file_hash(accounting), result=read(accounting))
            if accounting.exists() else None,
        final_plan=dict(path=str(final_path), sha256=file_hash(final_path)),
        interpretation=plan['scope'],
        reward_definition='Efficiency=pairs/8-responses/96; training also subtracts.02formaterrors+.02nativeinvalids. Native score=pairs.',
        limitations=['One shared training seed; no estimate of training variability',
            'W256 arm continues a W1024-trained parent; it is not trained solely atW256 from the base',
            'W1024 control uses prefix sliding during training; no Full-RL causal-attention control',
            'Thinking cap uses a forced delimiter transition; forced tokens have no direct policy loss',
            'Strict native-rollout/teacher BF16 numerical parity was not established; clipping and discrepancy metrics are saved',
            'Good performance alone does not establish readable textual memory; intervention validity and counts control that claim'])
    write(output / 'results.json', result)
    lines = ['# Paired compressed-memory continuation results', '',
        'One shared seed11parent,128updates atW1024,then64matched additional updates perW1024/W256branch. The branches are not independent seed replications.', '',
        '| Policy | Eval attention | Pairs/8 | Responses | Efficiency reward | Reward incl. errors |',
        '|---|---|---:|---:|---:|---:|']
    for row in conditions:
        if row['split'] == 'test' and row['expected_boards'] == 512:
            lines.append(f'|{row["name"]}(v{row["policy_version"]})|{row["attention"]} W{row["window"]}|'
                f'{row["mean_pairs"]:.3f}|{row["mean_attempts"]:.2f}|{row["mean_return"]:+.4f}|{row["mean_shaped_reward"]:+.4f}|')
    lines.extend(['', f'W256-minus-W1024continuation under the sameW256mask: {delta["mean_difference"]:+.4f} efficiency reward, '
        f'95% paired board-bootstrap interval {delta["ci95"]}. This interval does not measure training-seed variability.', '',
        f'Recovery relative to the W1024control evaluated with full history: {delta["recovery"]}; raw full-reference gap {delta["full_reference_gap"]:+.4f}.', '',
        'Checkpoint selection uses only the128validationboards under each trainingwindow,with earlier ties. If a branch selects128,it selects the shared parent baseline and demonstrates no newly selected continuation checkpoint.', '',
        'Readable memory conclusions depend on the intervention table,including all qualified and untestable states and the cold-rebuild/streaming comparison. Fewer than32qualified states is insufficient.', '',
        '[Training curves](return-vs-training-episodes.png) · [Window sweep](final-return-vs-window.png) · '
        '[Known-match utilization](known-match-utilization-vs-age.png) · [Interventions](intervention-table.csv) · [Full metrics/provenance](results.json)', '',
        'Twenty fixed-seed-order greedy transcripts perselected arm,with eviction positions,are in examples/.', '',
        'Parameter-scaled4×8results appear in size-generalization.csv;this is not the registered POPGymConcentrationHard benchmark.'])
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return dict(results=str(output / 'results.json'), paired_difference=delta['mean_difference'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory', required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.final_plan, args.output, args.memory), indent=2))


if __name__ == '__main__':
    main()
