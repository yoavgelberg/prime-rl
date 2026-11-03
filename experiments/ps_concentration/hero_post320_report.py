"""Report the fixed448 continuation from committed batches and fresh held-out boards."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .hero_exports import file_hash
from .hero_extension_report import table
from .hero_pair_final import condition_result
from .hero_post320_final import configurations
from .hero_v2 import identity, read, utc, write
from .overlap_config import SPLITS, episode_metrics
from .rng_report import analyze_condition, paired_bootstrap


def training_rows(config):
    root = Path(config.output)
    state_path = root / 'state.json'
    sources = {}

    def captured(path):
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    snapshot = dict(state=captured(state_path), config=captured(root / 'resolved_config.json'))
    state = snapshot['state']
    version = state['policy_version']
    if (snapshot['config'] != json.loads(json.dumps(asdict(config))) or state['identity'] != identity(config)
            or not 320 <= version <= config.updates):
        raise ValueError(f'Report the declared continuation identity and committed320–{config.updates} updates')
    rows = []
    for update in range(321, version + 1):
        path = root / f'update-metrics/update-{update:04d}.json'
        metrics = captured(path)['metrics']
        if (metrics['update'] != update or metrics['episodes'] != 128
                or metrics['training_episodes'] != update * 128
                or metrics.get('exact_categorical_kl_active') is not True):
            raise ValueError('Require every complete128-trajectory committed batch with exact KL')
        rows.append(dict(update=update, collection_policy_version=update - 1,
            new_training_episodes=(update - 320) * 128, lineage_episodes=update * 128,
            shaped_reward=metrics['mean_return'], efficiency_reward=metrics['mean_native_return'],
            pairs=metrics['mean_pairs'], responses=metrics['mean_attempts'],
            seconds=metrics['seconds'], tokens_per_game_second=metrics['mean_per_game_tokens_per_second'],
            generated_tokens=metrics['generated_tokens'], gradient_norm=metrics['gradient_norm'],
            kl_per_generated_token=metrics['kl_per_generated_token']))
    return rows, sources, snapshot


def window_rows(conditions):
    rows, seen = [], set()
    first = SPLITS['post320_test'][0]
    for condition in conditions:
        if condition['split'] != 'post320_test' or condition['attention'] != 'ps':
            continue
        key = condition['arm'], condition['window']
        if key in seen:
            raise ValueError('Use exactly one fixed condition for each policy and window')
        seen.add(key)
        boards = sorted(condition['rows'], key=lambda row: row['board_seed'])[:128]
        if [row['board_seed'] for row in boards] != list(range(first, first + 128)):
            raise ValueError('Every window point must use the same first128 fresh7M boards')
        rows.append(dict(arm=condition['arm'], window=condition['window'],
            source_condition=condition['name'], derived_subset=condition['expected_boards'] == 512,
            **episode_metrics(boards)))
    return sorted(rows, key=lambda row: (row['arm'], row['window']))


def comparison(conditions, final):
    named = {row['name']: row for row in conditions}
    if not all(name in named for name in final['primary']):
        return None
    treatment, control = [named[name] for name in final['primary']]
    if any(row['attention'] != 'ps' or row['window'] != 1024
            or row['split'] != 'post320_test' or row['expected_boards'] != 512
            for row in (treatment, control)):
        raise ValueError('The primary comparison requires both fixed512-board PS1024 tests')
    return dict(paired_bootstrap(treatment['rows'], control['rows']),
        treatment=treatment['name'], control=control['name'],
        treatment_return=treatment['mean_return'], control_return=control['mean_return'],
        independent_training_lineages=1, training_seed_std=None,
        scope=f"Frozen update{final.get('fixed_final_update', 448)} minus PS-trained parent320; not a trained Full-RL control or a temperature-only effect")


def memory_result(final_path, final):
    from .hero_memory import aggregate, read_states, state_hash

    output = Path(final_path).parent / 'memory/continued-w1024'
    summary_path = output / 'summary.json'
    if not summary_path.exists():
        return None
    inputs = read(output / 'collection-inputs.json')
    session_path = output / 'session.json'
    session = read(session_path)
    state_path = output / 'collection/states.jsonl.gz'
    states = read_states(state_path)
    entry, = [row for row in final['conditions'] if row['split'] == 'intervention']
    expected_episodes = str(Path(entry['summary']).parent / 'episodes.jsonl')
    if (inputs['boundary'] != 'before_thinking' or inputs['maximum'] != 128
            or inputs['episodes'] != expected_episodes or inputs['checkpoint'] != entry['checkpoint']
            or file_hash(inputs['episodes']) != inputs['episodes_sha256']
            or session['checkpoint'] != str(Path(inputs['checkpoint']).resolve())
            or session['samples'] != 32 or session['checkpoint_metadata'] != file_hash(
                Path(entry['checkpoint']) / 'experiment.json')
            or session['states'] != [dict(state_id=state['state_id'], sha256=state_hash(state)) for state in states]):
        raise ValueError('Preserve the fixed prethinking collection and checkpoint')
    actual = aggregate(states, output / 'states', checkpoint=session['checkpoint'], samples=32)
    summary = read(summary_path)
    if actual != summary or summary['pending_states']:
        raise ValueError('The saved intervention summary must match every qualified-state result')
    return dict(path=str(summary_path), sha256=file_hash(summary_path), result=summary,
        states_sha256=file_hash(state_path), session_sha256=file_hash(session_path),
        collection_inputs_sha256=file_hash(output / 'collection-inputs.json'))


def figures(curve, windows, conditions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.savefig(output / f'{name}.png', dpi=180)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)

    if curve:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, metric, label in zip(axes, ('shaped_reward', 'efficiency_reward'),
                ('Sampled reward including error penalties', 'Sampled efficiency reward'), strict=True):
            ax.plot([row['new_training_episodes'] for row in curve], [row[metric] for row in curve])
            ax.set(xlabel='Additional episodes after parent update320', ylabel=label)
            ax.grid(alpha=.25)
        save(fig, 'return-vs-training-episodes')
    if windows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for arm in ('continued', 'parent320', 'base'):
            points = [row for row in windows if row['arm'] == arm]
            if not points:
                continue
            for ax, metric in zip(axes, ('mean_return', 'mean_pairs'), strict=True):
                ax.plot([row['window'] for row in points], [row[metric] for row in points], marker='o', label=arm)
        for ax in axes:
            ax.set(xscale='log', xlabel='Rolling window W (tokens), identical128 boards')
            ax.grid(alpha=.25)
            ax.legend(frameon=False)
        axes[0].set_ylabel('Greedy efficiency reward')
        axes[1].set(ylabel='Pairs found /8', ylim=(0, 8.1))
        save(fig, 'final-return-vs-window')
    known = [row for row in conditions if row.get('known_match')]
    if known:
        bins = ['<=W', '(W,2W]', '>2W']
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for row in known:
            ax.plot(bins, [row['known_match'][age]['utilization'] for age in bins], marker='o', label=row['arm'])
        ax.set(ylim=(0, 1.05), xlabel='Latest direct reveal age, W1024', ylabel='Known-partner utilization')
        ax.legend(frameon=False)
        ax.grid(alpha=.25)
        save(fig, 'known-match-utilization-vs-age')


def report(final_path, output, *, allow_partial=False):
    final_path, output = Path(final_path), Path(output)
    final = read(final_path)
    if (final['format'] not in ('hero-post320-final-v1', 'hero-goal25-final-v1')
            or (final['format'] == 'hero-post320-final-v1' and final['fixed_final_update'] != 448)
            or final['baseline_update'] != 320 or file_hash(final['plan']) != final['plan_sha256']):
        raise ValueError('Use the frozen post320 training and fixed448 test plans')
    target = final['fixed_final_update']
    configs = configurations(read(final['plan']))
    curve, sources, snapshot = training_rows(configs['continued'])
    committed = snapshot['state']['policy_version']
    available, missing = [], []
    for entry in final['conditions']:
        if not Path(entry['summary']).exists():
            missing.append(entry['condition'])
            continue
        condition = condition_result(entry) | dict(arm=entry['arm'])
        config = configs['parent320' if entry['arm'] == 'parent320' else 'continued']
        condition['mean_shaped_reward'] = sum(row['training_return']
            - config.format_penalty * row['format_errors'] - config.invalid_penalty * row['invalid_actions']
            for row in condition['rows']) / len(condition['rows'])
        for path in (entry['summary'], condition['episodes_file'], entry['argv'][entry['argv'].index('--config') + 1]):
            sources[path] = file_hash(path)
        available.append(condition)
    if available and committed != target:
        raise ValueError(f'Frozen update{target} test results cannot precede completion of training')
    memory = memory_result(final_path, final)
    pending = committed != target or bool(missing) or memory is None
    if pending and not allow_partial:
        raise ValueError(f'Final results require{target} committed updates, every evaluation and the memory analysis')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'training-state.json', snapshot['state'])
    write(output / 'resolved_config.json', snapshot['config'])
    targets = [row for row in available if row['split'] == 'post320_test'
        and row['attention'] == 'ps' and row['window'] == 1024 and row['expected_boards'] == 512]
    if targets:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(configs['continued'].model,
            revision=configs['continued'].revision, local_files_only=True)
        for row in targets:
            row.update(analyze_condition(row, tokenizer,
                examples=output / 'examples/continued' if row['arm'] == 'continued' else None))
    windows = window_rows(available)
    fields = ['name', 'policy_version', 'attention', 'window', 'episodes', 'total_pairs', 'max_responses',
        'mean_return', 'mean_shaped_reward', 'mean_pairs', 'matched_fraction', 'completion_rate',
        'mean_attempts', 'mean_attempts_to_completion', 'completed_only_mean_attempts',
        'completed_only_denominator', 'format_errors', 'native_invalid_actions', 'resource_terminations',
        'generated_tokens', 'raw_transcript_tokens', 'mean_per_game_tokens_per_second',
        'peak_kv_pool_allocated_bytes', 'completed_cohort_inference_seconds', 'completed_cohort_gpu_hours']
    table(output / 'main-comparison.csv', [row for row in available
        if row['split'] == 'post320_test' and row['expected_boards'] == 512], fields)
    table(output / 'size-generalization.csv', [row for row in available if row['split'] == 'post320_larger'], fields)
    table(output / 'window-sweep.csv', windows,
        ['arm', 'window', 'episodes', 'mean_return', 'mean_pairs', 'source_condition', 'derived_subset'])
    table(output / 'training-curve.csv', curve, list(curve[0]) if curve else ['update'])
    known = [dict(policy=row['arm'], age=age, **counts) for row in targets
        for age, counts in row['known_match'].items()]
    table(output / 'known-match-utilization.csv', known, ['policy', 'age', 'opportunities', 'correct', 'utilization'])
    if memory is not None:
        from .hero_memory import save_table
        save_table(memory['result'], output)
        cache = memory['result'].get('streaming_vs_unedited_cold')
        parity = memory['result'].get('streaming_teacher_parity', {})
        table(output / 'intervention-cache-control.csv', [dict(cache,
            native_teacher_mean_abs_logp=parity['mean_abs'], native_teacher_p99_abs_logp=parity['p99'],
            native_teacher_parity_status=parity['status'])] if cache else [],
            ['denominator', 'streaming_greedy_correct', 'cold_greedy_correct', 'cold_minus_streaming',
             'native_teacher_mean_abs_logp', 'native_teacher_p99_abs_logp', 'native_teacher_parity_status'])
    figures(curve, windows, available, output)
    result = dict(utc=utc(), status='PARTIAL' if pending else 'COMPLETE', committed_update=committed,
        final_plan=dict(path=str(final_path), sha256=file_hash(final_path)),
        evaluation_source=dict(root=final['evaluation_source_root'], sha256=final['evaluation_source_sha256']),
        missing_conditions=missing, missing_memory=memory is None, paired_comparison=comparison(available, final),
        conditions=[{key:value for key, value in row.items() if key != 'rows'} for row in available],
        intervention=memory, input_sha256=sources, training_snapshot=snapshot,
        independent_training_lineages=1, training_seed_std=None,
        analysis_implementation={name:file_hash(Path(__file__).with_name(name)) for name in
            ('hero_post320_report.py', 'rng_report.py', 'rng_memory_metrics.py', 'rng_trace.py',
             'rng_intervention_summary.py', 'hero_memory.py')},
        scope=final['scope'], limitations=['No trained Full-RL control or independent training-seed replication',
            'Continuation and sampling-temperature change are confounded',
            'Native actor/learner strict numerical parity remains unresolved',
            'Known-partner utilization compares different visited-history populations',
            'Readable notes and PS performance alone do not establish causal use of generated text'])
    write(output / 'results.json', result)
    lines = ['# W1024 continuation results', '', f'Status: {result["status"]}. ' + final['scope'], '',
        f'Training committed through {committed}/{target}. Missing evaluation conditions: {len(missing)}. '
        f'Memory analysis missing: {memory is None}.', '',
        '| Policy | Attention | Pairs /8 | Responses | Efficiency reward | With error penalties |',
        '|---|---|---:|---:|---:|---:|']
    for row in available:
        if row['split'] == 'post320_test' and row['expected_boards'] == 512:
            lines.append(f'| {row["arm"]} v{row["policy_version"]} | {row["attention"]} W{row["window"]} | '
                f'{row["mean_pairs"]:.3f} | {row["mean_attempts"]:.2f} | {row["mean_return"]:+.4f} | '
                f'{row["mean_shaped_reward"]:+.4f} |')
    paired = result['paired_comparison']
    if paired is not None:
        lines += ['', f'Frozen update{target} minus parent320 PS1024 efficiency: {paired["mean_difference"]:+.6f}, '
            f'95% paired board-bootstrap interval {paired["ci95"]} (10,000 resamples).']
    lines += ['', 'Native score is pairs found. Efficiency is pairs/8 minus responses/96; '
        'the training reward also subtracts0.02per format error and native invalid action. '
        'The larger board uses16pairs and192responses and is reported separately.', '',
        'Training curves use sampled boards; final tables use greedy decoding. All window points use '
        'the same first128 fresh7Mboards. Board intervals do not measure training-seed variability.', '',
        'Textual memory requires at least32qualified separate4Mstates and a swap effect beyond sham, '
        'with all states counted and streaming versus unedited cold continuation reported separately.', '',
        '[Metrics and provenance](results.json) · [Training](training-curve.csv) · '
        '[Window sweep](window-sweep.csv) · [Larger boards](size-generalization.csv)']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
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
