import csv
import hashlib
import json

import pytest

from .rng_report import analyze_condition, load_condition, paired_bootstrap, plots, save_tables, seed_summary, summarize_manifest, training_curve_rows, window_series
from .test_rng_select import selection_run


def test_training_curves_keep_sampled_returns_separate_and_exclude_uncommitted_updates(tmp_path):
    root = tmp_path / 'run'
    (root / 'update-metrics').mkdir(parents=True)
    for update in (1, 2):
        (root / 'update-metrics' / f'update-{update:04d}.json').write_text(json.dumps(dict(metrics=dict(
            update=update, episodes=64, training_episodes=update * 64, mean_return=-.5 + update / 10))))
    for update in (0, 2):
        path = root / 'validation' / f'update-{update:04d}' / 'summary.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(policy_version=update, episodes=128, greedy=True, mean_return=.2)))
    run = dict(run_dir=str(root), training_seed=11)
    state = root / 'state.json'
    state.write_text(json.dumps(dict(policy_version=1)))
    rows = training_curve_rows([run])
    assert rows == [dict(training_seed=11, kind='sampled_training', update=1, training_episodes=64, mean_return=-.4),
                    dict(training_seed=11, kind='greedy_validation', update=0, training_episodes=0, mean_return=.2)]
    report = tmp_path / 'report'
    report.mkdir()
    plots(dict(conditions=[]), dict(training_runs=[run]), report)
    assert (report / 'return-vs-training-episodes.png').stat().st_size > 1000
    with (report / 'training-return-curve.csv').open() as handle:
        saved = list(csv.DictReader(handle))
    assert [row['kind'] for row in saved] == ['sampled_training', 'greedy_validation']
    assert [int(row['training_episodes']) for row in saved] == [64, 0]
    state.unlink()
    assert training_curve_rows([run]) == rows[1:]


def test_return_curves_reject_partial_batches_and_nonfinite_measurements(tmp_path):
    (tmp_path / 'update-metrics').mkdir()
    (tmp_path / 'state.json').write_text(json.dumps(dict(policy_version=1)))
    path = tmp_path / 'update-metrics/update-0001.json'
    base = dict(update=1, episodes=64, training_episodes=64, mean_return=0.)
    for changed, message in [(dict(episodes=63), '64-trajectory'),
                             (dict(training_episodes=63), '64-trajectory'),
                             (dict(mean_return=float('nan')), 'nonfinite')]:
        path.write_text(json.dumps(dict(metrics=base | changed)))
        with pytest.raises(ValueError, match=message):
            training_curve_rows([dict(run_dir=str(tmp_path), training_seed=11)])


def deferred_curve_run(plan, run):
    conditions = json.loads(plan.read_text())['layouts']['4']
    return dict(run_dir=str(run), training_seed=11,
                validation_summaries=[entry['summary'] for entry in conditions if entry['attention'] == 'ps'])


def test_deferred_validation_curves_read_external_results_without_moving_them(selection_run, tmp_path):
    plan, run = selection_run
    entry = deferred_curve_run(plan, run)
    assert not (run / 'validation').exists()
    rows = training_curve_rows([entry])
    assert [row['update'] for row in rows] == [0, 16, 128]
    assert [row['mean_return'] for row in rows] == pytest.approx([-.5, 2 / 3, 2 / 3])
    assert all(row['kind'] == 'greedy_validation' for row in rows)
    output = tmp_path / 'report'
    output.mkdir()
    plots(dict(conditions=[]), dict(training_runs=[entry]), output)
    with (output / 'training-return-curve.csv').open() as handle:
        saved = list(csv.DictReader(handle))
    assert [int(row['training_episodes']) for row in saved] == [0, 1024, 8192]
    assert (output / 'return-vs-training-episodes.png').stat().st_size > 1000


@pytest.mark.parametrize('change,message', [
    ('test_split', 'only PS validation'),
    ('full_attention', 'only PS validation'),
    ('controller', 'scientific identities'),
    ('partial', 'expected 128'),
    ('checkpoint', 'another checkpoint'),
    ('episode_controller', 'another generation controller'),
    ('return', 'differs from its records'),
    ('future', 'uncommitted checkpoint'),
    ('duplicate', 'one result per checkpoint'),
    ('empty', 'cannot be empty'),
])
def test_deferred_validation_curves_reject_mixed_or_incomplete_sources(selection_run, change, message):
    from pathlib import Path

    plan, run = selection_run
    entry = deferred_curve_run(plan, run)
    summary = Path(entry['validation_summaries'][-1])
    path, value = summary, json.loads(summary.read_text())
    if change == 'test_split':
        value['split'] = 'test'
    elif change == 'full_attention':
        value['attention'] = 'full'
    elif change == 'controller':
        path = summary.parent / 'evaluation-protocol.json'
        value = json.loads(path.read_text())
        value['config']['thinking_token_limit'] = 64
    elif change == 'partial':
        path = summary.parent / 'episodes.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        path = None
    elif change in ('checkpoint', 'episode_controller'):
        path = summary.parent / 'episodes.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0].update(dict(policy_version=16) if change == 'checkpoint' else dict(thinking_token_limit=64))
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        path = None
    elif change == 'return':
        value['mean_return'] = 999
    elif change == 'future':
        path = run / 'state.json'
        value = json.loads(path.read_text()) | dict(policy_version=127)
    elif change == 'duplicate':
        entry['validation_summaries'].append(str(summary))
        path = None
    elif change == 'empty':
        entry['validation_summaries'] = []
        path = None
    if path is not None:
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        training_curve_rows([entry])


def episodes(seed=11, delta=0.):
    return [dict(board_seed=2000000 + index, episode_id=f's{seed}-b{index}', greedy=True,
        training_return=.5 + delta, score=8, response_count=48, done=True,
        censored=False, generated_tokens=100, raw_transcript_tokens=200, seconds=2.,
        invalid_actions=0, format_errors=0, attention='ps', window=1024, max_responses=96)
        for index in range(4)]


def condition(tmp_path, rows, **kwargs):
    path = tmp_path / (kwargs.get('name', 'trained') + '.jsonl')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return dict(name='trained', policy='PS RL', training_seed=11, split='test', attention='ps',
        window=1024, greedy=True, episodes=str(path), expected_boards=4) | kwargs


@pytest.mark.parametrize('split,start,count,cols,pairs,limit', [
    ('post320_test', 7_000_000, 128, 4, 8, 96),
    ('post320_larger', 8_000_000, 4, 8, 16, 192),
])
def test_post320_report_preserves_fresh_seeds_and_board_protocol(tmp_path, split, start, count, cols, pairs, limit):
    rows = [dict(episodes()[0], board_seed=start + index, episode_id=f'fresh-b{start + index}',
        board_rows=4, board_cols=cols, distinct_ranks=8, total_pairs=pairs, score=pairs,
        response_count=limit // 2, max_responses=limit) for index in range(count)]
    entry = condition(tmp_path, rows, split=split, expected_boards=count)
    result = load_condition(entry)
    assert result['complete_split'] is True
    assert result['mean_return'] == .5 and result['mean_pairs'] == pairs
    assert result['total_pairs'] == pairs and result['max_responses'] == limit
    assert result['evaluation_scope'] == ('window_sweep' if split == 'post320_test' else split)
    rows[0]['board_seed'] = 5_000_000
    condition(tmp_path, rows)
    with pytest.raises(ValueError, match='reserved board seeds'):
        load_condition(entry)


@pytest.mark.parametrize('field,expected,actual', [
    ('policy_version', 16, 0),
    ('policy_identity', 'selected-policy', 'another-policy'),
    ('thinking_token_limit', 128, 64),
    ('action_token_limit', None, 64),
    ('thinking_budget_transition', '\n\nOkay, time to give the answer.</think>\n\n', None),
])
def test_final_report_rejects_a_mixed_checkpoint_or_controller(tmp_path, field, expected, actual):
    rows = episodes()
    for row in rows:
        row[field] = expected
    identity = {field: expected} if field.startswith('policy_') else dict(controller={field: expected})
    entry = condition(tmp_path, rows, **identity)
    load_condition(entry)
    rows[1][field] = actual
    condition(tmp_path, rows)
    with pytest.raises(ValueError, match='different checkpoint|different generation controller'):
        load_condition(entry)


def test_final_report_requires_an_explicit_uncapped_action_controller(tmp_path):
    entry = condition(tmp_path, episodes(), controller=dict(action_token_limit=None))
    with pytest.raises(ValueError, match='different generation controller'):
        load_condition(entry)


@pytest.mark.parametrize('full,gap,status', [(1., .5, 'GO'), (.79, .5, 'HOLD'), (1., .09, 'HOLD')])
def test_report_prints_exact_deferred_qualification_gates_and_locks_receipt(tmp_path, full, gap, status):
    selection = tmp_path / 'selected.json'
    selected = dict(split='validation', attention='ps', episodes=128,
        full_context_matched_fraction=full, full_minus_base_ps_matched_fraction=gap,
        competence_and_forgetting_status=status, update=16, validation_return=.5)
    selection.write_text(json.dumps(selected))
    manifest = dict(conditions=[], checkpoint_selection=dict(path=str(selection),
        sha256=hashlib.sha256(selection.read_bytes()).hexdigest()))
    result = summarize_manifest(manifest)
    gate = result['deferred_qualification']
    assert gate['status'] == status and gate['full_context_matched_fraction'] == full
    assert gate['full_minus_base_ps_matched_fraction'] == gap
    output = tmp_path / 'report'
    save_tables(result, output)
    text = (output / 'README.md').read_text()
    assert f'**{status}**' in text and f'{full:.6f}' in text and f'{gap:.6f}' in text
    assert 'separate from final-test results' in text
    selection.write_text(json.dumps(selected | dict(update=128)))
    with pytest.raises(ValueError, match='receipt changed'):
        summarize_manifest(manifest)


def test_paired_intervals_preserve_board_pairing_not_episode_order():
    left, right = episodes(), episodes()
    for index in range(4):
        left[index]['training_return'] = index + .25
        right[index]['training_return'] = index
    result = paired_bootstrap(left, list(reversed(right)))
    assert result['mean_difference'] == .25
    assert result['ci95'] == [.25, .25] and result['resamples'] == 10000
    with pytest.raises(ValueError, match='identical board'):
        paired_bootstrap(left, right[:-1])


def test_training_variability_is_over_independent_training_seeds():
    result = seed_summary([dict(training_seed=seed, mean_return=value)
                           for seed, value in [(11, .1), (22, .2), (33, .3)]])
    assert result['independent_training_runs'] == 3
    assert result['mean_return'] == pytest.approx(.2)
    assert result['training_seed_std'] == pytest.approx(.1)
    assert seed_summary([dict(training_seed=11, mean_return=.1)])['training_seed_std'] is None
    with pytest.raises(ValueError, match='independent training'):
        seed_summary([dict(training_seed=11, mean_return=.1)] * 512)


def test_seed_statistics_cover_task_metrics_and_retain_undefined_denominators():
    rows = [dict(training_seed=seed, mean_return=score / 10, mean_pairs=score,
                 completion_rate=score / 8, completed_only_mean_attempts=attempts)
            for seed, score, attempts in [(11, 2, None), (22, 4, 40), (33, 6, 60)]]
    result = seed_summary(rows)
    assert result['metrics']['mean_pairs'] == dict(mean=4., sample_std=2., defined_training_runs=3)
    assert result['metrics']['completion_rate']['sample_std'] == .25
    completed = result['metrics']['completed_only_mean_attempts']
    assert completed['defined_training_runs'] == 2 and completed['mean'] == 50
    assert completed['sample_std'] == pytest.approx(2**.5 * 10)


def test_reveal_age_statistics_keep_empty_bins_undefined(tmp_path):
    import gzip
    import hashlib
    from .test_rng_memory_metrics import CharacterTokenizer, example

    record, game = example()
    game_path, token_path = tmp_path / 'game.json', tmp_path / 'tokens.json.gz'
    game_path.write_text(json.dumps(game))
    record['native_game_json'] = str(game_path)
    with gzip.open(token_path, 'wt') as handle:
        json.dump(record, handle)
    row = dict(board_seed=2000000, token_record=str(token_path),
               token_record_sha256=hashlib.sha256(token_path.read_bytes()).hexdigest())
    diagnostic = analyze_condition(dict(rows=[row]), CharacterTokenizer())
    assert diagnostic['known_match_old_utilization'] == 1.
    assert diagnostic['known_match_old_opportunities'] == 1
    assert diagnostic['known_match_recent_utilization'] is None
    stats = seed_summary([dict(training_seed=11, mean_return=0., **diagnostic)])['metrics']
    assert stats['known_match_recent_utilization'] == dict(mean=None, sample_std=None, defined_training_runs=0)
    assert stats['known_match_old_utilization'] == dict(mean=1., sample_std=None, defined_training_runs=1)


def test_context_censor_keeps_pairs_but_has_explicit_resource_reward(tmp_path):
    rows = episodes()
    rows[0].update(censored=True, score=2, done=False, response_count=77, training_return=-1.)
    result = load_condition(condition(tmp_path, rows))
    assert result['mean_pairs'] == 6.5
    assert result['mean_return'] == pytest.approx(.125)
    assert result['resource_terminations'] == 1
    assert result['mean_attempts_to_completion'] == 60


def test_incomplete_splits_and_sampled_training_rows_cannot_be_final_results(tmp_path):
    entry = condition(tmp_path, episodes()[:-1])
    with pytest.raises(ValueError, match='partial reports'):
        load_condition(entry)
    assert not load_condition(entry, allow_partial=True)['complete_split']
    rows = episodes(); rows[0]['greedy'] = False
    with pytest.raises(ValueError, match='greedy evaluation'):
        load_condition(condition(tmp_path, rows))
    rows = episodes(); rows[0]['board_seed'] = 1000000
    with pytest.raises(ValueError, match='reserved board seeds'):
        load_condition(condition(tmp_path, rows))


def test_system_metrics_cannot_be_taken_from_a_different_attention_condition(tmp_path):
    summary = tmp_path / 'summary.json'
    values = dict(episodes=4, attention='ps', window=1024, peak_kv_pool_allocated_bytes=2048,
        completed_cohort_inference_seconds=10., completed_cohort_gpu_hours=.01, unfinished_cohort_accounting=0)
    summary.write_text(json.dumps(values))
    entry = condition(tmp_path, episodes(), summary=str(summary))
    assert load_condition(entry)['peak_kv_pool_allocated_bytes'] == 2048
    summary.write_text(json.dumps(dict(values, window=512)))
    with pytest.raises(ValueError, match='different evaluation condition'):
        load_condition(entry)


def test_base_control_is_not_mislabeled_full_rl_and_small_gap_has_no_recovery(tmp_path):
    treatment = episodes()
    ps = episodes(); full = episodes()
    for row in ps:
        row.update(score=6, training_return=.25)
    for row in full:
        row.update(score=6, training_return=.25, attention='full')
    manifest = dict(conditions=[condition(tmp_path, treatment),
        condition(tmp_path, ps, name='base-ps', policy='base', training_seed=None),
        condition(tmp_path, full, name='base-full', policy='base', training_seed=None, attention='full')],
        comparisons=[dict(treatment='trained', sliding_control='base-ps', full_control='base-full')])
    result = summarize_manifest(manifest)
    comparison = result['paired_comparisons'][0]
    assert comparison['mean_difference'] == .25
    assert comparison['recovery'] is None
    assert comparison['full_rl_control_skipped']


def test_size_generalization_uses_sixteen_pairs_and_separate_table(tmp_path):
    standard = condition(tmp_path, episodes())
    rows = [dict(row, board_seed=3000000 + index, board_rows=4, board_cols=8, distinct_ranks=8,
                 total_pairs=16, max_responses=192, training_return=.25, done=False)
            for index, row in enumerate(episodes())]
    larger = condition(tmp_path, rows, name='larger', split='larger')
    result = summarize_manifest(dict(conditions=[standard, larger]))
    assert result['conditions'][1]['matched_fraction'] == .5
    assert result['conditions'][1]['mean_attempts_to_completion'] == 192
    save_tables(result, tmp_path / 'report')
    import csv
    with (tmp_path / 'report/size-generalization.csv').open() as handle:
        values = list(csv.DictReader(handle))
    assert len(values) == 1 and values[0]['total_pairs'] == '16' and values[0]['name'] == 'larger'
    assert '/16' in (tmp_path / 'report/README.md').read_text()
    with (tmp_path / 'report/main-comparison.csv').open() as handle:
        assert [value['name'] for value in csv.DictReader(handle)] == ['trained']
    rows[0]['total_pairs'] = 8
    with pytest.raises(ValueError, match='board sizes'):
        load_condition(condition(tmp_path, rows, name='wrong', split='larger'))


def test_window_sweeps_and_full_test_remain_distinct_in_statistics_curves_and_tables(tmp_path):
    entries = []
    for seed in (11, 22, 33):
        for window, count, label in [(1024, 512, 'main'), (1024, 128, 'sweep'), (512, 128, 'sweep')]:
            rows = [dict(episodes(seed)[0], board_seed=2000000 + index,
                         episode_id=f's{seed}-b{index}', window=window) for index in range(count)]
            entries.append(condition(tmp_path, rows, name=f's{seed}-{window}-{label}', training_seed=seed,
                                     window=window, expected_boards=count))
    result = summarize_manifest(dict(conditions=entries))
    groups = result['training_seed_summaries']
    assert len(groups) == 3 and all(group['independent_training_runs'] == 3 for group in groups)
    assert sorted(group['boards_per_training_seed'] for group in groups) == [128, 128, 512]
    curves = window_series(result['conditions'])
    assert len(curves) == 6
    for rows in curves.values():
        assert len({row['board_seeds_sha256'] for row in rows}) == 1
        assert [row['window'] for row in rows] == ([512,1024] if rows[0]['episodes'] == 128 else [1024])
    save_tables(result, tmp_path / 'report')
    import csv
    with (tmp_path / 'report/window-sweep.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    with (tmp_path / 'report/main-comparison.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (tmp_path / 'report/training-seed-summary.csv').open() as handle:
        pairs = [row for row in csv.DictReader(handle) if row['metric'] == 'mean_pairs']
    assert len(pairs) == 3 and all(row['defined_training_runs'] == '3' for row in pairs)
