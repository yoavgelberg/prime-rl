import json

import pytest

from .rng_report import analyze_condition, load_condition, paired_bootstrap, save_tables, seed_summary, summarize_manifest, window_series


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
