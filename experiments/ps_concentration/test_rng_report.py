import json

import pytest

from .rng_report import load_condition, paired_bootstrap, seed_summary, summarize_manifest


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
