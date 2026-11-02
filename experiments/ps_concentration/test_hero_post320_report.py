import csv
import gzip
import json
from pathlib import Path

import pytest

from . import hero_post320_final, hero_post320_report
from .hero_exports import file_hash
from .hero_memory import aggregate
from .hero_v2 import identity, read, write
from .test_hero_post320_final import post320_plan


def committed(config, update=321):
    root = Path(config.output)
    write(root / 'state.json', dict(policy_version=update, identity=identity(config)))
    for value in range(321, update + 1):
        write(root / f'update-metrics/update-{value:04d}.json', dict(metrics=dict(
            update=value, episodes=128, training_episodes=value * 128,
            exact_categorical_kl_active=True, mean_return=.4, mean_native_return=.5,
            mean_pairs=8, mean_attempts=48, seconds=180., mean_per_game_tokens_per_second=50.,
            generated_tokens=800000, gradient_norm=.1, kl_per_generated_token=.06)))


def test_partial_report_never_labels_sampled_training_as_a_final_test(post320_plan, tmp_path):
    path, _, config = post320_plan
    final = hero_post320_final.prepare(path, tmp_path / 'final', gpus=8,
        source_root=tmp_path / 'captured', source_sha256='c' * 64)
    committed(config)
    future = Path(config.output) / 'update-metrics/update-0322.json'
    write(future, dict(metrics=dict(update=322, mean_return=99.)))
    output = tmp_path / 'report'
    result = hero_post320_report.report(tmp_path / 'final/plan.json', output, allow_partial=True)
    assert result['status'] == 'PARTIAL' and result['committed_update'] == 321
    assert result['missing_conditions'] == [entry['condition'] for entry in final['conditions']]
    assert result['missing_memory'] is True and result['paired_comparison'] is None
    assert result['independent_training_lineages'] == 1 and result['training_seed_std'] is None
    with (output / 'training-curve.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and float(rows[0]['shaped_reward']) == .4
    assert int(rows[0]['new_training_episodes']) == 128 and int(rows[0]['lineage_episodes']) == 41088
    assert str(future) not in result['input_sha256']
    assert read(output / 'training-state.json') == result['training_snapshot']['state']
    assert read(output / 'resolved_config.json') == result['training_snapshot']['config']
    committed(config, update=322)
    assert read(output / 'training-state.json')['policy_version'] == 321
    assert not list(output.glob('final-return-vs-window.*'))
    assert all((output / f'return-vs-training-episodes.{ext}').stat().st_size > 1000 for ext in ('png', 'pdf'))
    with pytest.raises(ValueError, match='448 committed'):
        hero_post320_report.report(tmp_path / 'final/plan.json', tmp_path / 'strict')


@pytest.mark.parametrize('change,error', [
    ('identity', 'declared continuation'), ('missing_batch', 'No such file'),
    ('partial_batch', 'complete128'), ('wrong_kl', 'complete128'), ('past_target', 'committed320'),
])
def test_training_rows_reject_wrong_identity_missing_batches_and_wrong_objective(post320_plan, change, error):
    _, _, config = post320_plan
    committed(config, update=322)
    state = Path(config.output) / 'state.json'
    metrics = Path(config.output) / 'update-metrics/update-0321.json'
    if change == 'identity':
        write(state, read(state) | dict(identity={}))
    elif change == 'past_target':
        write(state, read(state) | dict(policy_version=449))
    elif change == 'missing_batch':
        metrics.unlink()
    else:
        value = read(metrics)
        value['metrics'].update(dict(episodes=127) if change == 'partial_batch'
            else dict(exact_categorical_kl_active=False))
        write(metrics, value)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        hero_post320_report.training_rows(config)


def window_condition(window=1024, count=512):
    rows = [dict(board_seed=7000000 + i, score=0 if i < 128 else 8,
        training_return=-1. if i < 128 else 0., done=i >= 128, response_count=96,
        generated_tokens=10, raw_transcript_tokens=20, format_errors=0, invalid_actions=0, seconds=1.)
        for i in range(count)]
    return dict(name=f'continued-ps-w{window}-post320_test', arm='continued', split='post320_test',
        attention='ps', window=window, expected_boards=count, rows=rows)


def test_window_curve_derives_the_same128_boards_instead_of_mixing512_and128():
    result = hero_post320_report.window_rows([window_condition(), window_condition(512, 128)])
    assert len(result) == 2 and all(row['mean_pairs'] == 0 and row['mean_return'] == -1. for row in result)
    assert all(row['episodes'] == 128 for row in result)
    assert [row['derived_subset'] for row in result] == [False, True]
    changed = window_condition(512, 128)
    changed['rows'][0]['board_seed'] = 5000000
    with pytest.raises(ValueError, match='fresh7M'):
        hero_post320_report.window_rows([changed])
    with pytest.raises(ValueError, match='exactly one'):
        hero_post320_report.window_rows([window_condition(), window_condition()])


def test_primary_is_paired_parent_comparison_without_requiring_full_attention():
    rows = [dict(name=name, split='post320_test', attention='ps', window=1024,
        expected_boards=512, mean_return=value,
        rows=[dict(board_seed=7000000 + i, training_return=value) for i in range(512)])
        for name, value in [('continued', .6), ('parent', .4)]]
    final = dict(primary=['continued', 'parent'])
    value = hero_post320_report.comparison(rows, final)
    assert value['mean_difference'] == pytest.approx(.2) and value['boards'] == 512
    assert value['resamples'] == 10000 and value['training_seed_std'] is None
    assert value['independent_training_lineages'] == 1 and 'recovery' not in value
    rows[1]['rows'][0]['board_seed'] = 5000000
    with pytest.raises(ValueError, match='identical board seed'):
        hero_post320_report.comparison(rows, final)


def test_zero_qualified_states_is_a_finished_negative_memory_result_not_missing(post320_plan, tmp_path):
    path, _, config = post320_plan
    final = hero_post320_final.prepare(path, tmp_path / 'final', gpus=8,
        source_root=tmp_path / 'captured', source_sha256='c' * 64)
    entry, = [row for row in final['conditions'] if row['split'] == 'intervention']
    entry['checkpoint'] = str(tmp_path / 'checkpoint')
    episodes = Path(entry['summary']).parent / 'episodes.jsonl'
    episodes.parent.mkdir(parents=True)
    episodes.write_text('')
    metadata = Path(entry['checkpoint']) / 'experiment.json'
    write(metadata, dict(policy_version=448))
    output = tmp_path / 'final/memory/continued-w1024'
    write(output / 'collection-inputs.json', dict(episodes=str(episodes), episodes_sha256=file_hash(episodes),
        checkpoint=entry['checkpoint'], maximum=128, boundary='before_thinking'))
    write(output / 'session.json', dict(checkpoint=str(Path(entry['checkpoint']).resolve()),
        samples=32, checkpoint_metadata=file_hash(metadata), states=[]))
    state_path = output / 'collection/states.jsonl.gz'
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(gzip.compress(b''))
    summary = aggregate([], output / 'states', checkpoint=entry['checkpoint'])
    write(output / 'summary.json', summary)
    result = hero_post320_report.memory_result(tmp_path / 'final/plan.json', final)
    assert result['result']['qualified_states'] == 0
    assert result['result']['causal_textual_memory_established'] is False
    write(output / 'summary.json', summary | dict(causal_textual_memory_established=True))
    with pytest.raises(ValueError, match='every qualified-state'):
        hero_post320_report.memory_result(tmp_path / 'final/plan.json', final)


def test_changed_training_plan_cannot_be_reported_as_the_frozen_study(post320_plan, tmp_path):
    path, _, config = post320_plan
    hero_post320_final.prepare(path, tmp_path / 'final', gpus=8,
        source_root=tmp_path / 'source', source_sha256='c' * 64)
    committed(config)
    write(path, read(path) | dict(initial_target_update=449))
    with pytest.raises(ValueError, match='frozen post320'):
        hero_post320_report.report(tmp_path / 'final/plan.json', tmp_path / 'report', allow_partial=True)


def test_complete_report_keeps_sizes_windows_memory_and_board_uncertainty_separate(
        post320_plan, tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from .overlap_config import board_protocol
    from .rng_report import load_condition

    path, _, config = post320_plan
    final_path = tmp_path / 'final/plan.json'
    final = hero_post320_final.prepare(path, final_path.parent, gpus=8,
        source_root=tmp_path / 'source', source_sha256='c' * 64)
    committed(config, update=448)
    loaded = {}
    for entry in final['conditions']:
        board = board_protocol(entry['split']).identity()
        attempts = dict(continued=32, parent320=48, base=64)[entry['arm']]
        episodes = Path(entry['summary']).parent / 'episodes.jsonl'
        episodes.parent.mkdir(parents=True)
        records = [dict(board_seed=entry['first_board_seed'] + index, greedy=True,
            episode_id=f'{entry["condition"]}-{index}', score=board['total_pairs'],
            policy_version=entry['policy_version'],
            response_count=attempts, done=True, censored=False, generated_tokens=100,
            raw_transcript_tokens=200, format_errors=0, invalid_actions=0, seconds=2.,
            training_return=1 - attempts / board['max_responses'],
            attention=entry['attention'], window=entry['window'], **board) for index in range(entry['count'])]
        episodes.write_text(''.join(json.dumps(row) + '\n' for row in records))
        write(Path(entry['summary']), {})
        loaded[entry['condition']] = load_condition(dict(name=entry['condition'], policy=entry['arm'],
            training_seed=11 if entry['policy_version'] else None, policy_version=entry['policy_version'],
            episodes=str(episodes), expected_boards=entry['count'], greedy=True, split=entry['split'],
            attention=entry['attention'], window=entry['window']))
    monkeypatch.setattr(hero_post320_report, 'condition_result', lambda entry: loaded[entry['condition']])
    analyzed = []
    def analyze(condition, tokenizer, *, examples=None):
        analyzed.append((condition['policy'], examples))
        return dict(known_match={age:dict(opportunities=2, correct=1, utilization=.5)
            for age in ('<=W', '(W,2W]', '>2W')}, example_transcripts=20 if examples else 0)
    monkeypatch.setattr(hero_post320_report, 'analyze_condition', analyze)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    memory = dict(result=aggregate([], tmp_path / 'states', checkpoint=tmp_path / 'checkpoint'))
    monkeypatch.setattr(hero_post320_report, 'memory_result', lambda *a: memory)
    output = tmp_path / 'report'
    result = hero_post320_report.report(final_path, output)
    assert result['status'] == 'COMPLETE' and len(result['conditions']) == 22
    assert result['paired_comparison']['mean_difference'] == pytest.approx(1 / 6)
    assert result['paired_comparison']['boards'] == 512 and result['training_seed_std'] is None
    assert result['intervention']['result']['causal_textual_memory_established'] is False
    assert len(analyzed) == 3 and sum(examples is not None for _, examples in analyzed) == 1
    with (output / 'main-comparison.csv').open() as handle:
        main = list(csv.DictReader(handle))
    with (output / 'size-generalization.csv').open() as handle:
        larger = list(csv.DictReader(handle))
    with (output / 'window-sweep.csv').open() as handle:
        windows = list(csv.DictReader(handle))
    with (output / 'intervention-table.csv').open() as handle:
        interventions = list(csv.DictReader(handle))
    assert len(main) == 6 and all(row['total_pairs'] == '8' for row in main)
    assert len(larger) == 3 and all(row['total_pairs'] == '16' and row['max_responses'] == '192' for row in larger)
    assert len(windows) == 15 and all(row['episodes'] == '128' for row in windows)
    assert len(interventions) == 6 and all(row['qualified_states'] == '0' for row in interventions)
    for name in ('return-vs-training-episodes', 'final-return-vs-window', 'known-match-utilization-vs-age'):
        assert all((output / f'{name}.{ext}').stat().st_size > 1000 for ext in ('png', 'pdf'))
