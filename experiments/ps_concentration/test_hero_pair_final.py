from dataclasses import asdict
import json
from pathlib import Path

import pytest

from . import hero_pair_final as final
from .hero_pair_plan import prepare
from .hero_pair_report import comparison, training_rows
from .hero_v2 import identity, read, write
from .overlap_config import RunConfig, episode_metrics
from .test_hero_v2 import config


@pytest.fixture
def paired(tmp_path, monkeypatch):
    parent_run = tmp_path / 'parent'
    write(parent_run / 'resolved_config.json', asdict(config()))
    plan = prepare(parent_run, tmp_path / 'paired', source_root='/captured', source_sha256='b' * 64)
    path = tmp_path / 'paired/plan.json'
    monkeypatch.setattr(final, 'policy_cache', lambda plan_path, arm, version:
        final.policy_location(read(plan_path), arm, version) if version == 128
        else tmp_path / 'cache' / arm / str(version))
    for cfg in final.branch_configs(plan).values():
        run = Path(cfg.output)
        write(run / 'resolved_config.json', asdict(cfg))
        write(run / 'state.json', dict(policy_version=192, identity=identity(cfg)))
        write(run / 'training-complete.json', dict(policy_version=192, reached_target=True))
        checkpoint = run / 'checkpoints/latest.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'CPU fixture for completed optimizer state')
        write(checkpoint.with_suffix('.pt.json'), dict(policy_version=192,
            identity=identity(cfg), bytes=checkpoint.stat().st_size))
    return path, plan


def validation_result(entry, *, attempts):
    cfg = RunConfig(**read(entry['argv'][entry['argv'].index('--config') + 1]))
    rows = [dict(board_seed=1100000 + i, episode_id=f'{entry["condition"]}-{i}',
        policy_version=entry['policy_version'], policy_identity=entry['policy_identity'],
        greedy=True, attention='ps', window=entry['window'], score=8, total_pairs=8,
        response_count=attempts, max_responses=96, done=True, censored=False,
        training_return=1 - attempts / 96, invalid_actions=0, format_errors=0,
        generated_tokens=500, raw_transcript_tokens=1000, seconds=1.,
        thinking_token_limit=cfg.thinking_token_limit, action_token_limit=cfg.action_token_limit,
        thinking_budget_transition=cfg.thinking_budget_transition) for i in range(128)]
    folder = Path(entry['summary']).parent
    folder.mkdir(parents=True)
    (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(folder / 'summary.json', dict(episode_metrics(rows), split='validation', greedy=True,
        attention='ps', window=cfg.window, policy_version=entry['policy_version'],
        policy_identity=entry['policy_identity']))
    write(folder / 'evaluation-protocol.json', dict(config=cfg.model_dump()))
    write(Path(entry['checkpoint']) / 'experiment.json',
        dict(entry['policy_training_identity'], policy_version=entry['policy_version']))


@pytest.fixture
def validated(paired, tmp_path):
    path, plan = paired
    validation = final.prepare_validation(path, tmp_path / 'validation', gpus=8)
    for entry in validation['conditions']:
        validation_result(entry, attempts={128: 40, 144: 32, 160: 32, 176: 35, 192: 36}[entry['policy_version']])
    return path, plan, validation


def test_both_branches_must_finish_before_any_selection(paired):
    _, plan = paired
    final.completed_branches(plan)
    cfg = final.branch_configs(plan)['w256']
    write(Path(cfg.output) / 'state.json', dict(policy_version=191, identity=identity(cfg)))
    with pytest.raises(ValueError, match='complete training budget'):
        final.completed_branches(plan)


def test_validation_preserves_parent_provenance_and_selects_earliest_tie(validated):
    _, plan, validation = validated
    selected = final.select(validation)
    assert {row['version'] for row in selected['selected'].values()} == {144}
    assert {row['additional_episodes'] for row in selected['selected'].values()} == {2048}
    assert selected['selected']['w256']['training_window'] == 256
    parents = [entry for entry in validation['conditions'] if entry['policy_version'] == 128]
    assert len({entry['checkpoint'] for entry in parents}) == 1
    assert all(entry['policy_training_identity'] == plan['parent_identity'] for entry in parents)
    assert {entry['window'] for entry in parents} == {256, 1024}


@pytest.mark.parametrize('change', [dict(split='test'), dict(window=64), dict(count=32)])
def test_selection_cannot_use_test_sweep_or_short_screening(validated, change):
    _, _, validation = validated
    validation['conditions'][0].update(change)
    with pytest.raises(ValueError, match='fixed validation split'):
        final.select(validation)


def test_selection_requires_complete_unselected_checkpoints(validated):
    _, _, validation = validated
    path = Path(validation['conditions'][-1]['summary']).parent / 'episodes.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    with pytest.raises(ValueError, match='expected 128'):
        final.select(validation)


def test_selection_rejects_mixed_checkpoint_identity(validated):
    _, _, validation = validated
    path = Path(validation['conditions'][-1]['summary']).parent / 'episodes.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['policy_identity'] = 'another-checkpoint'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='checkpoint identity'):
        final.select(validation)


def test_selection_rejects_changed_export_training_identity(validated):
    _, _, validation = validated
    entry = validation['conditions'][-1]
    path = Path(entry['checkpoint']) / 'experiment.json'
    write(path, read(path) | dict(window=512))
    with pytest.raises(ValueError, match='Export metadata'):
        final.select(validation)


def test_final_matrix_uses_selected_models_same_windows_and_disjoint_splits(validated, tmp_path):
    path, _, validation = validated
    selection = final.select(validation)
    selection['selected']['w256'].update(version=128, shared_parent_baseline=True)
    result = final.prepare_final(path, selection, tmp_path / 'final', gpus=4)
    assert len(result['conditions']) == 32
    primary = [entry for entry in result['conditions'] if entry['condition'] in result['primary']]
    assert {entry['window'] for entry in primary} == {256}
    assert {entry['count'] for entry in primary} == {512}
    assert {entry['first_board_seed'] for entry in primary} == {2000000}
    assert {entry['policy_version'] for entry in primary} == {128, 144}
    assert len([entry for entry in result['conditions'] if entry['split'] == 'intervention']) == 2
    assert all(entry['first_board_seed'] == 4000000 for entry in result['conditions'] if entry['split'] == 'intervention')
    assert all(entry['first_board_seed'] == 3000000 for entry in result['conditions'] if entry['split'] == 'larger')
    for arm in ('w1024', 'w256'):
        sweeps = [entry for entry in result['conditions'] if entry['condition'].startswith(f'{arm}-window-')]
        assert {entry['window'] for entry in sweeps} == {64, 128, 256, 512, 1024, 2048, 4096}
        assert all(entry['count'] == 128 for entry in sweeps)


@pytest.mark.parametrize('gpus', [4, 8])
def test_reference_memory_preserves_selection_controller_and_reserved_boards(validated, tmp_path, gpus):
    from . import hero_reference_memory as reference
    from .hero_protocol import THINKING_TRANSITION

    path, _, validation = validated
    selection = final.select(validation)
    selection['selected']['w256'].update(version=128, shared_parent_baseline=True)
    original = json.loads(json.dumps(selection))
    entries = reference.prepare(path, selection, tmp_path / f'reference-{gpus}', gpus=gpus)
    assert selection == original
    assert {entry['arm']: entry['policy_version'] for entry in entries} == {'w1024': 144, 'w256': 128}
    assert all(entry['attention'] == 'ps' and entry['window'] == 1024 for entry in entries)
    assert all(entry['split'] == 'intervention' and entry['count'] == 512 for entry in entries)
    assert all((entry['first_board_seed'], entry['last_board_seed']) == (4000000, 4000511) for entry in entries)
    for entry in entries:
        cfg = RunConfig(**read(entry['argv'][entry['argv'].index('--config') + 1]))
        assert cfg.thinking_token_limit == 128 and cfg.action_token_limit == 32
        assert cfg.thinking_budget_transition == THINKING_TRANSITION
        assert cfg.evaluation_concurrency * gpus == 128


def test_reference_memory_rejects_incomplete_main_or_reselected_checkpoints(validated, tmp_path, monkeypatch):
    from . import hero_reference_memory as reference

    path, _, validation = validated
    digest = final.file_hash(path)
    monkeypatch.setattr(reference, 'MAIN_PLAN_SHA256', digest)
    main = tmp_path / 'main'
    write(main / 'status.json', dict(status='CONTINUATIONS_COMPLETE_FINAL_EVAL_PENDING'))
    with pytest.raises(ValueError, match='Complete the main study'):
        reference.validate_main(path, main)
    selection = final.select(validation)
    write(main / 'selection.json', selection)
    write(main / 'final/plan.json', dict(plan_sha256=digest, selection=selection))
    write(main / 'status.json', dict(status='PAIRED_RESULTS_AND_INTERVENTIONS_COMPLETE'))
    assert reference.validate_main(path, main)[1] == selection
    altered = json.loads(json.dumps(selection))
    altered['selected']['w1024']['version'] = 192
    write(main / 'selection.json', altered)
    with pytest.raises(ValueError, match='validation-selected'):
        reference.validate_main(path, main)


def test_paired_comparison_reports_board_uncertainty_not_seed_replication():
    rows = []
    for name, value in [('w256-ps-w256-test', .6), ('w1024-ps-w256-test', .3), ('w1024-full-test', .7)]:
        rows.append(dict(name=name, mean_return=value,
            rows=[dict(board_seed=2000000 + i, training_return=value) for i in range(512)]))
    result = comparison(rows)
    assert result['mean_difference'] == pytest.approx(.3)
    assert result['recovery'] == pytest.approx(.75)
    assert result['boards'] == 512 and result['independent_training_seeds'] == 1
    assert 'not training variability' in result['uncertainty']


def test_training_curve_keeps_shared_parent_and_new_episodes_distinct(paired):
    _, plan = paired
    roots = [(Path(plan['parent_run']), 1, 128)]
    roots.extend((Path(entry['output']), 129, 192) for entry in plan['branches'])
    for root, first, last in roots:
        write(root / 'state.json', dict(policy_version=last))
        for update in range(first, last + 1):
            write(root / f'update-metrics/update-{update:04d}.json', dict(metrics=dict(update=update,
                episodes=128, training_episodes=update * 128, mean_return=.3, mean_native_return=.4,
                mean_pairs=8, mean_attempts=50, seconds=300, mean_per_game_tokens_per_second=50)))
    rows, sources = training_rows(plan)
    assert len(rows) == len(sources) == 256
    assert len([row for row in rows if row['arm'] == 'shared-parent']) == 128
    branches = [row for row in rows if row['arm'] != 'shared-parent']
    assert min(row['new_episodes'] for row in branches) == 128
    assert max(row['new_episodes'] for row in branches) == 8192
    assert max(row['lineage_episodes'] for row in branches) == 24576
