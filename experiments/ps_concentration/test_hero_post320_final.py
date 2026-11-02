from dataclasses import asdict, replace
from pathlib import Path

import pytest

from . import hero_post320_final
from .hero_v2 import identity, read, write
from .test_hero_v2 import config


@pytest.fixture
def post320_plan(tmp_path):
    parent = replace(config(4), output=str(tmp_path / 'extension/arms/w1024'), updates=320,
        learning_rate=1e-6, continuation_start_update=192, exact_kl_start_update=215)
    continued = replace(parent, output=str(tmp_path / 'continued'), gpus=8, updates=448,
        temperature=.8, continuation_start_update=320, source_sha256='b' * 64)
    write(Path(parent.output) / 'resolved_config.json', asdict(parent))
    write(Path(continued.output) / 'resolved_config.json', asdict(continued))
    plan = dict(format='hero-w1024-post320-v1', parent_policy_version=320, initial_target_update=448,
        parent_run=parent.output, output=continued.output, identity=identity(continued),
        fresh_future_test_reserved=dict(standard=[7000000, 7000511], larger=[8000000, 8000255]))
    path = Path(continued.output) / 'plan.json'
    write(path, plan)
    return path, parent, continued


def test_fixed_post320_matrix_uses_fresh_boards_and_unselected_checkpoints(post320_plan, tmp_path):
    path, parent, continued = post320_plan
    options = dict(gpus=8, source_root=tmp_path / 'captured-source', source_sha256='c' * 64)
    final = hero_post320_final.prepare(path, tmp_path / 'results', **options)
    assert len(final['conditions']) == 22
    assert sum(row['count'] for row in final['conditions']) == 5888
    assert [row['condition'] for row in final['conditions'][:2]] == final['primary']
    assert final['default_phase'] == 'ps'
    assert final['conditions'][3]['split'] == 'intervention'
    for entry in final['conditions']:
        inference = read(entry['argv'][entry['argv'].index('--config') + 1])
        assert inference['thinking_token_limit'] == 128 and inference['action_token_limit'] == 32
        assert inference['thinking_budget_transition'] == '\n\nOkay, time to give the answer.</think>\n\n'
        assert inference['evaluation_concurrency'] == (16 if entry['attention'] == 'full' else 64)
        expected = dict(continued=448, parent320=320, base=0)[entry['arm']]
        assert entry['policy_version'] == expected
        if expected:
            assert entry['policy_training_identity'] == identity(continued if expected == 448 else parent)
            assert entry['archive'].endswith(f'w1024-{expected:04d}')
        else:
            assert entry['checkpoint'] is None and entry['archive'] is None
        if entry['split'] == 'post320_test':
            assert entry['first_board_seed'] == 7000000
            assert entry['count'] == (128 if entry['condition'].endswith('first128') else 512)
        elif entry['split'] == 'post320_larger':
            assert entry['first_board_seed'] == 8000000 and entry['count'] == 256
            assert entry['attention'] == 'ps' and entry['window'] == 1024
        else:
            assert entry['arm'] == 'continued' and entry['first_board_seed'] == 4000000
    assert hero_post320_final.prepare(path, tmp_path / 'results', **options) == final
    assert not list(tmp_path.rglob('episodes.jsonl'))


@pytest.mark.parametrize('field,value,error', [
    ('initial_target_update', 447, 'fixed448'),
    ('parent_policy_version', 319, '320 parent'),
    ('fresh_future_test_reserved', dict(standard=[5000000, 5000511], larger=[8000000, 8000255]), 'fresh7M/8M'),
])
def test_post320_plan_rejects_changed_budget_or_inspected_seeds(post320_plan, tmp_path, field, value, error):
    path, _, _ = post320_plan
    plan = read(path)
    plan[field] = value
    write(path, plan)
    with pytest.raises(ValueError, match=error):
        hero_post320_final.prepare(path, tmp_path / 'results', gpus=8,
            source_root=tmp_path / 'source', source_sha256='c' * 64)
    assert not (tmp_path / 'results').exists()


def test_post320_tests_cannot_start_while_training_is_incomplete(post320_plan, tmp_path, monkeypatch):
    path, _, continued = post320_plan
    final = hero_post320_final.prepare(path, tmp_path / 'results', gpus=8,
        source_root=tmp_path / 'source', source_sha256='c' * 64)
    write(Path(continued.output) / 'state.json', dict(policy_version=355, identity=identity(continued)))
    calls = []
    monkeypatch.setattr(hero_post320_final, 'require_idle_devices', lambda *a: calls.append(a))
    monkeypatch.setattr(hero_post320_final, 'restore', lambda *a: calls.append(a))
    for function in (hero_post320_final.execute, hero_post320_final.memory):
        with pytest.raises(ValueError, match='complete training budget'):
            function(tmp_path / 'results/plan.json', gpus=8)
    assert calls == []


def test_post320_four_gpu_layout_preserves_all_evaluation_boards_and_controllers(post320_plan, tmp_path):
    path, _, _ = post320_plan
    results = [hero_post320_final.prepare(path, tmp_path / f'results{gpus}', gpus=gpus,
        source_root=tmp_path / 'source', source_sha256='c' * 64) for gpus in (4, 8)]
    for four, eight in zip(results[0]['conditions'], results[1]['conditions'], strict=True):
        for field in ('condition', 'arm', 'policy_version', 'attention', 'window', 'split',
                      'count', 'first_board_seed', 'last_board_seed', 'policy_training_identity'):
            assert four[field] == eight[field]
        left, right = [read(row['argv'][row['argv'].index('--config') + 1]) for row in (four, eight)]
        for field in ('gpus', 'output'):
            left.pop(field)
            right.pop(field)
        assert left == right


def test_post320_execution_defers_full_conditions_by_default(post320_plan, tmp_path, monkeypatch):
    path, _, _ = post320_plan
    final = hero_post320_final.prepare(path, tmp_path / 'results', gpus=8,
        source_root=tmp_path / 'source', source_sha256='c' * 64)
    calls = []
    monkeypatch.setattr(hero_post320_final, 'inputs', lambda *a: (final, {}))
    monkeypatch.setattr(hero_post320_final, 'require_idle_devices', lambda *a: None)
    monkeypatch.setattr(hero_post320_final, 'restore', lambda *a: None)
    monkeypatch.setattr(hero_post320_final, 'condition_result', lambda *a: None)
    monkeypatch.setattr(hero_post320_final, 'execute_conditions',
        lambda entries, output, **kwargs: calls.append((entries[0], kwargs)))
    hero_post320_final.execute(tmp_path / 'results/plan.json', gpus=8)
    assert len(calls) == 19 and all(entry['attention'] == 'ps' for entry, _ in calls)
    assert all(options == dict(gpus=8) for _, options in calls)
    calls.clear()
    hero_post320_final.execute(tmp_path / 'results/plan.json', gpus=4, phase='full')
    assert len(calls) == 3 and all(entry['attention'] == 'full' for entry, _ in calls)
    assert all(options == dict(gpus=4, evaluation_concurrency=16, full_kv_token_bound=53568)
        for _, options in calls)


def test_post320_parent_controller_cannot_silently_change(post320_plan, tmp_path):
    path, parent, _ = post320_plan
    write(Path(parent.output) / 'resolved_config.json', asdict(replace(parent, seed=22)))
    with pytest.raises(ValueError, match='parent model, task and generation controller'):
        hero_post320_final.prepare(path, tmp_path / 'results', gpus=8,
            source_root=tmp_path / 'source', source_sha256='c' * 64)
