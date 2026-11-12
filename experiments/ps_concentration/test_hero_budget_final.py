from dataclasses import asdict, replace
from pathlib import Path

import pytest

from . import hero_budget_final, hero_post320_final, hero_post320_report
from .hero_v2 import identity, read, write
from .test_hero_post320_final import post320_plan
from .test_hero_post320_report import committed


@pytest.fixture
def budget_stop(post320_plan):
    _, _, prior = post320_plan
    config = replace(prior, updates=2048)
    root = Path(config.output)
    write(root / 'resolved_config.json', asdict(config))
    committed(config, 1600)
    for update in range(1593, 1601):
        path = root / f'update-metrics/update-{update:04d}.json'
        row = read(path)
        row['metrics'].update(mean_attempts=27., completed_only_denominator=128,
            resource_terminations=0, mean_return=1-27/96, mean_native_return=1-27/96)
        write(path, row)
    criterion = dict(consecutive_complete_batches=8, episodes=1024, mean_responses_at_most=25.5,
        mean_pairs_at_least=7.99, completion_rate_at_least=.99,
        include_incomplete_games_and_all_error_attempts=True)
    write(root / 'goal25-plan-a1.json', dict(criterion=criterion))
    write(root / 'stop-after-update.json', dict(identity=identity(config), after_update=1600,
        requested_by_user=True))
    write(root / 'training-complete.json', dict(policy_version=1600, target=1600,
        configured_horizon=2048, reached_target=True))
    checkpoint = root / 'checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'x')
    write(checkpoint.with_suffix('.pt.json'), dict(policy_version=1600, identity=identity(config), bytes=1))
    return root


def freeze(root, output, gpus=8):
    return hero_budget_final.freeze(root, output, updates=1600, gpus=gpus,
        source_root=output.parent / 'captured-source', source_sha256='c' * 64)


def test_budget_stop_preserves_matrix_without_claiming_performance_target(budget_stop, tmp_path):
    four, eight = [freeze(budget_stop, tmp_path/f'final-{gpus}', gpus) for gpus in (4, 8)]
    for final in (four, eight):
        assert final['format'] == 'hero-user-budget-final-v1'
        assert final['fixed_final_update'] == 1600
        assert len(final['conditions']) == 22 and sum(c['count'] for c in final['conditions']) == 5888
        assert final['training_stop']['reason'] == 'explicit_user_budget'
        assert final['training_stop']['performance_target_met'] is False
        assert final['training_stop']['mean_responses'] == 27.
        assert not (budget_stop/'goal25-target-evidence.json').exists()
        for c in final['conditions']:
            if c['arm'] == 'continued':
                assert c['policy_version'] == 1600 and c['archive'].endswith('w1024-1600')
    for a, b in zip(four['conditions'], eight['conditions'], strict=True):
        for key in ['condition', 'policy_version', 'first_board_seed', 'last_board_seed', 'count', 'attention', 'window']:
            assert a[key] == b[key]


@pytest.mark.parametrize('change', ['not_explicit', 'wrong_stop', 'incomplete'])
def test_budget_freeze_rejects_unapproved_or_incomplete_stops(budget_stop, tmp_path, change):
    path = budget_stop / ('training-complete.json' if change == 'incomplete' else 'stop-after-update.json')
    field, value = dict(not_explicit=('requested_by_user', False), wrong_stop=('after_update', 1599),
        incomplete=('reached_target', False))[change]
    write(path, read(path) | {field:value})
    with pytest.raises(ValueError, match='training budget'):
        freeze(budget_stop, tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_budget_evidence_tampering_is_rejected(budget_stop, tmp_path):
    final = freeze(budget_stop, tmp_path/'final')
    plan = read(final['plan'])
    path = Path(plan['stopping_evidence'])
    write(path, read(path) | dict(performance_target_met=True))
    with pytest.raises(ValueError, match='frozen evidence'):
        hero_post320_final.configurations(plan)


def test_budget_completion_report_still_requires_reserved_tests(budget_stop, tmp_path):
    freeze(budget_stop, tmp_path/'final')
    output = tmp_path/'report'
    result = hero_post320_report.report(tmp_path/'final/plan.json', output, allow_partial=True)
    assert result['status'] == 'PARTIAL' and result['committed_update'] == 1600
    assert result['training_stop']['performance_target_met'] is False
    assert len(result['missing_conditions']) == 22 and result['missing_memory']
    text = (output/'README.md').read_text()
    assert '1600/1600' in text and 'Performance target met in the final stopping window: False' in text
    assert not list(tmp_path.rglob('episodes.jsonl'))
