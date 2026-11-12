from dataclasses import asdict, replace
from pathlib import Path

import pytest

from . import hero_goal25_final, hero_post320_final, hero_post320_report
from .hero_exports import file_hash
from .hero_v2 import identity, read, write
from .test_hero_post320_final import post320_plan
from .test_hero_post320_report import committed


@pytest.fixture
def goal_stop(post320_plan):
    initial, parent, config = post320_plan
    config = replace(config, updates=1024)
    root = Path(config.output)
    write(root / 'resolved_config.json', asdict(config))
    committed(config, 501)
    criteria = dict(consecutive_complete_batches=8, mean_responses_at_most=25.5,
        mean_pairs_at_least=7.99, completion_rate_at_least=.99)
    sources = {}
    for update in range(493, 501):
        path = root / f'update-metrics/update-{update:04d}.json'
        value = read(path)
        value['metrics'].update(mean_attempts=25., mean_pairs=8., completed_only_denominator=128,
            resource_terminations=0)
        write(path, value)
        sources[str(path)] = file_hash(path)
    write(root / 'goal25-plan-a1.json', dict(criterion=criteria))
    write(root / 'goal25-target-evidence.json', dict(status='TARGET_MET', first_update=493,
        last_update=500, criterion=criteria, mean_responses=25., mean_pairs=8.,
        completion_rate=1., input_sha256=sources))
    write(root / 'stop-after-update.json', dict(identity=identity(config), after_update=501))
    write(root / 'training-complete.json', dict(policy_version=501, reached_target=True))
    path = root / 'checkpoints/latest.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'x')
    write(path.with_suffix('.pt.json'), dict(policy_version=501, identity=identity(config), bytes=1))
    return root, config


def test_goal_stop_freezes_actual_checkpoint_and_keeps_boards_equal_on_four_or_eight_gpus(goal_stop, tmp_path):
    root, config = goal_stop
    results = [hero_goal25_final.freeze(root, tmp_path / f'final{gpus}', gpus=gpus,
        source_root=tmp_path / 'captured', source_sha256='c'*64) for gpus in (4, 8)]
    for final in results:
        assert final['format'] == 'hero-goal25-final-v1' and final['fixed_final_update'] == 501
        assert len(final['conditions']) == 22 and sum(c['count'] for c in final['conditions']) == 5888
        assert 'no test-based' in final['checkpoint_rule']
        assert all(c['policy_version'] == 501 and c['archive'].endswith('w1024-0501')
            for c in final['conditions'] if c['arm'] == 'continued')
        frozen = hero_post320_final.configurations(read(final['plan']))
        assert frozen['continued'].updates == 1024 and frozen['continued'] == config
    for left, right in zip(results[0]['conditions'], results[1]['conditions'], strict=True):
        for key in ['policy_version', 'first_board_seed', 'last_board_seed', 'count', 'window', 'attention']:
            assert left[key] == right[key]
    result = hero_post320_report.report(tmp_path / 'final4/plan.json', tmp_path / 'report', allow_partial=True)
    assert result['committed_update'] == 501 and result['status'] == 'PARTIAL'
    assert '501/501' in (tmp_path / 'report/README.md').read_text()


@pytest.mark.parametrize('change', ['incomplete', 'wrong_stop', 'changed_batch', 'false_evidence'])
def test_goal_stop_rejects_incomplete_or_unverified_selection_before_writing_a_plan(goal_stop, tmp_path, change):
    root, _ = goal_stop
    if change == 'incomplete':
        path = root / 'training-complete.json'
        write(path, read(path) | dict(reached_target=False))
    elif change == 'wrong_stop':
        path = root / 'stop-after-update.json'
        write(path, read(path) | dict(after_update=502))
    elif change == 'changed_batch':
        path = root / 'update-metrics/update-0493.json'
        value = read(path)
        value['metrics']['mean_attempts'] = 80.
        write(path, value)
    else:
        path = root / 'goal25-target-evidence.json'
        write(path, read(path) | dict(mean_responses=20.))
    with pytest.raises(ValueError):
        hero_goal25_final.freeze(root, tmp_path / 'final', gpus=4,
            source_root=tmp_path / 'captured', source_sha256='c'*64)
    assert not (tmp_path / 'final').exists()
