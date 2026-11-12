import pytest

from .allocation_accounting import account, gpu_resources


def row(job, state, start, end, seconds, tres='', restart=0):
    return '|'.join(map(str, [job, 'experiment', state, start, end, seconds, tres, '0:0', restart, 'yejin', 'node7'])) + '\n'


def test_requeues_and_cleanup_are_charged_once_while_pending_costs_zero():
    tres = 'cpu=32,gres/gpu:b200=4,gres/gpu=4,mem=1664G'
    text = row(42, 'REQUEUED', '2026-10-29T02:00:00', '2026-10-29T02:10:00', 600, tres)
    text += row('42.batch', 'CANCELLED', '2026-10-29T02:00:01', '2026-10-29T02:10:00', 599)
    text += row('42.extern', 'COMPLETED', '2026-10-29T02:00:00', '2026-10-29T02:10:03', 603)
    text += row(42, 'REQUEUED', '2026-10-29T02:20:00', '2026-10-29T02:25:00', 300, tres, restart=1)
    text += row('42.extern', 'COMPLETED', '2026-10-29T02:20:00', '2026-10-29T02:25:01', 301)
    text += row(42, 'CANCELLED', 'None', '2026-10-29T02:30:00', 0, restart=2)
    text += row(43, 'PENDING', 'Unknown', 'Unknown', 0)
    result = account(text, observed_utc='2026-10-29T03:00:00+00:00', reservations={'43':dict(gpus=8, max_hours=24)})
    assert len(result['allocation_records']) == 2
    assert result['total_allocated_gpu_hours'] == pytest.approx((603 + 301) * 4 / 3600)
    assert result['gpu_hours_by_type']['b200'] == result['total_allocated_gpu_hours']
    assert result['pending_maximum_gpu_hours'] == 192
    assert result['complete_accounting_for_requested_records']


def test_generic_and_typed_gpu_resources_do_not_double_gpu_count():
    assert gpu_resources('gres/gpu:b200=8,gres/gpu=8,cpu=64') == (8, {'b200':8})
    assert gpu_resources('gres/gpu:h200=1') == (1, {'h200':1})
    assert gpu_resources('cpu=64,mem=1T') == (0, {})
    with pytest.raises(ValueError, match='counts disagree'):
        gpu_resources('gres/gpu:b200=4,gres/gpu=8')


def test_duplicate_parent_observations_and_steps_do_not_double_charge():
    text = row(42, 'COMPLETED', '2026-10-29T02:00:00', '2026-10-29T03:00:00', 3600, 'gres/gpu=4')
    result = account(text + text, observed_utc='2026-10-29T04:00:00+00:00')
    assert result['total_allocated_gpu_hours'] == 4
    assert len(result['allocation_records']) == 1


def test_missing_requeue_history_is_reported_incomplete():
    text = row(42, 'COMPLETED', '2026-10-29T02:00:00', '2026-10-29T03:00:00', 3600, 'gres/gpu=4', restart=2)
    result = account(text, observed_utc='2026-10-29T04:00:00+00:00')
    assert not result['complete_accounting_for_requested_records']
    assert 'restart indices [0, 1]' in result['issues'][0]


def test_running_cost_uses_observed_elapsed_not_an_assumed_end_time():
    text = row(42, 'RUNNING', '2026-10-29T02:00:00', 'Unknown', 120, 'gres/gpu:h200=1,gres/gpu=1')
    result = account(text, observed_utc='2026-10-29T04:00:00+00:00')
    assert result['total_allocated_gpu_hours'] == pytest.approx(120 / 3600)
