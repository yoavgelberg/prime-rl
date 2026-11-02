import csv

import pytest

from .hero_report import collect, report
from .hero_v2 import read, write


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / 'run'

    def prepare(parent, last):
        write(run / 'resolved_config.json', dict(boards=16, group_size=8, window=1024, seed=11))
        write(run / 'state.json', dict(policy_version=last))
        if parent:
            write(run / 'fork-receipt.json', dict(parent_policy_version=parent))
        for update in range(parent + 1, last + 1):
            metrics = dict(update=update, episodes=128, training_episodes=update * 128,
                mean_return=.4, mean_native_return=.5, mean_pairs=8., mean_attempts=48.,
                completed_only_denominator=128, format_errors=0, native_invalid_actions=0,
                generated_tokens=64000, raw_transcript_tokens=128000, seconds=200.,
                mean_per_game_tokens_per_second=50., generated_tokens_per_second=320.,
                peak_training_allocated_bytes=16 * 1024**3, kl_per_generated_token=.1,
                entropy_per_generated_token=.3, gradient_norm=.1, learning_rate=1e-6)
            write(run / f'update-metrics/update-{update:04d}.json',
                dict(metrics=metrics, steps=[dict(peak_allocated_bytes=16 * 1024**3)]))
        return run

    return prepare


def test_report_preserves_base_run_episode_counts(saved_run):
    rows, _, _, _ = collect(saved_run(0, 2))
    assert [row['new_training_episodes'] for row in rows] == [128, 256]
    assert all(row['new_training_episodes'] == row['training_episodes'] for row in rows)


def test_continuation_report_counts_only_new_episodes(saved_run, tmp_path):
    run = saved_run(320, 322)
    output = tmp_path / 'report'
    result = report(run, output)
    assert result['updates'] == 2
    rows = list(csv.DictReader((output / 'training-progress.csv').open()))
    assert [int(row['new_training_episodes']) for row in rows] == [128, 256]
    assert [int(row['training_episodes']) for row in rows] == [41088, 41216]
    assert read(output / 'provenance.json')['continuation_parent_update'] == 320
    assert read(output / 'provenance.json')['new_training_episodes'] == 256
    assert '256 episodes in this run; 41,216 in the policy lineage' in (output / 'README.md').read_text()
    for name in ('return-pairs-responses.png', 'return-pairs-responses.pdf', 'systems.png', 'systems.pdf'):
        assert (output / name).stat().st_size > 1000


def test_continuation_report_rejects_a_missing_update(saved_run):
    run = saved_run(320, 322)
    (run / 'update-metrics/update-0321.json').unlink()
    with pytest.raises(ValueError, match='Every committed update'):
        collect(run)


def test_continuation_report_excludes_uncommitted_metrics(saved_run):
    run = saved_run(320, 323)
    write(run / 'state.json', dict(policy_version=322))
    rows, _, _, _ = collect(run)
    assert [row['update'] for row in rows] == [321, 322]
