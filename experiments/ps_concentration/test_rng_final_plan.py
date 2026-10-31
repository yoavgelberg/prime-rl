import json
from pathlib import Path

import pytest

from .rng_final_plan import prepare_final_plan
from .rng_select import argument, select_checkpoint
from .test_rng_select import save, selection_run


def inputs(selection_run, tmp_path):
    plan, run = selection_run
    value = json.loads(plan.read_text())
    value.update(source_root=str(tmp_path / 'frozen-runtime'), source_sha256='a' * 64)
    for entry in value['layouts']['4']:
        if entry['attention'] == 'ps' and entry['policy_version'] == 0:
            entry['condition'] = 'base-ps'
    save(plan, value)
    selection = tmp_path / 'selected.json'
    save(selection, select_checkpoint(plan, run, gpus=4))
    return plan, run, selection, tmp_path / 'final'


@pytest.mark.parametrize('gpus', [4, 8])
def test_final_plan_preserves_controller_reserved_splits_and_selection(selection_run, tmp_path, gpus):
    plan, run, selection, output = inputs(selection_run, tmp_path)
    result = prepare_final_plan(plan, run, selection, output, gpus=gpus, validation_gpus=4)
    assert result['selected']['update'] == 16
    assert result['status'] == 'PREPARED_NOT_EXECUTED'
    assert len(result['commands']) == 12
    for entry in result['commands']:
        config = json.loads(Path(argument(entry['argv'], '--config')).read_text())
        assert config['thinking_token_limit'] == 128 and config['action_token_limit'] is None
        assert config['window'] == 1024 and config['gpus'] == gpus
        assert config['evaluation_concurrency'] == (4 if entry['attention'] == 'full' else 128 // gpus)
        assert entry['split'] != 'validation'
        assert ('--checkpoint' in entry['argv']) == (entry['policy'] == 'PS RL')
        if entry['split'] == 'test':
            assert entry['first_board_seed'] == 2000000
            assert entry['count'] == (128 if '--window' in entry['argv'] else 512)
        elif entry['split'] == 'larger':
            assert entry['first_board_seed'] == 3000000 and entry['count'] == 256
            assert entry['board']['max_responses'] == 192 and entry['board']['total_pairs'] == 16
        else:
            assert entry['split'] == 'intervention' and entry['first_board_seed'] == 4000000
    report = json.loads((output / 'report-manifest.json').read_text())
    assert len(report['conditions']) == 11
    assert report['training_runs'][0]['validation_summaries'] == [
        str(tmp_path / name / 'evaluation/validation/summary.json') for name in ('ps-0', 'ps-16', 'ps-128')]
    assert report['comparisons'][0]['sliding_control'] == 'base-ps-test'
    assert result == prepare_final_plan(plan, run, selection, output, gpus=gpus, validation_gpus=4)
    assert not (output / 'selected-ps-test' / 'evaluation').exists()


def test_final_plan_never_labels_selected_starting_policy_as_trained(selection_run, tmp_path):
    plan, run, selection, output = inputs(selection_run, tmp_path)
    value = json.loads(plan.read_text())
    base = next(entry for entry in value['layouts']['4'] if entry['condition'] == 'base-ps')
    folder = Path(base['summary']).parent
    rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
    for row in rows:
        row.update(score=8, done=True, response_count=16, training_return=1 - 16 / 96)
    (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    from .overlap_config import episode_metrics
    save(folder / 'summary.json', json.loads((folder / 'summary.json').read_text()) | episode_metrics(rows))
    save(selection, select_checkpoint(plan, run, gpus=4))
    result = prepare_final_plan(plan, run, selection, output, gpus=8, validation_gpus=4)
    assert result['selected']['update'] == 0
    assert result['selected_test'] == dict(full='base-full-test', ps='base-ps-test')
    assert all('--checkpoint' not in entry['argv'] and entry['policy'] == 'base' for entry in result['commands'])
    assert result['intervention_status'] == 'NO_TRAINED_CHECKPOINT_SELECTED'
    assert not any(entry['split'] == 'intervention' for entry in result['commands'])
    report = json.loads((output / 'report-manifest.json').read_text())
    assert all(entry['training_seed'] is None for entry in report['conditions'])


def test_final_plan_rejects_changed_selection_and_incomplete_training_before_writing(selection_run, tmp_path):
    plan, run, selection, output = inputs(selection_run, tmp_path)
    original = selection.read_text()
    save(selection, json.loads(original) | dict(update=128))
    with pytest.raises(ValueError, match='Selection receipt differs'):
        prepare_final_plan(plan, run, selection, output, gpus=4, validation_gpus=4)
    assert not output.exists()
    selection.write_text(original)
    save(run / 'state.json', json.loads((run / 'state.json').read_text()) | dict(policy_version=127))
    with pytest.raises(ValueError, match='training barrier'):
        prepare_final_plan(plan, run, selection, output, gpus=4, validation_gpus=4)
    assert not output.exists()
