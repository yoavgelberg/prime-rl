import hashlib
import json
from pathlib import Path

import pytest

from .overlap_config import RunConfig, episode_metrics
from .rng_select import select_checkpoint


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selection_run(tmp_path):
    revision = '768f209d9ea81521153ed38c47d515654e938aea'
    run = tmp_path / 'training'
    config = RunConfig(model=str(tmp_path / revision), source=str(tmp_path / 'native'),
        output=str(run), source_sha256='a' * 64, updates=128, gpus=4,
        thinking_token_limit=128, thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n',
        kl_coefficient=0.)
    save(run / 'state.json', dict(config.scientific_identity(), policy_version=128))
    conditions = []
    for attention, version in [('full', 0), ('ps', 0), ('ps', 16), ('ps', 128)]:
        name = f'{attention}-{version}'
        folder = tmp_path / name
        current = config.model_dump() | dict(output=str(folder))
        config_path = folder / 'config.json'
        save(config_path, current)
        checkpoint = tmp_path / 'policies' / str(version)
        identity = str(checkpoint.resolve()) if version else revision
        if version:
            save(checkpoint / 'experiment.json', dict(config.scientific_identity(), policy_version=version))
        score, attempts = (4, 96) if (attention, version) == ('ps', 0) else (8, 26 if attention == 'full' else 32)
        rows = [dict(board_seed=1100000 + index, episode_id=f'{name}-{index}',
            policy_version=version, policy_identity=identity, greedy=True, attention=attention,
            window=1024, score=score, total_pairs=8, response_count=attempts, max_responses=96,
            done=score == 8, censored=False, training_return=score / 8 - attempts / 96,
            invalid_actions=0, format_errors=0, generated_tokens=500, raw_transcript_tokens=1000,
            seconds=1., thinking_token_limit=128, action_token_limit=None,
            thinking_budget_transition=config.thinking_budget_transition) for index in range(128)]
        output = folder / 'evaluation/validation'
        output.mkdir(parents=True)
        (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        save(output / 'summary.json', dict(episode_metrics(rows), split='validation', greedy=True,
            attention=attention, window=1024, policy_version=version, policy_identity=identity))
        save(output / 'evaluation-protocol.json', dict(config=current))
        argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
            '--config', str(config_path), '--attention', attention, '--split', 'validation']
        if version:
            argv.extend(['--checkpoint', str(checkpoint)])
        conditions.append(dict(condition=name, attention=attention, policy_version=version, count=128,
            argv=argv, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            summary=str(output / 'summary.json')))
    plan = tmp_path / 'plan.json'
    save(plan, dict(training_barrier=128, exports=dict(versions=[16, 128]), layouts={'4': conditions}))
    return plan, run


def test_deferred_selection_uses_all_validation_candidates_and_earlier_ties(selection_run):
    plan, run = selection_run
    result = select_checkpoint(plan, run, gpus=4)
    assert result['update'] == 16
    assert result['validation_return'] == pytest.approx(2 / 3)
    assert [row['update'] for row in result['candidates']] == [0, 16, 128]
    assert result['competence_and_forgetting_status'] == 'GO'
    assert result['full_minus_base_ps_matched_fraction'] == .5
    assert result['split'] == 'validation' and result['episodes'] == 128
    assert not (run / 'selected-checkpoint.json').exists()


def test_deferred_selection_cannot_run_during_training(selection_run):
    plan, run = selection_run
    path = run / 'state.json'
    save(path, json.loads(path.read_text()) | dict(policy_version=127))
    with pytest.raises(ValueError, match='training barrier'):
        select_checkpoint(plan, run, gpus=4)


def test_deferred_selection_rejects_final_test_even_if_its_return_is_better(selection_run):
    plan, run = selection_run
    value = json.loads(plan.read_text())
    argv = value['layouts']['4'][-1]['argv']
    argv[argv.index('--split') + 1] = 'test'
    save(plan, value)
    with pytest.raises(ValueError, match='only the fixed 128-board validation'):
        select_checkpoint(plan, run, gpus=4)


def test_deferred_selection_requires_complete_unselected_candidates_too(selection_run):
    plan, run = selection_run
    value = json.loads(plan.read_text())
    path = Path(value['layouts']['4'][-1]['summary']).parent / 'episodes.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    with pytest.raises(ValueError, match='expected 128'):
        select_checkpoint(plan, run, gpus=4)


@pytest.mark.parametrize('target,updates,message', [
    ('metadata', dict(action_token_limit=64), 'controller identity'),
    ('protocol', dict(action_token_limit=64), 'Saved evaluation protocol'),
    ('summary', dict(mean_return=999), 'disagrees with its episode records'),
    ('episodes', dict(policy_identity='another-policy'), 'another checkpoint'),
    ('episodes', dict(thinking_token_limit=64), 'generation controller'),
])
def test_deferred_selection_rejects_mixed_protocols_or_mislabelled_results(selection_run, target, updates, message):
    plan, run = selection_run
    entry = json.loads(plan.read_text())['layouts']['4'][-1]
    summary = Path(entry['summary'])
    if target == 'episodes':
        path = summary.parent / 'episodes.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].update(updates)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        path = summary if target == 'summary' else (summary.parent / 'evaluation-protocol.json' if target == 'protocol'
            else Path(entry['argv'][entry['argv'].index('--checkpoint') + 1]) / 'experiment.json')
        value = json.loads(path.read_text())
        (value['config'] if target == 'protocol' else value).update(updates)
        save(path, value)
    with pytest.raises(ValueError, match=message):
        select_checkpoint(plan, run, gpus=4)
