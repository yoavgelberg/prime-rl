from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from .hero_protocol import inference_config
from .hero_v2 import HeroConfig, identity
from .hero_validation_plan import prepare
from .rng_final_plan import prepare_final_plan
from .rng_report import training_curve_rows
from .rng_select import argument, select_checkpoint
from .test_rng_select import save, selection_run


@pytest.fixture
def hero_selection_run(selection_run):
    plan, run = selection_run
    original = json.loads(plan.read_text())
    first = original['layouts']['4'][0]
    config = json.loads(Path(argument(first['argv'], '--config')).read_text())
    hero = HeroConfig(model=config['model'], source=config['source'], output=str(run),
        node_dir=str(run / 'node'), source_sha256='b' * 64, deadline=2000000000.)
    expected = identity(hero)
    save(run / 'resolved_config.json', asdict(hero))
    save(run / 'state.json', dict(identity=expected, policy_version=128, training_episodes=128 * 128))
    original['training_identity'] = expected
    for entry in original['layouts']['4']:
        current = inference_config(hero, output=Path(entry['summary']).parents[2], gpus=4).model_dump()
        config_path = Path(argument(entry['argv'], '--config'))
        save(config_path, current)
        entry['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        folder = Path(entry['summary']).parent
        save(folder / 'evaluation-protocol.json', dict(config=current))
        path = folder / 'episodes.jsonl'
        rows = [json.loads(line) | dict(action_token_limit=32) for line in path.read_text().splitlines()]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        if entry['policy_version']:
            save(Path(argument(entry['argv'], '--checkpoint')) / 'experiment.json',
                dict(expected, policy_version=entry['policy_version']))
    save(plan, original)
    return plan, run


def test_hero_selection_preserves_controller_and_earlier_tie_break(hero_selection_run):
    plan, run = hero_selection_run
    result = select_checkpoint(plan, run, gpus=4)
    assert result['update'] == 16 and result['training_seed'] == 11
    assert result['competence_and_forgetting_status'] == 'GO'
    assert result['validation_return'] == pytest.approx(2 / 3)


def test_hero_selection_requires_prespecified_training_identity(hero_selection_run):
    plan, run = hero_selection_run
    value = json.loads(plan.read_text())
    value['training_identity']['entropy_coefficient'] = .002
    save(plan, value)
    with pytest.raises(ValueError, match='complete hero training identity'):
        select_checkpoint(plan, run, gpus=4)


def test_hero_selection_rejects_changed_answer_budget_before_reading_results(hero_selection_run):
    plan, run = hero_selection_run
    path = run / 'resolved_config.json'
    value = json.loads(path.read_text()) | dict(action_token_limit=64)
    save(path, value)
    with pytest.raises(ValueError, match='committed state and resolved training identity'):
        select_checkpoint(plan, run, gpus=4)


def test_hero_curves_count_128_episodes_and_validate_deferred_records(hero_selection_run):
    plan, run = hero_selection_run
    entries = json.loads(plan.read_text())['layouts']['4']
    summaries = [x['summary'] for x in entries if x['attention'] == 'ps']
    save(run / 'update-metrics/update-0016.json', dict(metrics=dict(
        update=16, episodes=128, training_episodes=2048, mean_return=.1, mean_native_return=.2)))
    rows = training_curve_rows([dict(run_dir=str(run), training_seed=11, validation_summaries=summaries)])
    assert {r['training_episodes'] for r in rows if r['kind'] == 'sampled_training'} == {2048}
    assert [r['mean_return'] for r in rows if r['kind'] == 'sampled_training'] == [.2]
    assert [r['training_episodes'] for r in rows if r['kind'] == 'greedy_validation'] == [0, 2048, 16384]
    assert all(r['training_seed'] == 11 for r in rows)


def test_hero_curves_reject_legacy_batch_accounting(hero_selection_run):
    _, run = hero_selection_run
    save(run / 'update-metrics/update-0016.json', dict(metrics=dict(
        update=16, episodes=64, training_episodes=1024, mean_return=.1)))
    with pytest.raises(ValueError, match='complete 128-trajectory'):
        training_curve_rows([dict(run_dir=str(run), training_seed=11)])


def test_deferred_hero_plan_pins_all_candidates_and_identical_controllers(hero_selection_run, tmp_path):
    _, run = hero_selection_run
    output = tmp_path / 'hero-plan'
    plan = prepare(run, output, tmp_path / 'exports', source_root=tmp_path / 'source', source_sha256='c' * 64)
    assert plan['status'] == 'PREPARED_NOT_EXECUTED' and plan['training_barrier'] == 128
    assert plan['training_batch_size'] == 128
    assert plan['exports']['versions'] == list(range(16, 129, 16))
    for gpus, entries in plan['layouts'].items():
        assert len(entries) == 10
        for entry in entries:
            config = json.loads(Path(argument(entry['argv'], '--config')).read_text())
            assert config['thinking_token_limit'] == 128 and config['action_token_limit'] == 32
            assert config['gpus'] == int(gpus) and config['window'] == 1024
            assert argument(entry['argv'], '--split') == 'validation'
            assert argument(entry['argv'], '--count') == '128'
    assert plan == prepare(run, output, tmp_path / 'exports',
        source_root=tmp_path / 'source', source_sha256='c' * 64)
    assert not list(output.rglob('episodes.jsonl'))


@pytest.mark.parametrize('gpus', [4, 8])
def test_hero_final_plan_uses_selected_checkpoint_and_current_controller(hero_selection_run, tmp_path, gpus):
    plan_path, run = hero_selection_run
    plan = json.loads(plan_path.read_text())
    plan.update(source_root=str(tmp_path / 'source'), source_sha256='c' * 64)
    next(entry for entry in plan['layouts']['4'] if entry['attention'] == 'ps'
        and entry['policy_version'] == 0)['condition'] = 'base-ps'
    save(plan_path, plan)
    selection = tmp_path / 'selected.json'
    save(selection, select_checkpoint(plan_path, run, gpus=4))
    output = tmp_path / 'final'
    final = prepare_final_plan(plan_path, run, selection, output, gpus=gpus, validation_gpus=4)
    assert final['selected']['update'] == 16 and len(final['commands']) == 12
    for entry in final['commands']:
        config = json.loads(Path(argument(entry['argv'], '--config')).read_text())
        assert config['thinking_token_limit'] == 128 and config['action_token_limit'] == 32
        assert config['thinking_budget_transition'].endswith('</think>\n\n')
    manifest = json.loads((output / 'report-manifest.json').read_text())
    curve = training_curve_rows(manifest['training_runs'])
    assert [r['training_episodes'] for r in curve] == [0, 2048, 16384]
