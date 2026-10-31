from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from .hero_exports import file_hash
from .hero_finish import completed_condition, execute_conditions, preserve_candidates, ready, require_idle_devices
from .overlap_config import RunConfig, evaluation_gpu_count, evaluation_specs
from .rng_select import argument
from .test_hero_protocol import hero_selection_run, selection_run
from .test_rng_select import save


@pytest.fixture
def completed_training(hero_selection_run):
    plan_path, run = hero_selection_run
    plan = json.loads(plan_path.read_text())
    state = json.loads((run / 'state.json').read_text())
    save(run / 'training-complete.json', dict(policy_version=128, reached_target=True))
    (run / 'checkpoints').mkdir()
    (run / 'checkpoints/latest.pt').write_bytes(b'CPU orchestration fixture')
    save(run / 'checkpoints/latest.pt.json', dict(policy_version=128,
        identity=state['identity'], bytes=(run / 'checkpoints/latest.pt').stat().st_size))
    return plan, run


def test_training_barrier_rejects_unfinished_budget_before_evaluation(completed_training):
    plan, run = completed_training
    assert ready(run, plan)['policy_version'] == 128
    save(run / 'state.json', json.loads((run / 'state.json').read_text()) | dict(policy_version=127))
    with pytest.raises(ValueError, match='complete training budget'):
        ready(run, plan)


def test_completed_state_cannot_skip_worker_shutdown(completed_training, monkeypatch):
    plan, run = completed_training
    config = run / 'active-config.json'
    save(config, dict(output=str(run)))
    save(run / 'processes-fixture.json', dict(coordinator=os.getpid(), children=[]))
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path(f'/proc/{os.getpid()}/cmdline'):
            return '\0'.join(['python', '-m', 'experiments.ps_concentration.hero_v2', '--config', str(config)])
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(ValueError, match='workers are still live'):
        ready(run, plan)
    save(config, dict(output=str(run / 'another-campaign')))
    assert ready(run, plan)['policy_version'] == 128


def test_idle_check_only_considers_assigned_gpus(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3,4,5')
    processes = ['GPU-0, 123']

    def query(argv, **kwargs):
        if '--query-gpu=index,uuid' in argv:
            return '\n'.join(f'{i}, GPU-{i}' for i in range(8))
        return '\n'.join(processes)

    monkeypatch.setattr('subprocess.check_output', query)
    require_idle_devices(4)
    processes.append('GPU-2, 456')
    with pytest.raises(ValueError, match='still have compute processes'):
        require_idle_devices(4)


def test_physical_evaluation_layout_keeps_controller_and_board_plan(hero_selection_run):
    plan, _ = hero_selection_run
    entry = json.loads(plan.read_text())['layouts']['4'][0]
    config = RunConfig(**json.loads(Path(argument(entry['argv'], '--config')).read_text()))
    eight = replace(config, gpus=8)
    four = evaluation_gpu_count(eight, 4, mode='evaluate')
    assert four.scientific_identity() == eight.scientific_identity()
    left = evaluation_specs(eight, 'test', 'fixture', count=128)
    right = evaluation_specs(four, 'test', 'fixture', count=128)
    assert [{k:v for k,v in row.items() if k != 'rank'} for row in left] == [
        {k:v for k,v in row.items() if k != 'rank'} for row in right]
    with pytest.raises(ValueError, match='evaluation-only'):
        evaluation_gpu_count(eight, 4, mode='train')


def test_complete_conditions_are_verified_and_skipped_on_resume(hero_selection_run, tmp_path):
    plan, _ = hero_selection_run
    entry = json.loads(plan.read_text())['layouts']['4'][-1]
    calls = []
    execute_conditions([entry], tmp_path / 'executor', gpus=4, invoke=calls.append)
    assert calls == [] and completed_condition(entry)


def test_partial_condition_resumes_with_actual_gpu_count(hero_selection_run, tmp_path):
    plan, _ = hero_selection_run
    entry = json.loads(plan.read_text())['layouts']['4'][-1]
    summary = Path(entry['summary'])
    original = summary.read_text()
    summary.unlink()
    calls = []

    def execute(argv):
        calls.append(argv)
        summary.write_text(original)

    execute_conditions([entry], tmp_path / 'executor', gpus=8, invoke=execute)
    assert calls[0][-2:] == ['--gpus', '8']
    assert completed_condition(entry)


def test_window_sweep_completion_uses_the_actual_evaluation_window(hero_selection_run):
    plan, _ = hero_selection_run
    entry = json.loads(plan.read_text())['layouts']['4'][-1]
    entry['argv'].extend(['--window', '256'])
    folder = Path(entry['summary']).parent
    summary = json.loads((folder / 'summary.json').read_text()) | dict(window=256)
    save(folder / 'summary.json', summary)
    protocol = json.loads((folder / 'evaluation-protocol.json').read_text())
    protocol['config']['window'] = 256
    save(folder / 'evaluation-protocol.json', protocol)
    rows = [json.loads(line) | dict(window=256) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
    (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert completed_condition(entry)


def test_preservation_copies_only_verified_exports_and_rechecks_existing_files(completed_training, tmp_path):
    plan, run = completed_training
    plan['exports']['directory'] = str(tmp_path / 'preserved')
    for version in plan['exports']['versions']:
        source = tmp_path / f'node-policy-{version}'
        source.mkdir()
        (source / 'model.safetensors').write_bytes(b'CPU copy fixture')
        save(source / 'experiment.json', dict(plan['training_identity'], policy_version=version))
        files = [dict(name=p.name, bytes=p.stat().st_size, sha256=file_hash(p)) for p in source.iterdir()]
        save(run / f'export-catalog/policy-{version:04d}.json', dict(status='VERIFIED',
            policy_version=version, identity=plan['training_identity'], unique_parameters=4022468096,
            policy=str(source), files=files))
    receipts = preserve_candidates(run, plan)
    assert [r['policy_version'] for r in receipts] == plan['exports']['versions']
    assert preserve_candidates(run, plan) == receipts
    (Path(receipts[0]['policy']) / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Previously preserved policy changed'):
        preserve_candidates(run, plan)


def test_corrupt_source_never_publishes_a_candidate(completed_training, tmp_path):
    plan, run = completed_training
    plan['exports']['versions'] = [16]
    plan['exports']['directory'] = str(tmp_path / 'preserved')
    source = tmp_path / 'node'
    source.mkdir()
    weights = source / 'model.safetensors'
    weights.write_bytes(b'original')
    files = [dict(name=weights.name, bytes=weights.stat().st_size, sha256=file_hash(weights))]
    save(run / 'export-catalog/policy-0016.json', dict(status='VERIFIED', policy_version=16,
        identity=plan['training_identity'], unique_parameters=4022468096, policy=str(source), files=files))
    weights.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Copied inference export differs'):
        preserve_candidates(run, plan)
    assert not (Path(plan['exports']['directory']) / 'policy-0016').exists()
