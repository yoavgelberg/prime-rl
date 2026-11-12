import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import hero_memory as module
from .rng_intervene import summarize_results


def test_intervention_coordinator_import_never_probes_or_initializes_cuda():
    program = '''
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('Intervention coordinator probed CUDA before its idle-device check')
torch.cuda.is_available = lambda: True
torch.cuda.is_bf16_supported = forbidden
torch.cuda.current_device = forbidden
import experiments.ps_concentration.hero_memory
assert not torch.cuda.is_initialized()
'''
    subprocess.run([sys.executable, '-c', program], check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def save_states(path, states):
    path.write_bytes(gzip.compress((''.join(json.dumps(state) + '\n' for state in states)).encode(), mtime=0))


def state(index, *, testable=True):
    item = dict(state_id=f's{index}', board_seed=4000000 + index,
        state_boundary='before_thinking', partner='aA', streaming_action='aA', correct_partners=['aA'])
    item.update(dict(edits=[], sham_edits=[]) if testable else dict(untestable_reason='No equal sham'))
    return item


def result(item, checkpoint):
    row = dict(item, checkpoint=str(checkpoint.resolve()), samples=32, state_sha256=module.state_hash(item))
    if 'edits' in item:
        row['streaming_parity'] = dict(absolute_error=.001)
        for branch in ('unedited', 'swap', 'sham'):
            false = branch == 'swap'
            row[branch] = dict(greedy=dict(action='bB' if false else 'aA', correct=not false,
                follows_false=false), probability_correct_partner=float(not false),
                probability_following_false=float(false))
    return row


def test_only_prethinking_separate_split_states_can_run(tmp_path):
    path = tmp_path / 'states.jsonl.gz'
    for bad in [state(0) | dict(board_seed=2000000), state(0) | dict(state_boundary='first_coordinate')]:
        save_states(path, [bad])
        with pytest.raises(ValueError, match='Primary interventions'):
            module.read_states(path)
    for bad in [[state(0), state(0)], [state(i) for i in range(129)]]:
        save_states(path, bad)
        with pytest.raises(ValueError, match='at most 128 distinct'):
            module.read_states(path)


def test_four_and_eight_gpu_shards_keep_every_state_once_in_fixed_order():
    states = [state(i) for i in range(128)]
    for count in (4, 8):
        shards = module.state_shards(states, count)
        assert sorted(row['state_id'] for subset in shards for row in subset) == sorted(row['state_id'] for row in states)
        assert all(subset == states[rank::count] for rank, subset in enumerate(shards))


def test_aggregation_counts_untestable_states_and_requires_every_testable_branch(tmp_path):
    checkpoint = tmp_path / 'policy'
    states = [state(0), state(1, testable=False), state(2)]
    for item in states:
        (tmp_path / (item['state_id'] + '.json')).write_text(json.dumps(result(item, checkpoint)))
    summary = module.aggregate(states, tmp_path, checkpoint=checkpoint)
    assert (summary['qualified_states'], summary['executed_states'], summary['untestable_states'], summary['pending_states']) == (3, 2, 1, 0)
    assert summary['branches']['denominator'] == 2
    assert not summary['causal_textual_memory_established']
    path = tmp_path / 's0.json'
    wrong = result(states[0], checkpoint)
    wrong.pop('sham')
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='all three branches'):
        module.aggregate(states, tmp_path, checkpoint=checkpoint)


@pytest.mark.parametrize('key,value', [('checkpoint', 'another'), ('samples', 16), ('state_sha256', 'changed')])
def test_aggregation_rejects_mixed_state_policy_and_sample_count(tmp_path, key, value):
    item, checkpoint = state(0), tmp_path / 'policy'
    row = result(item, checkpoint)
    row[key] = value
    (tmp_path / 's0.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='different state, policy or sample count'):
        module.aggregate([item], tmp_path, checkpoint=checkpoint)


def test_cpu_executor_uses_disjoint_states_shared_results_and_unique_worker_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'require_idle_devices', lambda gpus: None)
    checkpoint = tmp_path / 'policy'
    checkpoint.mkdir()
    (checkpoint / 'experiment.json').write_text('{}')
    states = [state(i, testable=i != 1) for i in range(12)]
    path = tmp_path / 'states.jsonl.gz'
    save_states(path, states)
    output, calls = tmp_path / 'results', []

    def invoke(commands):
        calls.append(copy.deepcopy(commands))
        assert len({command['argv'][command['argv'].index('--summary-output') + 1] for command in commands}) == len(commands)
        assert len({command['argv'][command['argv'].index('--output') + 1] for command in commands}) == 1
        for command in commands:
            source = command['argv'][command['argv'].index('--states') + 1]
            subset = module.read_states(source)
            assert [item['state_id'] for item in subset] == command['states']
            for item in subset:
                target = output / 'states' / (item['state_id'] + '.json')
                if not target.exists():
                    target.write_text(json.dumps(result(item, checkpoint)))

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    first = module.execute(checkpoint, path, 'original', 'native-source', output, gpus=4, invoke=invoke)
    hashes = [row['sha256'] for row in first['result_files']]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    second = module.execute(checkpoint, path, 'original', 'native-source', output, gpus=8, invoke=invoke)
    assert [row['sha256'] for row in second['result_files']] == hashes
    assert first['qualified_states'] == second['qualified_states'] == 12
    assert first['executed_states'] == second['executed_states'] == 11
    assert len(calls[0]) == 4 and len(calls[1]) == 8
    changed = copy.deepcopy(states)
    changed[0]['board_seed'] += 100
    save_states(path, changed)
    with pytest.raises(ValueError, match='same intervention inputs'):
        module.execute(checkpoint, path, 'original', 'native-source', output, gpus=8, invoke=invoke)
    assert len(calls) == 2


def test_zero_qualified_states_still_produce_a_complete_explicit_table(tmp_path):
    summary = summarize_results([], [])
    module.save_table(summary, tmp_path)
    lines = (tmp_path / 'intervention-table.csv').read_text().splitlines()
    assert len(lines) == 7
    assert summary['qualified_states'] == 0 and not summary['causal_textual_memory_established']


def test_memory_workflow_cannot_run_before_performance_finishes(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps(dict(status='EVALUATING')))
    with pytest.raises(ValueError, match='Finish deferred performance'):
        module.finish_memory(tmp_path, tmp_path, tmp_path / 'memory', gpus=8)
