from dataclasses import asdict, replace
import csv
from pathlib import Path

import pytest

from . import hero_extension_final as final_module
from .hero_extension import extension_config
from .hero_extension_report import comparison, report, window_rows
from .hero_v2 import read, write
from .test_hero_v2 import config


@pytest.fixture
def protocol(tmp_path):
    parent_run = tmp_path/'old/arms/w1024'
    parent = replace(config(4), output=str(parent_run), updates=192, schedule_updates=128)
    write(parent_run/'resolved_config.json', asdict(parent))
    root = tmp_path/'extension'
    branches = [extension_config(parent, window=w, output=root/f'arms/w{w}',
        node_dir=tmp_path/f'node/w{w}/control', source_sha256='b'*64, deadline=0.) for w in (1024,512)]
    plan = dict(output=str(root), parent_run=str(parent_run), parent_policy_version=192,
        target_update=320, source_sha256='b'*64, branches=[asdict(c) for c in branches])
    path = root/'plan.json';write(path, plan)
    final = final_module.prepare(path, root/'final', gpus=8,
        source_root=tmp_path/'source', source_sha256='c'*64)
    for branch in branches:
        write(Path(branch.output)/'state.json', dict(policy_version=192))
    return root/'final/plan.json', final, branches


def test_partial_report_uses_committed_batches_without_claiming_final_results(protocol, tmp_path):
    path, final, branches = protocol
    for i, branch in enumerate(branches):
        root = Path(branch.output)
        write(root/'state.json', dict(policy_version=193))
        write(root/'update-metrics/update-0193.json', dict(metrics=dict(update=193,
            episodes=128, training_episodes=193*128, mean_return=.4-i,
            mean_native_return=.5-i, mean_pairs=8-i*5, mean_attempts=45+i*45,
            seconds=350+i*300, mean_per_game_tokens_per_second=33., generated_tokens=1000000,
            gradient_norm=.1, kl_per_generated_token=.06)))
    output = tmp_path/'report'
    result = report(path, output, allow_partial=True)
    assert result['status']=='PARTIAL' and len(result['missing_conditions'])==24
    assert len(result['missing_memory'])==3 and result['paired_comparison'] is None
    assert result['training_seed_std'] is None and result['independent_training_seeds']==1
    with (output/'training-curve.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(r['pairs']) for r in rows]==[8.,3.]
    assert all(int(r['new_training_episodes'])==128 for r in rows)
    assert (output/'return-vs-training-episodes.png').stat().st_size>1000
    assert (output/'return-vs-training-episodes.pdf').stat().st_size>1000
    assert not (output/'final-return-vs-window.png').exists()
    with pytest.raises(ValueError, match='every frozen evaluation'):
        report(path, tmp_path/'strict')


def test_window_subset_does_not_use_the_other384_boards():
    rows = [dict(board_seed=5000000+i, score=0 if i<128 else 8,
        training_return=-1. if i<128 else 0., done=i>=128, response_count=96,
        generated_tokens=10, raw_transcript_tokens=20, format_errors=0, invalid_actions=0,
        seconds=1.) for i in range(512)]
    result = window_rows([dict(name='w512-ps-w512-extension_test', arm='w512', split='extension_test',
        attention='ps', window=512, expected_boards=512, rows=rows)])
    assert result[0]['mean_pairs']==0 and result[0]['mean_return']==-1
    assert result[0]['episodes']==128 and result[0]['derived_subset'] is True


def test_primary_recovery_uses_full_reference_for_the_sliding_control():
    names = ['treatment', 'control', 'w1024-full-w1024-extension_test']
    conditions = [dict(name=name, mean_return=sum(values)/2, mean_pairs=4,
        rows=[dict(board_seed=5000000+i, training_return=value) for i,value in enumerate(values)])
        for name,values in zip(names, ([.2,.4],[0.,.1],[.6,.8]), strict=True)]
    value = comparison(conditions, dict(primary=names[:2]))
    assert value['mean_difference']==pytest.approx(.25)
    assert value['full_reference_gap']==pytest.approx(.65)
    assert value['recovery']==pytest.approx(.25/.65)
    assert value['resamples']==10000 and value['training_seed_std'] is None


def test_memory_dispatch_preserves_three_collections_and_prethinking_boundary(protocol, monkeypatch):
    from . import hero_memory
    path, final, branches = protocol
    entries = [r for r in final['conditions'] if r['split']=='intervention']
    calls, barriers = [], []
    monkeypatch.setattr(final_module, 'ready', lambda output, options: barriers.append(options['training_barrier']))
    monkeypatch.setattr(final_module, 'require_idle_devices', lambda gpus: None)
    monkeypatch.setattr(final_module, 'condition_result', lambda entry: {})
    monkeypatch.setattr(final_module, 'hydrate', lambda archive, checkpoint: None)
    catalogs = {entry['archive']:dict(catalog=dict(policy_version=320,
        identity=entry['policy_training_identity'])) for entry in entries}
    monkeypatch.setattr(final_module, 'verify_policy', lambda archive: catalogs[archive])
    def execute(checkpoint, states, reference, source, output, *, gpus):
        inputs = read(Path(output)/'collection-inputs.json')
        calls.append((inputs['window'], inputs['boundary'], inputs['maximum'], gpus, str(output)))
    monkeypatch.setattr(hero_memory, 'execute', execute)
    for entry in entries:
        write(Path(entry['summary']).parent/'episodes.jsonl', [])
        output = path.parent/'memory'/f'{entry["arm"]}-w{entry["window"]}'
        write(output/'collection/collection-summary.json', dict(qualified_states=0))
    final_module.memory(path, gpus=4)
    assert barriers==[320,320]
    assert [call[:4] for call in calls]==[(512,'before_thinking',128,4),
        (512,'before_thinking',128,4),(1024,'before_thinking',128,4)]
    assert {Path(call[4]).name for call in calls}=={'w512-w512','w1024-w512','w1024-w1024'}


def test_memory_cannot_start_before_both_final_training_barriers(protocol, monkeypatch):
    path, _, _ = protocol
    def unfinished(*args):
        raise ValueError('training incomplete')
    monkeypatch.setattr(final_module, 'ready', unfinished)
    monkeypatch.setattr(final_module, 'require_idle_devices', lambda *args: pytest.fail('No device work before the barrier'))
    with pytest.raises(ValueError, match='training incomplete'):
        final_module.memory(path, gpus=8)
