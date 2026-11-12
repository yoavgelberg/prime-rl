from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import hero_pair as pair
from .hero_pair_plan import prepare
from .hero_v2 import identity, read, write
from .test_hero_v2 import config


def test_assigned_devices_form_disjoint_four_gpu_branches():
    assert pair.device_groups(['4', '5', '6', '7']) == [['4', '5', '6', '7']]
    assert pair.device_groups(list('01234567')) == [list('0123'), list('4567')]
    with pytest.raises(ValueError, match='distinct'):
        pair.device_groups(['0', '0', '1', '2'])
    with pytest.raises(ValueError, match='four-or-eight'):
        pair.device_groups(list('012'))


def test_resume_rejects_uncommitted_snapshot_and_wrong_identity(tmp_path):
    cfg = replace(config(), output=str(tmp_path))
    snapshot = tmp_path / 'checkpoints/latest.pt'
    snapshot.parent.mkdir()
    snapshot.write_bytes(b'exact saved state')
    metadata = dict(policy_version=144, bytes=snapshot.stat().st_size, identity=identity(cfg))
    write(snapshot.with_suffix('.pt.json'), metadata)
    write(tmp_path / 'state.json', dict(policy_version=144))
    assert pair.validate_resume(cfg) == 144
    write(tmp_path / 'state.json', dict(policy_version=145))
    with pytest.raises(ValueError, match='lacks its exact optimizer'):
        pair.validate_resume(cfg)
    write(tmp_path / 'state.json', dict(policy_version=144))
    with pytest.raises(ValueError, match='exact declared branch identity'):
        pair.validate_resume(replace(cfg, window=256))
    snapshot.write_bytes(b'short')
    with pytest.raises(ValueError, match='different size'):
        pair.validate_resume(cfg)


@pytest.mark.parametrize('gpus', [4, 8])
@pytest.mark.parametrize('failed_window', [None, 256])
def test_executor_sweeps_then_launches_paired_branches_on_assigned_devices(
        tmp_path, monkeypatch, gpus, failed_window):
    parent_run = tmp_path / 'parent'
    parent = replace(config(), output=str(parent_run))
    write(parent_run / 'resolved_config.json', asdict(parent))
    output = tmp_path / 'paired'
    plan = prepare(parent_run, output, source_root='/captured', source_sha256='b' * 64)
    write(parent_run / 'export-catalog/policy-0128.json', {})
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'yejin')
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(gpus))))
    monkeypatch.setattr(pair, 'parent_checkpoint', lambda path: (path / 'checkpoints/latest.pt', parent))
    monkeypatch.setattr(pair, 'require_idle_devices', lambda count: None)
    monkeypatch.setattr(pair, 'preserve_policy', lambda *args, **kwargs: None)
    monkeypatch.setattr(pair, 'preserve_parent', lambda *args: None)
    monkeypatch.setattr(pair.shutil, 'disk_usage', lambda path: SimpleNamespace(free=500 * 1024**3))
    events = []

    def fork(path, configs):
        for cfg in configs:
            snapshot = Path(cfg.output) / 'checkpoints/latest.pt'
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b'exact fork')
            write(snapshot.with_suffix('.pt.json'), dict(policy_version=128,
                identity=identity(cfg), bytes=snapshot.stat().st_size))
            write(Path(cfg.output) / 'fork-receipt.json', dict(parent_policy_version=128))

    def sweep(entries, path, *, gpus):
        assert len(entries) == 7
        events.append(('sweep', gpus))

    class Process:
        def __init__(self, argv, *, env, **kwargs):
            self.cfg = read(argv[-1])
            self.pid = 100 + self.cfg['window']
            self.returncode = 1 if self.cfg['window'] == failed_window else 0
            assert pair.saved_version(self.cfg['output']) == 128
            events.append(('launch', self.cfg['window'], env['CUDA_VISIBLE_DEVICES']))
            snapshot = Path(self.cfg['output']) / 'checkpoints/latest.pt'
            metadata = read(snapshot.with_suffix('.pt.json'))
            write(snapshot.with_suffix('.pt.json'), metadata | dict(policy_version=192))

        def poll(self):
            events.append(('poll', self.cfg['window']))
            return self.returncode

    monkeypatch.setattr(pair, 'fork_checkpoint', fork)
    monkeypatch.setattr(pair, 'execute_conditions', sweep)
    monkeypatch.setattr(pair, 'observe', lambda *args: None)
    monkeypatch.setattr(pair.subprocess, 'Popen', Process)
    kwargs = dict(gpus=gpus, node_base=tmp_path / 'node', deadline=9999999999)
    if failed_window:
        with pytest.raises(RuntimeError, match='continuation failed'):
            pair.run(output / 'plan.json', **kwargs)
    else:
        result = pair.run(output / 'plan.json', **kwargs)
        assert result['status'] == 'CONTINUATIONS_COMPLETE_FINAL_EVAL_PENDING'
    assert events[0] == ('sweep', gpus)
    launches = [event for event in events if event[0] == 'launch']
    assert [event[1] for event in launches] == [1024, 256]
    assert launches[0][2] == '0,1,2,3'
    assert launches[1][2] == ('4,5,6,7' if gpus == 8 else '0,1,2,3')
    if gpus == 8:
        assert events[1:3] == launches
    else:
        assert events.index(('poll', 1024)) < events.index(launches[1])
    assert all(read(Path(row['output']) / 'fork-receipt.json')['parent_policy_version'] == 128
        for row in plan['branches'])


def test_storage_budget_counts_already_preserved_unique_files(tmp_path):
    plan = dict(output=str(tmp_path), storage=dict(required_shared_headroom_gib=185))
    path = tmp_path / 'policies/parent-0128/model.safetensors'
    path.parent.mkdir(parents=True)
    with path.open('wb') as handle:
        handle.truncate(8 * 1024**3)
    assert pair.required_fork_storage(plan, []) == 177
