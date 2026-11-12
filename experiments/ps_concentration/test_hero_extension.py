from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from .hero_continue import state_fingerprint, training_state
from .hero_extension import extension_config, prepare
from .hero_v2 import identity, learning_rate, specs, write
from .test_hero_v2 import config


def test_extension_warmup_and_paired_boards():
    parent = replace(config(4), updates=192, schedule_updates=128)
    left, right = [extension_config(parent, window=w, output=f'/out/w{w}',
        node_dir=f'/node/w{w}/control', source_sha256='b'*64, deadline=99999) for w in (1024, 512)]
    assert learning_rate(left, 192) == pytest.approx(4e-7)
    assert learning_rate(left, 193) == pytest.approx(4.75e-7)
    assert learning_rate(left, 200) == pytest.approx(1e-6)
    assert learning_rate(left, 320) == pytest.approx(1e-6)
    assert identity(left) | {'window': 512} == identity(right)
    permutation = list(range(1_000_000))
    for update in (193, 200, 256, 320):
        for a, b in zip(specs(left, update, permutation), specs(right, update, permutation), strict=True):
            assert {k: v for k, v in a.items() if k != 'policy_identity'} == {
                k: v for k, v in b.items() if k != 'policy_identity'}


def test_extension_fork_preserves_all_training_state(tmp_path):
    parent_run = tmp_path / 'old/arms/w1024'
    parent = replace(config(4), output=str(parent_run), updates=192, schedule_updates=128)
    path = parent_run / 'checkpoints/latest.pt'
    path.parent.mkdir(parents=True)
    state = dict(policy_version=192, model={str(i): torch.tensor([float(i)]) for i in range(398)},
        optimizer=dict(state={i: dict(step=torch.tensor(192.), exp_avg=torch.tensor([.1])) for i in range(398)},
            param_groups=[dict(lr=4e-7)]), rngs=dict(torch=torch.tensor([1, 2], dtype=torch.uint8)),
        rngs_by_rank=[dict(torch=torch.tensor([i], dtype=torch.uint8)) for i in range(2)],
        identity=identity(parent), resolved_config=asdict(parent))
    torch.save(state, path)
    original = path.read_bytes()
    write(parent_run / 'resolved_config.json', asdict(parent))
    write(path.with_suffix('.pt.json'), dict(policy_version=192, identity=identity(parent)))
    output = tmp_path / 'extension'
    plan = prepare(parent_run, output, 'b'*64, tmp_path / 'node')
    assert path.read_bytes() == original
    for branch in plan['branches']:
        new = torch.load(Path(branch['output']) / 'checkpoints/latest.pt', weights_only=True)
        assert state_fingerprint(training_state(new)) == state_fingerprint(training_state(state))
        assert new['optimizer']['param_groups'][0]['lr'] == 4e-7
        assert new['identity'] == identity(type(parent)(**branch))
    assert prepare(parent_run, output, 'b'*64, tmp_path / 'node') == plan
