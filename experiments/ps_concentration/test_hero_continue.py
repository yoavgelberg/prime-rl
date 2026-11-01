from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from .hero_continue import branch_config, fork_checkpoint, parent_checkpoint, state_fingerprint, training_state
from .hero_v2 import identity, specs, write
from .test_hero_v2 import config


@pytest.fixture
def parent(tmp_path):
    run = tmp_path / 'parent'
    cfg = replace(config(), output=str(run))
    checkpoint = run / 'checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    state = dict(format='ps-hero-v2-adamw-v1', policy_version=128,
        model={'weight': torch.tensor([1., 2., 3.])},
        optimizer={'state': {0: dict(step=torch.tensor(128.), exp_avg=torch.tensor([.1, .2, .3]),
                                   exp_avg_sq=torch.tensor([.01, .02, .03]))},
                   'param_groups': [dict(lr=4e-7, betas=(.9, .95), eps=1e-8, weight_decay=0., params=[0])]},
        rngs=dict(torch=torch.arange(8, dtype=torch.uint8), cuda=torch.tensor([4, 2], dtype=torch.uint8)),
        rngs_by_rank=[dict(torch=torch.tensor([i], dtype=torch.uint8),
                          cuda=torch.tensor([i + 4], dtype=torch.uint8)) for i in range(4)],
        identity=identity(cfg), resolved_config=asdict(cfg))
    torch.save(state, checkpoint)
    write(run / 'resolved_config.json', asdict(cfg))
    write(run / 'state.json', dict(policy_version=128, identity=identity(cfg)))
    write(run / 'training-complete.json', dict(policy_version=128, reached_target=True))
    write(checkpoint.with_suffix('.pt.json'), dict(policy_version=128, identity=identity(cfg),
        bytes=checkpoint.stat().st_size))
    stat = checkpoint.stat()
    write(run / 'export-catalog/policy-0128.json', dict(status='VERIFIED', policy_version=128,
        identity=identity(cfg), unique_parameters=3,
        snapshot_stamp=dict(inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)))
    return run, cfg, state


def branches(cfg, tmp_path):
    return [branch_config(cfg, window=window, output=tmp_path / f'w{window}',
        node_dir=tmp_path / f'node/w{window}/control', source_sha256='b' * 64, deadline=20000)
        for window in (1024, 256)]


def test_both_forks_preserve_complete_training_state_and_are_idempotent(parent, tmp_path):
    run, cfg, original = parent
    configs = [type(cfg)(**json.loads(json.dumps(asdict(value)))) for value in branches(cfg, tmp_path)]
    receipts = fork_checkpoint(run, configs, expected_parameters=3)
    expected = state_fingerprint(training_state(original))
    assert {r['training_state_sha256'] for r in receipts} == {expected}
    for branch in configs:
        loaded = torch.load(Path(branch.output) / 'checkpoints/latest.pt', weights_only=True)
        assert state_fingerprint(training_state(loaded)) == expected
        assert loaded['optimizer']['param_groups'][0]['lr'] == 4e-7
        assert loaded['identity'] == identity(branch)
        assert loaded['resolved_config']['updates'] == 192
        assert (Path(branch.output) / 'checkpoints').is_symlink()
    assert fork_checkpoint(run, configs, expected_parameters=3) == receipts


def test_pairing_keeps_all_board_and_generation_rngs_identical(parent, tmp_path):
    _, cfg, _ = parent
    left, right = branches(cfg, tmp_path)
    permutation = list(range(1_000_000))
    for update in (129, 144, 192):
        a, b = specs(left, update, permutation), specs(right, update, permutation)
        for x, y in zip(a, b, strict=True):
            assert {k:v for k,v in x.items() if k != 'policy_identity'} == {
                k:v for k,v in y.items() if k != 'policy_identity'}
        assert len(a) == 128


def test_parent_requires_complete_budget_and_matching_export(parent):
    run, _, _ = parent
    parent_checkpoint(run, expected_parameters=3)
    write(run / 'training-complete.json', dict(policy_version=128, reached_target=False))
    with pytest.raises(ValueError, match='complete training budget'):
        parent_checkpoint(run, expected_parameters=3)


def test_fork_rejects_optimizer_or_seed_change(parent, tmp_path):
    run, cfg, _ = parent
    left, right = branches(cfg, tmp_path)
    for changed in (replace(left, learning_rate=1e-6), replace(left, seed=22)):
        with pytest.raises(ValueError, match='paired continuation protocol'):
            fork_checkpoint(run, [changed, right], expected_parameters=3)
    assert not Path(left.output).exists()


def test_fork_never_overwrites_modified_optimizer(parent, tmp_path):
    run, cfg, _ = parent
    configs = branches(cfg, tmp_path)
    fork_checkpoint(run, configs, expected_parameters=3)
    path = Path(configs[1].output) / 'checkpoints/latest.pt'
    state = torch.load(path, weights_only=True)
    state['optimizer']['state'][0]['exp_avg'][0] += 1
    torch.save(state, path)
    with pytest.raises(ValueError, match='Preserve an existing branch state'):
        fork_checkpoint(run, configs, expected_parameters=3)
    assert torch.load(path, weights_only=True)['optimizer']['state'][0]['exp_avg'][0] > 1


def test_fingerprint_detects_rng_and_moment_changes(parent):
    _, _, state = parent
    initial = state_fingerprint(training_state(state))
    state['rngs_by_rank'][3]['cuda'][0] += 1
    assert state_fingerprint(training_state(state)) != initial
