from dataclasses import replace

import pytest

from .hero_protocol import inference_config
from .hero_v2 import identity, learning_rate, specs, stop_after_update, validate_resume_identity, write
from .test_hero_v2 import config


def continued(gpus=8):
    return replace(config(gpus), updates=448, learning_rate=1e-6, schedule_updates=128,
        continuation_start_update=320, temperature=.8, exact_kl_start_update=215)


@pytest.mark.parametrize('horizon', [449, 1024, 2048, 62500])
def test_longer_runtime_horizons_preserve_the_ml_configuration(horizon):
    original = continued()
    longer = replace(original, updates=horizon)
    assert identity(longer) == identity(original)
    inference = inference_config(longer, output='/extended', gpus=8)
    assert inference.updates == horizon and inference.window == 1024
    assert inference.thinking_token_limit == 128 and inference.action_token_limit == 32
    for update in (328, 448, 449, horizon):
        assert learning_rate(longer, update) == learning_rate(original, update) == 1e-6


@pytest.mark.parametrize('horizon', [0, -1, 62501, True, 1024.5])
def test_runtime_horizon_rejects_invalid_or_exhausted_training_budgets(horizon):
    with pytest.raises(ValueError, match='million-board pool'):
        inference_config(replace(continued(), updates=horizon), output='/extended', gpus=8)


def test_448_boundary_continues_fresh_boards_rngs_and_the_same_four_or_eight_gpu_batch():
    prior = continued()
    longer = replace(prior, updates=1024)
    permutation = list(range(1000000))
    before = specs(prior, 448, permutation)
    after = specs(longer, 449, permutation)
    assert len(before) == len(after) == 128
    assert {r['board_seed'] for r in before}.isdisjoint(r['board_seed'] for r in after)
    assert {r['board_seed'] for r in after} == set(range(448 * 16, 449 * 16))
    four = specs(replace(longer, gpus=4), 449, permutation)
    assert [{k:v for k,v in row.items() if k != 'actor_rank'} for row in after] == [
        {k:v for k,v in row.items() if k != 'actor_rank'} for row in four]
    assert identity(longer) == identity(replace(longer, gpus=4))
    final = specs(replace(longer, updates=62500), 62500, permutation)
    assert max(r['board_seed'] for r in final) == 999999


def test_metadata_only_source_migration_cannot_change_an_optimizer_or_attention_setting():
    prior = continued()
    new = replace(prior, updates=1024, source_sha256='b' * 64, accepted_resume_sources=('a' * 64,))
    validate_resume_identity(identity(prior), new, policy_version=448)
    for field, value in [('window', 512), ('learning_rate', 5e-7), ('temperature', .6), ('kl_coefficient', .02)]:
        with pytest.raises(ValueError, match='cannot change'):
            validate_resume_identity(identity(prior), replace(new, **{field:value}), policy_version=448)


def test_graceful_stop_is_opt_in_and_preserves_the_training_identity(tmp_path):
    cfg = replace(continued(), updates=1024, output=str(tmp_path))
    assert stop_after_update(cfg) == 1024
    path = tmp_path / 'stop-after-update.json'
    write(path, dict(after_update=721, identity=identity(cfg), reason='Performance target met'))
    assert stop_after_update(cfg) == 721
    assert learning_rate(cfg, 721) == 1e-6
    for request in [dict(after_update=True, identity=identity(cfg)),
            dict(after_update=0, identity=identity(cfg)),
            dict(after_update=1025, identity=identity(cfg)),
            dict(after_update=721, identity=identity(replace(cfg, window=512)))]:
        write(path, request)
        with pytest.raises(ValueError, match='training identity and horizon'):
            stop_after_update(cfg)
