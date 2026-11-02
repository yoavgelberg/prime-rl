from dataclasses import replace

import pytest

from .hero_protocol import inference_config
from .hero_v2 import identity, kl_estimator, learning_rate, specs
from .test_hero_v2 import config


def test_post320_temperature_warmup_and_fresh_boards():
    parent = replace(config(4), updates=320, temperature=.6, learning_rate=1e-6,
        continuation_start_update=192, exact_kl_start_update=215)
    continued = replace(parent, gpus=8, updates=448, temperature=.8,
        continuation_start_update=320)
    inference = inference_config(continued, output='/new-rollouts', gpus=1)
    assert inference.temperature == .8 and inference.updates == 448
    assert inference.thinking_token_limit == 128 and inference.action_token_limit == 32
    assert kl_estimator(continued, 321) == 'categorical'
    assert learning_rate(continued, 321) == pytest.approx(4.75e-7)
    assert learning_rate(continued, 328) == pytest.approx(1e-6)
    before, after = identity(parent), identity(continued)
    for field in ('temperature', 'continuation_lr_schedule'):
        before.pop(field)
        after.pop(field)
    assert before == after
    permutation = list(range(1_000_000))
    previous = specs(parent, 320, permutation)
    following = specs(continued, 321, permutation)
    assert len(following) == 128
    assert len({row['board_seed'] for row in following}) == 16
    assert {row['board_seed'] for row in previous}.isdisjoint(row['board_seed'] for row in following)
    assert continued.live_games == 32
    fallback = replace(continued, gpus=4)
    assert identity(fallback) == identity(continued)
    assert fallback.live_games == 64
