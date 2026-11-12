from dataclasses import replace

import pytest

from .hero_protocol import inference_config
from .hero_v2 import identity, kl_estimator, learning_rate, specs
from .overlap_config import SPLITS, board_protocol, evaluation_specs
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


@pytest.mark.parametrize('split,first,count,pairs,limit', [
    ('post320_test', 7_000_000, 512, 8, 96),
    ('post320_larger', 8_000_000, 256, 16, 192),
])
def test_post320_fresh_evaluation_boards_and_physical_layout(split, first, count, pairs, limit):
    from .rng_overlap import worker_options

    hero = replace(config(8), updates=448, temperature=.8, continuation_start_update=320)
    inference = inference_config(hero, output='/fresh-evaluation', gpus=8)
    rows = evaluation_specs(inference, split, 'fixed448')
    assert [row['board_seed'] for row in rows] == list(range(first, first + count))
    other_reserved = [set(range(start, start + size)) for name, (start, size) in SPLITS.items() if name != split]
    assert set(range(first, first + count)).isdisjoint(set.union(*other_reserved))
    assert min(row['board_seed'] for row in rows) >= 1_000_000
    four = evaluation_specs(replace(inference, gpus=4), split, 'fixed448')
    assert [{k: v for k, v in row.items() if k != 'rank'} for row in rows] == [
        {k: v for k, v in row.items() if k != 'rank'} for row in four]
    assert set(row['rank'] for row in rows) == set(range(8))
    assert set(row['rank'] for row in four) == set(range(4))
    board = board_protocol(split)
    options = worker_options(inference, rows, '/fresh-evaluation', greedy=True, board=board)
    assert options['greedy'] is True
    assert options['rows'] * options['cols'] // 2 == pairs
    assert options['max_responses'] == limit
    assert options['num_cards'] == 8
    assert options['thinking_token_limit'] == 128 and options['action_token_limit'] == 32
    assert options['thinking_budget_transition'] == '\n\nOkay, time to give the answer.</think>\n\n'
    assert options['attention'] == 'ps' and options['window'] == 1024
    assert len(evaluation_specs(inference, split, 'sweep448', count=128)) == 128
    with pytest.raises(ValueError, match='reserved split'):
        evaluation_specs(inference, split, 'overflow', count=count + 1)
