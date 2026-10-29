import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .state_screen import count_recall, parse
from .rng_screen import AnswerLine


@pytest.mark.parametrize('name', ['CountRecallEasy','CountRecallMedium','CountRecallHard'])
def test_count_order_from_observations(name):
    env = getattr(count_recall, name)()
    observation, _ = env.reset(seed=19)
    counts = np.zeros(env.num_distinct_cards, dtype=np.int64)
    rewards = []
    for _ in range(env.max_episode_length):
        dealt, query = map(int, observation)
        counts[dealt] += 1
        observation, reward, terminated, truncated, _ = env.step(int(counts[query]))
        assert reward == 1/env.max_episode_length
        rewards.append(reward)
    assert terminated and not truncated
    assert math.isclose(math.fsum(rewards), 1.)


def test_count_answer_parser():
    assert parse('It has appeared twice.\nANSWER 2', 16) == (2, False)
    for text in ('ANSWER 2\nANSWER 3', 'ANSWER -1', 'ANSWER 17', 'ANSWER 2\nMore text', ''):
        assert parse(text,16) == (0,True)


@pytest.mark.parametrize('text,thinking,expected', [
    ('ANSWER 1',False,False),('ANSWER 12\n',False,True),
    ('ANSWER 1\n',True,False),('</think>\nANSWER 1\n',True,True)])
def test_answer_stop_waits_for_complete_action(text,thinking,expected):
    tokenizer = SimpleNamespace(decode=lambda *a,**kw:text)
    assert AnswerLine(tokenizer,0,thinking)(torch.tensor([[0]]),None) == expected
