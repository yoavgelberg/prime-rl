from collections import deque
from types import SimpleNamespace

import pytest
import torch

from .rng_batch import BatchDecoder
from .rng_collect import TrainingDecoder
from .rng_training_config import outcome_reward, training_rng


def test_training_sampler_preserves_tokens_and_actual_likelihoods():
    args = SimpleNamespace(temperature=.6, top_p=.95, top_k=20)
    logits_rng = torch.Generator().manual_seed(123)
    outputs = [[], []]
    decoders = []
    for number, cls in enumerate((BatchDecoder, TrainingDecoder)):
        decoder = object.__new__(cls)
        decoder.args, decoder.greedy = args, False
        decoder.clients = [SimpleNamespace(rng=torch.Generator().manual_seed(11 + i),
                                           _check_capacity=lambda count: None) for i in range(4)]
        decoder._emit = lambda index, token, lp, number=number: outputs[number].append((index, token, lp))
        decoders.append(decoder)
    for _ in range(20):
        logits = torch.randn(4, 101, generator=logits_rng)
        for decoder in decoders:
            decoder._emit_many(list(range(4)), logits)
        for index, token, _ in outputs[1][-4:]:
            expected = torch.log_softmax(logits[index] / args.temperature, -1)[token]
            assert abs(decoders[1].clients[index].next_tempered_logp - float(expected)) < 1e-6
    assert outputs[0] == outputs[1]


def test_tempered_likelihood_mask_includes_initial_scratch_tokens():
    decoder = object.__new__(TrainingDecoder)
    lane = SimpleNamespace(tokens=[1, 2], roles=['prefix', 'environment_template'],
        old_logps=[None, None], old_tempered_logps=[], queue=deque(),
        _check_capacity=lambda count: None, next_tempered_logp=-.4)
    decoder._append(lane, [3], 'assistant', True)
    decoder._append(lane, [4, 5], 'environment_template')
    lane.next_tempered_logp = -.7
    decoder._append(lane, [6], 'assistant', True)
    assert lane.old_tempered_logps == [None, None, -.4, None, None, -.7]
    assert [value is not None for value in lane.old_tempered_logps] == [role == 'assistant' for role in lane.roles]


def test_training_rng_is_independent_and_reproducible():
    seeds = [training_rng(11, 0, board, rollout) for board in range(8) for rollout in range(8)]
    assert len(set(seeds)) == 64
    assert seeds == [training_rng(11, 0, board, rollout) for board in range(8) for rollout in range(8)]
    assert not set(seeds).intersection(training_rng(11, 1, board, rollout) for board in range(8) for rollout in range(8))


def test_partial_progress_and_shorter_solutions_have_correct_rewards():
    assert outcome_reward(dict(score=8, response_count=16)) > outcome_reward(dict(score=8, response_count=30))
    assert outcome_reward(dict(score=7, response_count=96)) == -.125
    assert outcome_reward(dict(score=0, response_count=96)) == -1.
    with pytest.raises(ValueError, match='explicit'):
        outcome_reward(dict(censored=True))
    assert outcome_reward(dict(censored=True), context_exhaustion_reward=-1) == -1.
