import gzip
import json
from types import SimpleNamespace

import pytest

from .overlap_config import BoardProtocol
from .rng_vllm import GameCohort


@pytest.mark.parametrize('board', [BoardProtocol(), BoardProtocol(cols=8, max_responses=192)])
def test_native_result_saves_evaluation_identity_and_actual_board_denominator(tmp_path, board):
    subject = GameCohort.__new__(GameCohort)
    identity = dict(board_seed=1100000, episode_id='evaluation', greedy=True,
                    attention='ps', window=1024, **board.identity())
    subject.board, subject.max_responses = board, board.max_responses
    subject.attention, subject.window, subject.temperature, subject.greedy = 'ps', 1024, .6, True
    subject.source, subject.state, subject.device = tmp_path, SimpleNamespace(), None
    subject.context_exhaustion_reward = -1.
    subject.paths, subject.specs = {'evaluation':tmp_path}, {'evaluation':identity}
    chat = SimpleNamespace(entropy_tokens=4, entropy_sum=6., old_tempered_logps=[-.5] * 4,
        generated_tokens=4, tokens=[1, 2, 3, 4], started=0.,
        thinking_token_limit=128, action_token_limit=None, thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n',
        budget_enforcement='forced_thinking_transition', turns=[dict(thinking_tokens=2, thinking_budget_forced=False)],
        record=lambda:dict(old_tempered_logps=[-.5] * 4, loss_mask=[True] * 4),
        finish=lambda **kwargs:None)
    subject.chats = {'evaluation':chat}

    def native_game(**kwargs):
        assert (kwargs['rows'], kwargs['cols'], kwargs['num_cards'], kwargs['max_responses']) == (
            board.rows, board.cols, board.num_cards, board.max_responses)
        kwargs['out_dir'].mkdir()
        turns = [dict(round=1, parse=dict(ok=True, coord=position), env_result=dict(applied=True,
            revealed_face='0', phase_after=phase)) for position, phase in [('aA', 'flip_second'), ('aB', 'flip_first')]]
        native = dict(config=dict(total_pairs=board.rows * board.cols // 2, ground_truth_text='pinned board'), turns=turns)
        (kwargs['out_dir'] / 'game.json').write_text(json.dumps(native))
        return dict(score=1)

    subject.run_one_game = native_game
    row = subject._play('evaluation')
    assert row['training_return'] == pytest.approx(1 / identity['total_pairs'] - 2 / board.max_responses)
    assert all(row[key] == value for key, value in identity.items())
    with gzip.open(row['token_record'], 'rt') as handle:
        record = json.load(handle)
    assert record['total_pairs'] == identity['total_pairs'] and record['greedy'] is True
