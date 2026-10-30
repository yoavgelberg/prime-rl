from types import SimpleNamespace
import copy

import torch

from . import rng_intervene as module
from .test_rng_memory_metrics import CharacterTokenizer


def result(index, error=.001):
    row = dict(state_id=str(index), partner='aA', streaming_action='aA', correct_partners=['aA'],
        streaming_parity=dict(absolute_error=error))
    for branch in ('unedited', 'swap', 'sham'):
        false = branch == 'swap'
        row[branch] = dict(greedy=dict(action='bB' if false else 'aA', correct=not false, follows_false=false),
            probability_correct_partner=0. if false else 1., probability_following_false=1. if false else 0.)
    return row


def test_textual_memory_conclusion_requires_states_complete_branches_parity_and_effect():
    states = [dict(state_id=str(index)) for index in range(32)]
    rows = [result(index) for index in range(32)]
    summary = module.summarize_results(states, rows)
    assert summary['causal_textual_memory_established']
    assert summary['conditional_unedited_selects_p']['denominator'] == 32
    assert summary['false_association_effect_swap_minus_sham']['ci95'] == [1., 1.]
    assert not module.summarize_results(states[:-1], rows[:-1])['causal_textual_memory_established']
    bad = copy.deepcopy(rows)
    for row in bad:
        row['streaming_parity']['absolute_error'] = .06
    assert not module.summarize_results(states, bad)['causal_textual_memory_established']
    no_effect = copy.deepcopy(rows)
    for row in no_effect:
        row['swap'] = row['sham']
    assert not module.summarize_results(states, no_effect)['causal_textual_memory_established']
    incomplete = states + [dict(state_id='unexecuted')]
    assert not module.summarize_results(incomplete, rows)['causal_textual_memory_established']
    untestable = states + [dict(state_id='uneditable', untestable_reason='No sham')]
    assert not module.summarize_results(untestable, rows + [dict(state_id='uneditable')])['causal_textual_memory_established']


def test_uneditable_qualified_states_stay_in_reported_denominator():
    states = [dict(state_id='one', untestable_reason='No sham')]
    summary = module.summarize_results(states, [dict(state_id='one')])
    assert summary['qualified_states'] == summary['untestable_states'] == 1
    assert summary['executed_states'] == 0 and summary['pending_states'] == 0
    assert summary['branches']['unedited']['greedy_correct'] is None
    assert not summary['causal_textual_memory_established']


def test_branch_uses_native_stop_rule_independent_clones_and_separate_greedy_count(monkeypatch):
    class Stream:
        cache_bytes = 1234
        def __init__(self, base, prefix, window, copies):
            self.copies, self.index = copies, 0
        def logits(self):
            values = torch.full((self.copies, 128), -1000.)
            values[:, [ord('a'), ord('A'), ord('\n')][self.index]] = 1000.
            return values
        def prefill(self, tokens, positions):
            return self.logits()
        def step(self, tokens):
            self.index += 1
            return self.logits()
    class Env:
        rows = cols = 4
        def __init__(self):
            self.board = {'matched':False}
            self._last_result = None
        @property
        def last_result(self):
            return self._last_result
        def step(self, action):
            assert action == (0, 0)
            self.board['matched'] = True
            self._last_result = dict(matched=True)
    monkeypatch.setattr(module, 'NativeCold', Stream)
    parse = lambda text, rows, cols: (0, 0) if text == 'Action: aA\n' else None
    env = Env()
    state = dict(prefix_length=3, window=16, retained_positions=[0, 1, 2],
                 decision_position=100, false_partner='bB')
    row = module.continue_branch(SimpleNamespace(device='cpu'), CharacterTokenizer(), {127},
        state, [1, 2, 3], 'Action: ', samples=4, rng_seed=11, parse_coord=parse,
        environment=env, invalid_type=ValueError, transcript_cap=103)
    assert row['greedy']['action'] == 'aA' and len(row['sampled']) == 4
    assert row['probability_correct_partner'] == 1.
    assert row['resource_terminations'] == 0
    assert row['peak_kv_bytes'] == 1234
    assert all(value['stop_condition'] == 'action_line' for value in row['sampled'])
    assert not env.board['matched'] and env.last_result is None
