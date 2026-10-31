from types import SimpleNamespace
import copy
import gzip
import json
import sys

import pytest
import torch

from . import rng_intervene as module
from .test_rng_memory_metrics import CharacterTokenizer


def test_parallel_workers_can_store_disjoint_summaries_beside_shared_state_results(tmp_path, monkeypatch):
    import transformers

    revision = '768f209d9ea81521153ed38c47d515654e938aea'
    reference = tmp_path / revision
    reference.mkdir()
    (reference / 'generation_config.json').write_text(json.dumps(dict(eos_token_id=0)))
    checkpoint = tmp_path / 'policy'
    checkpoint.mkdir()
    (checkpoint / 'experiment.json').write_text(json.dumps(dict(policy_version=16, window=1024, model_revision=revision)))
    states = tmp_path / 'empty-states.jsonl.gz'
    states.write_bytes(gzip.compress(b''))
    monkeypatch.setattr(module, 'native_modules', lambda source: (None, None, None))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda reference: CharacterTokenizer())
    results = tmp_path / 'results'
    for rank in (0, 1):
        summary = tmp_path / f'worker-{rank}-summary.json'
        monkeypatch.setattr(sys, 'argv', ['rng_intervene', '--checkpoint', str(checkpoint),
            '--states', str(states), '--reference', str(reference), '--source', 'unused',
            '--output', str(results), '--summary-output', str(summary)])
        module.main()
        assert json.loads(summary.read_text())['qualified_states'] == 0
    assert not (results / 'summary.json').exists()


def test_intervention_cannot_mix_checkpoints_boards_windows_or_thinking_controllers(tmp_path):
    protocol = dict(thinking_token_limit=128, action_token_limit=None,
                    thinking_budget_transition='\n</think>\n\n')
    record = dict(policy_version=32, policy_identity=str(tmp_path.resolve()), greedy=True,
                  attention='ps', window=1024, board_seed=4000000, **protocol)
    state = dict(window=1024, board_seed=4000000, continuation_protocol=protocol)
    metadata = dict(policy_version=32, window=1024, **protocol)
    module.validate_source(record, state, metadata, tmp_path)
    for key, value in [('policy_version',16), ('policy_identity','another'), ('greedy',False),
                       ('attention','full'), ('window',512), ('board_seed',4000001),
                       ('thinking_token_limit',None), ('thinking_budget_transition',None)]:
        wrong = dict(record, **{key:value})
        with pytest.raises(ValueError):
            module.validate_source(wrong, state, metadata, tmp_path)


def result(index, error=.001):
    row = dict(state_id=str(index), board_seed=4000000 + index // 8,
        partner='aA', streaming_action='aA', correct_partners=['aA'],
        streaming_parity=dict(absolute_error=error))
    for branch in ('unedited', 'swap', 'sham'):
        false = branch == 'swap'
        row[branch] = dict(greedy=dict(action='bB' if false else 'aA', correct=not false, follows_false=false),
            probability_correct_partner=0. if false else 1., probability_following_false=1. if false else 0.)
    return row


def test_hero_controller_uses_its_fixed_transition_without_editing_export_metadata(tmp_path):
    from .hero_protocol import THINKING_TRANSITION

    metadata = dict(variant='hero-v2', policy_version=16, window=1024,
                    thinking_token_limit=128, action_token_limit=32)
    original = copy.deepcopy(metadata)
    protocol = dict(thinking_token_limit=128, action_token_limit=32,
                    thinking_budget_transition=THINKING_TRANSITION)
    record = dict(policy_version=16, policy_identity=str(tmp_path.resolve()), greedy=True,
                  attention='ps', window=1024, board_seed=4000000, **protocol)
    state = dict(window=1024, board_seed=4000000, continuation_protocol=protocol)
    module.validate_source(record, state, metadata, tmp_path)
    assert metadata == original
    for key, value in [('thinking_token_limit', 64), ('action_token_limit', None),
                       ('thinking_budget_transition', '\n</think>\n\n')]:
        wrong = dict(record, **{key: value})
        with pytest.raises(ValueError, match='thinking controller'):
            module.validate_source(wrong, state, metadata, tmp_path)
    with pytest.raises(ValueError, match='declares a different thinking controller'):
        module.validate_source(record, state, dict(metadata, thinking_budget_transition='different'), tmp_path)


def test_textual_memory_conclusion_requires_states_complete_branches_parity_and_effect():
    states = [dict(state_id=str(index), state_boundary='before_thinking') for index in range(32)]
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


def test_fixing_current_thinking_can_only_establish_action_cursor_text_usage():
    states = [dict(state_id=str(index), state_boundary='first_coordinate') for index in range(32)]
    summary = module.summarize_results(states, [result(index) for index in range(32)])
    assert summary['causal_action_cursor_text_usage_established']
    assert not summary['causal_textual_memory_established']


def test_parity_checks_every_sampled_token_rather_than_only_the_action_cursor():
    states = [dict(state_id=str(index), state_boundary='before_thinking') for index in range(32)]
    rows = [result(index) for index in range(32)]
    for row in rows:
        row['streaming_parity']['absolute_errors'] = [.001, .1]
    assert not module.summarize_results(states, rows)['causal_textual_memory_established']


def test_intervention_uncertainty_clusters_repeated_states_from_the_same_board():
    states = [dict(state_id=str(index)) for index in range(32)]
    rows = [result(index) for index in range(32)]
    for index, row in enumerate(rows):
        row['board_seed'] = 4000000 + index // 16
        if index >= 16:
            row['swap']['probability_following_false'] = 0.
            row['sham']['probability_following_false'] = 1.
    effect = module.summarize_results(states, rows)['false_association_effect_swap_minus_sham']
    assert effect['board_clusters'] == 2
    assert effect['ci95'] == [-1., 1.]
    for row in rows:
        row['board_seed'] = 4000000
    summary = module.summarize_results(states, rows)
    assert summary['false_association_effect_swap_minus_sham']['ci95'] is None
    assert not summary['causal_textual_memory_established']


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


def test_prethinking_continuation_reuses_native_budget_and_resamples_thoughts(monkeypatch):
    class Tokenizer:
        eos_token_id = 127
        def encode(self, text, **kwargs):
            return list(map(ord, text.replace('</think>', chr(126))))
        def decode(self, tokens, skip_special_tokens=True, **kwargs):
            return ''.join('</think>' if t == 126 else chr(t) for t in tokens if t != 127 or not skip_special_tokens)
    tokenizer = Tokenizer()
    transition = '\n</think>\n\n'
    action = tokenizer.encode('Action: aA\n')
    scripts = [[126] + action, [ord('x')] * 2 + [ord('?')] * 4 + action]
    forwarded = []
    class Stream:
        cache_bytes = 1234
        def __init__(self, base, prefix, window, copies):
            self.copies, self.index = copies, 0
        def logits(self):
            values = torch.full((self.copies, 128), -1000.)
            for i in range(self.copies):
                script = scripts[bool(i)]
                values[i, script[self.index] if self.index < len(script) else 127] = 1000.
            return values
        def prefill(self, tokens, positions):
            assert tokens == [1, 2, 3] and positions == [0, 1, 2]
            return self.logits()
        def step(self, tokens):
            forwarded.append(tokens.tolist())
            self.index += 1
            return self.logits()
    class Env:
        rows = cols = 4
        def __init__(self):
            self.board, self._last_result = {'matched':False}, None
        @property
        def last_result(self):
            return self._last_result
        def step(self, action):
            assert action == (0, 0)
            self.board['matched'] = True
            self._last_result = dict(matched=True)
    monkeypatch.setattr(module, 'NativeCold', Stream)
    protocol = dict(thinking_token_limit=2, action_token_limit=None, thinking_budget_transition=transition)
    record = dict(tokens=[1, 2, 3] + [0] * 97, roles=['prefix'] * 3 + ['environment_template'] * 97,
                  turns=[dict(start=100)], **protocol)
    state = dict(prefix_length=3, window=16, retained_positions=[0, 1, 2], decision_position=100,
                 response_index=0, false_partner='bB', continuation_protocol=protocol)
    env = Env()
    row = module.continue_turn(SimpleNamespace(device='cpu'), tokenizer, {127}, state, [1, 2, 3], record,
        samples=4, rng_seed=11, parse_coord=lambda text, rows, cols: (0, 0) if text == 'Action: aA' else None,
        environment=env, invalid_type=ValueError, transcript_cap=140)
    assert row['greedy']['correct'] and row['greedy']['thinking_tokens'] == 0
    assert all(r['correct'] and r['thinking_tokens'] == 2 for r in row['sampled'])
    assert all(r['thinking_budget_forced'] for r in row['sampled'])
    assert row['sampled'][0]['tokens'] == [ord('x')] * 2 + tokenizer.encode(transition) + action
    assert row['sampled'][0]['sampled_ids'] == [ord('x')] * 2 + action
    assert row['sampled'][0]['roles'].count('thinking_budget') == 4
    assert [tokens[1] for tokens in forwarded[:6]] == [ord('x')] * 2 + tokenizer.encode(transition)
    assert row['probability_correct_partner'] == 1. and row['resource_terminations'] == 0
    assert not env.board['matched'] and env.last_result is None
