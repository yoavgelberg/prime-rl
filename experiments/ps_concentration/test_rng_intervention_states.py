import copy

import pytest

from .rng_intervention_states import apply_edits, associations, collect_states, edit_size, swap_edits
from .test_rng_memory_metrics import CharacterTokenizer


class SpaceRowTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        tokens, offset = [], 0
        while offset < len(text):
            if text[offset] == ' ' and offset + 1 < len(text) and text[offset + 1] in 'abcd':
                tokens.append(65536 + ord(text[offset + 1]))
                offset += 2
            else:
                tokens.append(ord(text[offset]))
                offset += 1
        return tokens

    def decode(self, tokens, **kwargs):
        return ''.join(' ' + chr(token - 65536) if token >= 65536 else chr(token) for token in tokens)

    def __call__(self, text, **kwargs):
        tokens = self.encode(text)
        offsets, start = [], 0
        for token in tokens:
            end = start + len(self.decode([token]))
            offsets.append((start, end))
            start = end
        return dict(input_ids=tokens, offset_mapping=offsets)


def test_bullet_and_inline_associations_include_only_token_bound_leading_spaces():
    tokenizer = SpaceRowTokenizer()
    for text in ('- bA = 0', 'Previously, bA = 0', ' bA=0'):
        tokens = tokenizer.encode(text)
        record = dict(tokens=tokens, roles=['assistant'] * len(tokens))
        notes = associations(record, tokenizer, 0, len(tokens))
        assert len(notes) == 1 and notes[0]['position'] == 'bA'
        assert notes[0]['surface'] == ' bA'
        assert tokenizer.decode(tokens[notes[0]['start']:notes[0]['end']]) == ' bA'
        record['roles'] = ['environment_template'] * len(tokens)
        assert associations(record, tokenizer, 0, len(tokens)) == []


def test_space_bound_coordinate_swap_preserves_whitespace_when_partner_is_not_written():
    tokenizer = SpaceRowTokenizer()
    text = '- aA=0'
    tokens = tokenizer.encode(text)
    record = dict(tokens=tokens, roles=['assistant'] * len(tokens))
    notes = associations(record, tokenizer, 0, len(tokens))
    edits = swap_edits(notes, 'aA', 'bA', record, tokenizer)
    assert len(edits) == 1
    edit = edits[0]
    assert edit['replacement_surface'] == ' bA'
    assert len(edit['old']) == len(edit['new']) == 2
    changed = tokens[:edit['start']] + edit['new'] + tokens[edit['end']:]
    assert tokenizer.decode(changed) == '- bA=0'


def test_token_bound_bullets_allow_equal_swap_and_sham_without_changing_observations():
    record, game = fixture('- aA=0\n- aB=1\n- aC=2\n- aD=3')
    tokenizer = SpaceRowTokenizer()
    text = CharacterTokenizer().decode(record['tokens'])
    tokens = tokenizer.encode(text)
    offsets = [0]
    roles = []
    for token in tokens:
        finish = offsets[-1] + len(tokenizer.decode([token]))
        span_roles = record['roles'][offsets[-1]:finish]
        assert len(set(span_roles)) == 1
        roles.append(span_roles[0])
        offsets.append(finish)
    record['turns'] = [dict(turn, start=offsets.index(turn['start']), end=offsets.index(turn['end']))
                       for turn in record['turns']]
    record.update(tokens=tokens, roles=roles)
    state, = collect_states(record, game, tokenizer)
    original = tokenizer.decode(state['retained_token_ids'])
    changed = apply_edits(state, state['edits'])
    sham = apply_edits(state, state['sham_edits'])
    assert tokenizer.decode(changed) == original.replace(' aA=0', ' aB=0').replace(' aB=1', ' aA=1')
    assert tokenizer.decode(sham) == original.replace(' aC=2', ' aD=2').replace(' aD=3', ' aC=3')
    assert len(changed) == len(sham) == len(state['retained_token_ids'])
    assert edit_size(state['edits']) == edit_size(state['sham_edits']) == (4, 2)


def fixture(notes='aA=0; aB=1; aC=2; aD=3;', *, notes_in_current_turn=False):
    actions = ['aA', 'aB', 'aC', 'aD', 'bA', 'bB', 'bC', 'bD', 'cA', 'aA']
    ranks = ['0', '1', '2', '3', '4', '5', '6', '7', '0', '0']
    text, turns, native = 'SYS', [], []
    for index, (action, rank) in enumerate(zip(actions, ranks, strict=True)):
        visible = {actions[index - 1]: ranks[index - 1]} if index else {}
        board = '\n'.join(row + ' ' + ''.join('[' + visible.get(row + col, '*') + ']' for col in 'ABCD') for row in 'abcd')
        text += board + '\n'
        start = len(text)
        if index == (9 if notes_in_current_turn else 8):
            text += 'Thinking filler ' * 100 + '\n' + notes + '\n'
        text += '</think>\nAction: ' + action + '\n'
        turns.append(dict(start=start, end=len(text), stop_condition='action_line'))
        native.append(dict(round=index // 2 + 1, parse=dict(ok=True, coord=action),
            env_result=dict(applied=True, revealed_face=rank,
                phase_after='flip_second' if index % 2 == 0 else 'flip_first')))
    roles = ['environment_template'] * len(text)
    roles[:3] = ['prefix'] * 3
    for turn in turns:
        roles[turn['start']:turn['end']] = ['assistant'] * (turn['end'] - turn['start'])
    return dict(tokens=list(map(ord, text)), turns=turns, prefix_length=3, window=128,
        roles=roles, attention='ps', greedy=True, board_seed=4000000), dict(config=dict(rows=4, cols=4), turns=native)


def test_qualifying_state_precedes_current_thinking_and_has_equal_size_controls():
    record, game = fixture()
    states = collect_states(record, game, CharacterTokenizer())
    assert len(states) == 1
    state = states[0]
    assert state['partner'] == 'aA' and state['false_partner'] == 'aB'
    assert state['direct_reveal_age'] > 2 * state['window']
    assert state['last_direct_reveal'] not in state['retained_positions']
    assert state['state_boundary'] == 'before_thinking'
    assert state['decision_position'] == record['turns'][9]['start']
    assert all(item['end'] <= state['decision_position'] for item in state['qualifying_associations'])
    assert edit_size(state['edits']) == edit_size(state['sham_edits']) == (4, 2)
    assert state['retained_positions'][:3] == [0, 1, 2]
    assert state['retained_positions'][3] > 3
    original = list(state['retained_token_ids'])
    changed = apply_edits(state, state['edits'])
    sham = apply_edits(state, state['sham_edits'])
    assert len(changed) == len(sham) == len(original) == 131
    assert sum(a != b for a, b in zip(changed, original)) == sum(a != b for a, b in zip(sham, original)) == 2
    assert state['retained_token_ids'] == original


def test_current_thinking_cannot_qualify_as_retained_memory_for_the_primary_intervention():
    record, game = fixture(notes_in_current_turn=True)
    assert collect_states(record, game, CharacterTokenizer()) == []
    diagnostic = collect_states(record, game, CharacterTokenizer(), boundary='first_coordinate')
    assert len(diagnostic) == 1 and diagnostic[0]['state_boundary'] == 'first_coordinate'
    assert diagnostic[0]['decision_position'] == len(record['tokens']) - 3


def test_primary_qualification_does_not_filter_out_malformed_streaming_actions():
    record, game = fixture()
    start = record['turns'][-1]['start']
    text = '</think>\nMalformed\n'
    record['tokens'][start:] = list(map(ord, text))
    record['roles'][start:] = ['assistant'] * len(text)
    record['turns'][-1]['end'] = len(record['tokens'])
    game['turns'][-1]['parse'] = dict(ok=False, coord=None)
    game['turns'][-1]['env_result'] = dict(applied=False)
    states = collect_states(record, game, CharacterTokenizer())
    assert len(states) == 1 and states[0]['streaming_action'] is None
    assert states[0]['decision_position'] == start


def test_unfinished_source_turns_cannot_silently_hide_qualification_opportunities():
    record, game = fixture()
    record['turns'][-1]['stop_condition'] = 'transcript_capacity'
    with pytest.raises(ValueError, match='qualification coverage unknown'):
        collect_states(record, game, CharacterTokenizer())


def test_recent_notes_never_replace_old_direct_environment_reveal_age():
    record, game = fixture()
    state = collect_states(record, game, CharacterTokenizer())[0]
    assert state['query_position'] - state['qualifying_associations'][0]['start'] < record['window']
    assert state['query_position'] - state['last_direct_reveal'] > 2 * record['window']


def test_environment_text_and_past_actions_cannot_qualify_as_memory_notes():
    for action in ('Action:', '**Action**:', 'previous Action:'):
        record, game = fixture(action + ' aA=0; aB=1; aC=2; aD=3;')
        assert collect_states(record, game, CharacterTokenizer()) == []
    record, game = fixture('aA=0; aB=1; aC=2; aD=3; Action: cC')
    assert collect_states(record, game, CharacterTokenizer()) == []
    record, game = fixture()
    text = ''.join(map(chr, record['tokens']))
    start = text.index('aA=0')
    record['roles'][start:start + len('aA=0; aB=1; aC=2; aD=3;')] = ['environment_template'] * len('aA=0; aB=1; aC=2; aD=3;')
    assert collect_states(record, game, CharacterTokenizer()) == []


def test_edit_validation_rejects_forced_tokens_changed_length_and_reused_spans():
    record, game = fixture()
    state = collect_states(record, game, CharacterTokenizer())[0]
    changed = copy.deepcopy(state['edits'])
    changed[0]['new'].append(42)
    with pytest.raises(ValueError, match='preserve token count'):
        apply_edits(state, changed)
    with pytest.raises(ValueError, match='disjoint'):
        apply_edits(state, state['edits'] + state['edits'])
    forced = copy.deepcopy(state)
    offset = forced['retained_positions'].index(state['edits'][0]['start'])
    forced['retained_roles'][offset] = 'environment_template'
    with pytest.raises(ValueError, match='forced observations'):
        apply_edits(forced, state['edits'])


def test_qualified_but_untestable_states_are_retained_in_denominator():
    record, game = fixture('aA=0;')
    states = collect_states(record, game, CharacterTokenizer())
    assert len(states) == 1
    assert 'untestable_reason' in states[0] and 'edits' not in states[0]


def test_manual_annotations_must_point_to_exact_retained_assistant_coordinates():
    record, _ = fixture()
    text = ''.join(map(chr, record['tokens']))
    start = text.index('aA=0')
    manual = [dict(start=start, end=start + 2, position='aA', rank='0')]
    result = associations(record, CharacterTokenizer(), start, len(text), manual=manual)
    assert any(item['recognition'].startswith('Manual') for item in result)
    record['roles'][start] = 'environment_template'
    with pytest.raises(ValueError, match='retained sampled scratch'):
        associations(record, CharacterTokenizer(), start, len(text), manual=manual)


def test_manual_surface_aliases_preserve_learned_format_when_swapping_positions():
    record, game = fixture('(1,1)=0; (1,2)=1; (1,3)=2; (1,4)=3;')
    text = ''.join(map(chr, record['tokens']))
    manual = []
    for index, position in enumerate(('aA','aB','aC','aD')):
        surface = f'(1,{index + 1})'
        lo = text.index(surface)
        manual.append(dict(start=lo, end=lo + len(surface), position=position, surface=surface, rank=str(index)))
    state = collect_states(record, game, CharacterTokenizer(), manual={9:manual})[0]
    assert state['partner'] == 'aA' and state['false_partner'] == 'aB'
    assert edit_size(state['edits']) == edit_size(state['sham_edits']) == (10,2)
    changed = CharacterTokenizer().decode(apply_edits(state, state['edits']))
    assert '(1,2)=0; (1,1)=1;' in changed
    assert 'Action: aA' not in changed


def test_manual_alias_cannot_include_the_past_action_marker_itself():
    record, _ = fixture()
    text = ''.join(map(chr, record['tokens']))
    start = text.index('Action: aA')
    manual = [dict(start=start, end=start + len('Action: aA'), position='aA', surface='Action: aA', rank='0')]
    with pytest.raises(ValueError, match='past action line'):
        associations(record, CharacterTokenizer(), start, len(text), manual=manual)
