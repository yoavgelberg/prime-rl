import copy

import pytest

from .rng_intervention_states import apply_edits, associations, collect_states, edit_size
from .test_rng_memory_metrics import CharacterTokenizer


def fixture(notes='aA=0; aB=1; aC=2; aD=3;'):
    actions = ['aA', 'aB', 'aC', 'aD', 'bA', 'bB', 'bC', 'bD', 'cA', 'aA']
    ranks = ['0', '1', '2', '3', '4', '5', '6', '7', '0', '0']
    text, turns, native = 'SYS', [], []
    for index, (action, rank) in enumerate(zip(actions, ranks, strict=True)):
        visible = {actions[index - 1]: ranks[index - 1]} if index else {}
        board = '\n'.join(row + ' ' + ''.join('[' + visible.get(row + col, '*') + ']' for col in 'ABCD') for row in 'abcd')
        text += board + '\n'
        start = len(text)
        if index == 9:
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


def test_qualifying_state_is_before_first_coordinate_and_has_equal_size_controls():
    record, game = fixture()
    states = collect_states(record, game, CharacterTokenizer())
    assert len(states) == 1
    state = states[0]
    assert state['partner'] == 'aA' and state['false_partner'] == 'aB'
    assert state['direct_reveal_age'] > 2 * state['window']
    assert state['last_direct_reveal'] not in state['retained_positions']
    assert state['decision_position'] == len(record['tokens']) - 3
    assert record['tokens'][state['decision_position']] == ord('a')
    assert edit_size(state['edits']) == edit_size(state['sham_edits']) == (4, 2)
    assert state['retained_positions'][:3] == [0, 1, 2]
    assert state['retained_positions'][3] > 3
    original = list(state['retained_token_ids'])
    changed = apply_edits(state, state['edits'])
    sham = apply_edits(state, state['sham_edits'])
    assert len(changed) == len(sham) == len(original) == 131
    assert sum(a != b for a, b in zip(changed, original)) == sum(a != b for a, b in zip(sham, original)) == 2
    assert state['retained_token_ids'] == original


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
