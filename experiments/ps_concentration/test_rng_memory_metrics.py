from .rng_memory_metrics import known_match_decisions, summarize


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return ''.join(map(chr, tokens))

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(map(ord, text)), offset_mapping=[(i, i + 1) for i in range(len(text))])


def example(final_action='aA'):
    observations = ['a [*][*]\nb [*][*]\n', 'a [0][*]\nb [*][*]\n',
                    'a [0][*]\nb [1][*]\n\na [*][*]\nb [*][*]\n',
                    'a [*][0]\nb [*][*]\n']
    actions = ['aA', 'bA', 'aB', final_action]
    ranks = ['0', '1', '0', '0' if final_action == 'aA' else '1']
    text, turns, native = 'SYS', [], []
    for index, (observation, action, rank) in enumerate(zip(observations, actions, ranks, strict=True)):
        text += observation
        start = len(text)
        scratch = ('remember aA=0; ' * 60) if index == 3 else ''
        text += scratch + '</think>\nAction: ' + action + '\n'
        turns.append(dict(start=start, end=len(text), stop_condition='action_line'))
        native.append(dict(round=index // 2 + 1, parse=dict(ok=True, coord=action),
            env_result=dict(applied=True, revealed_face=rank,
                            phase_after='flip_second' if index % 2 == 0 else 'flip_first')))
    return dict(tokens=list(map(ord, text)), turns=turns, prefix_length=3, window=256), dict(
        config=dict(rows=2, cols=2), turns=native)


def test_generated_notes_do_not_reset_direct_reveal_age():
    record, game = example()
    decisions = known_match_decisions(record, game, CharacterTokenizer())
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision['partners'].keys() == {'aA'}
    reveal = decision['query_position'] - decision['partners']['aA']
    assert decision['turn_start'] - reveal <= record['window']
    assert decision['direct_reveal_age'] > 2 * record['window']
    assert decision['age_bin'] == '>2W' and decision['correct']
    assert summarize(decisions)['>2W'] == dict(opportunities=1, correct=1, utilization=1.)


def test_wrong_partner_is_scored_from_actual_action():
    record, game = example('bB')
    decisions = known_match_decisions(record, game, CharacterTokenizer())
    assert len(decisions) == 1
    assert not decisions[0]['correct']
    assert summarize(decisions)['>2W']['utilization'] == 0.


def test_action_mapping_uses_original_sampled_token_boundaries():
    class NonCanonicalAssistantTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            assert 'Action:' not in text, 'Sampled text can have a different valid BPE segmentation'
            return super().__call__(text, **kwargs)

    record, game = example()
    decisions = known_match_decisions(record, game, NonCanonicalAssistantTokenizer())
    assert decisions[0]['correct']
