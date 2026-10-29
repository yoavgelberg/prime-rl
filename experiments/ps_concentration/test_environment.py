import numpy as np
import pytest

from .environment import completed_action, deserialize, make_env, parse_action, serialize


@pytest.mark.parametrize('text', ['FLIP -1', 'FLIP 52', 'FLIP x\nFLIP 2', 'FLIP 2\nFLIP 3',
                                 'FLIP 2 trailing', 'FLIP 2\ntrailing', '', '17'])
def test_malformed_executes_zero(text):
    assert parse_action(text) == (0, True)


def test_partial_integer_requires_line_completion_or_eos():
    assert not completed_action('scratch\nFLIP 1')
    assert completed_action('scratch\nFLIP 17\n')
    assert parse_action('scratch\nFLIP 17') == (17, False)
    assert parse_action('scratch\nFLIP 17',stop_condition='length') == (0,True)
    assert parse_action('scratch\nFLIP 17\n',stop_condition='length') == (17,False)


def test_native_observations_roundtrip_and_forgetting():
    env = make_env()
    observation, _ = env.reset(seed=19)
    first = 0
    second = next(i for i in range(1, 52) if env.state[i] != env.state[first])
    for step, action in enumerate([first, second, 51], 1):
        observation, reward, _, _, _ = env.step(action)
        text = serialize(observation, env.facedown_card, step, last_action=action, reward=reward)
        assert np.array_equal(deserialize(text, 52, env.facedown_card), observation)
        if step == 2:
            assert observation[first] != env.facedown_card and observation[second] != env.facedown_card
        if step == 3:
            assert observation[first] == env.facedown_card and observation[second] == env.facedown_card
            keys = [item.split(':')[0] for item in text.split('visible={')[1].removesuffix('}').split(',')]
            assert str(first) not in keys and str(second) not in keys


def test_system_prefix_does_not_prescribe_memory():
    from .config import system_prompt
    prompt = system_prompt()
    assert 'You may write scratch work before your action.' in prompt
    assert 'updated compact list' not in prompt and 'remember' not in prompt
    assert 'updated compact list' in system_prompt(prompted=True)
