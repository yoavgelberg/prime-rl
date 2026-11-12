import re
from threading import Event, Thread

import pytest

from prime_rl.inference.vllm.ps_interactive import PersistentChat, TranscriptLimit


class ChatTokenizer:
    markers = ['<think>', '</think>', '<|im_start|>', '<|im_end|>', '<|endoftext|>']
    eos_token_id = 1004

    def encode(self, text, **kwargs):
        parts = re.split('(' + '|'.join(map(re.escape, self.markers)) + ')', text)
        return [token for part in parts for token in (
            [1000 + self.markers.index(part)] if part in self.markers else [ord(c) for c in part])]

    def decode(self, tokens, skip_special_tokens=False):
        return ''.join(self.markers[token - 1000] if token >= 1000 else chr(token)
            for token in tokens if not (skip_special_tokens and token >= 1002))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
        if add_generation_prompt:
            text += '<|im_start|>assistant\n<think>\n'
        return self.encode(text) if tokenize else text


def test_natural_formatting_has_only_sampled_assistant_tokens_and_resets_between_turns():
    tokenizer = ChatTokenizer()
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096,
                            timeout=3, thinking_token_limit=128, action_token_limit=16)
    messages = [dict(role='system', content='Rules'), dict(role='user', content='Board')]
    responses = []

    def play():
        response = client.chat(messages)
        responses.append(response)
        responses.append(client.chat([*messages, dict(role='assistant', content=response['content']),
                                      dict(role='user', content='Next board')]))
        client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    script = tokenizer.encode('brief.\n</think>\nAction: aA\n' + 'y</think>\nAction: bB\n')
    cursor = 0
    while cursor < len(script):
        if client.advance(script[cursor], -.3, -.5, entropy=.75).target != -100:
            cursor += 1
    worker.join(3)
    assert not worker.is_alive()
    record = client.record()
    assert [response['content'] for response in responses] == ['Action: aA', 'Action: bB']
    assert [turn['thinking_tokens'] for turn in record['turns']] == [7, 1]
    assert record['thinking_budget_forced_turns'] == 0
    assert sum(record['loss_mask']) == len(script) == record['entropy_tokens']
    assert set(record['roles']) <= {'prefix', 'template', 'environment_template', 'assistant'}
    for turn in record['turns']:
        assert record['tokens'][turn['start']:turn['end']] == turn['sampled_ids']
        assert all(record['loss_mask'][turn['start']:turn['end']])


@pytest.mark.parametrize('end_thinking', [False, True])
def test_budget_ends_overlong_replies_without_inserting_format_tokens(end_thinking):
    tokenizer = ChatTokenizer()
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096,
        timeout=3, thinking_token_limit=128, action_token_limit=16)
    responses = []

    def play():
        responses.append(client.chat([dict(role='system', content='Rules'), dict(role='user', content='Board')]))
        client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    script = ([1001] + [1001] * 16) if end_thinking else [ord('x')] * 128
    cursor = 0
    while cursor < len(script):
        if client.advance(script[cursor], -.3, -.5, entropy=.75).target != -100:
            cursor += 1
    worker.join(3)
    assert not worker.is_alive()
    record = client.record()
    turn = record['turns'][0]
    assert turn['stop_condition'] == ('action_token_limit' if end_thinking else 'thinking_token_limit')
    assert turn['thinking_tokens'] == (0 if end_thinking else 128)
    assert turn['action_tokens'] == (16 if end_thinking else 0)
    assert record['thinking_budget_forced_turns'] == 0
    assert sum(record['loss_mask']) == len(turn['sampled_ids']) == len(script)
    assert record['tokens'][turn['start']:turn['end']] == script
    assert all(record['loss_mask'][turn['start']:turn['end']])


@pytest.mark.parametrize("scratch_length", [16, 2304])
def test_native_multiturn_stream_preserves_tokens_and_only_samples_have_loss(scratch_length):
    tokenizer = ChatTokenizer()
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096, timeout=3)
    system, first, second = [dict(role=r, content=c) for r, c in (
        ('system', 'Game rules.'), ('user', 'All cards hidden.'), ('user', 'aA reveals rank 0.'))]
    messages = [system, first]
    responses = []

    def play():
        response = client.chat(messages)
        responses.append(response)
        responses.append(client.chat([*messages, dict(role='assistant', content=response['content']), second]))
        client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    prefix = client.initial_prompt()
    assert prefix == tokenizer.apply_chat_template([system], tokenize=True, add_generation_prompt=False)
    bodies = ['x' * scratch_length + '\n</think>\nAction: aA\n', 'Remember aA=0.\n</think>\nAction: bB\n']
    samples = tokenizer.encode(''.join(bodies))
    cursor, decisions = 0, []
    while cursor < len(samples):
        decision = client.advance(samples[cursor], -.3, -.5, entropy=.75)
        decisions.append(decision)
        if decision.role == 'assistant':
            cursor += 1
    worker.join(3)
    assert not worker.is_alive()
    assert [r['content'] for r in responses] == ['Action: aA', 'Action: bB']
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    expected += tokenizer.encode(bodies[0] + '<|im_end|>\n')
    expected += tokenizer.apply_chat_template([system, second], tokenize=True, add_generation_prompt=True)[len(prefix):]
    expected += tokenizer.encode(bodies[1])
    record = client.record()
    assert record['tokens'] == expected
    assert record['positions'] == list(range(len(expected)))
    assert record['tokens'][:len(prefix)] == prefix
    assert sum(record['loss_mask']) == len(samples)
    assert record['entropy_tokens'] == len(samples)
    assert record['mean_policy_entropy_at_temperature'] == .75
    assert len(record['turns']) == 2
    assert record['turns'][0]['start'] < len(prefix) + 1024
    assert all((lp is not None) == keep for lp, keep in zip(record['old_tempered_logps'], record['loss_mask']))
    assert [d.token for d in decisions if d.target != -100] == samples
    assert all(d.target == -100 for d in decisions if d.role != 'assistant')
    transport = client.advance(17, -.3, -.5)
    assert transport.target == -100 and transport.role == 'transport'
    assert client.record()['tokens'] == expected


def test_sampled_eos_is_retained_and_template_does_not_duplicate_im_end():
    tokenizer = ChatTokenizer()
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096, timeout=3)
    messages = [dict(role='system', content='Rules'), dict(role='user', content='Board')]
    responses = []

    def play():
        responses.append(client.chat(messages))
        client.chat([*messages, dict(role='assistant', content=''), dict(role='user', content='Invalid; retry.')])
        client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    cursor = 0
    script = [1003, *tokenizer.encode('</think>\nAction: aA\n')]
    while cursor < len(script):
        if client.advance(script[cursor], -.3, -.5).target != -100:
            cursor += 1
    worker.join(3)
    assert not worker.is_alive()
    record = client.record()
    assert record['turns'][0]['stop_condition'] == 'eos'
    assert responses[0]['content'] == ''
    boundary = record['turns'][0]['end']
    assert record['tokens'][boundary - 1] == 1003
    assert record['tokens'][boundary] == ord('\n')
    assert record['loss_mask'][boundary - 1] and not record['loss_mask'][boundary]


def test_context_limit_keeps_partial_thinking_losses_and_censors_native_runner():
    tokenizer = ChatTokenizer()
    messages = [dict(role='system', content='Rules'), dict(role='user', content='Board')]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=len(prompt) + 7, timeout=3)
    errors = []

    def play():
        try:
            client.chat(messages)
        except TranscriptLimit as error:
            errors.append(error)
        finally:
            client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    while not client.status()['done']:
        client.advance(ord('x'), -.3, -.5)
    worker.join(3)
    assert not worker.is_alive()
    assert len(errors) == 1
    record = client.record()
    assert len(record['tokens']) == len(prompt) + 7
    assert sum(record['loss_mask']) == 7
    assert record['turns'][0]['stop_condition'] == 'transcript_capacity'
    assert client.status()['resource_reason'] is not None


def test_engine_limit_between_completed_response_and_next_feedback_keeps_resource_censoring():
    tokenizer = ChatTokenizer()
    messages = [dict(role='system', content='Rules'), dict(role='user', content='Board')]
    script = tokenizer.encode('aA was revealed.\n</think>\nAction: aA\n')
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    client = PersistentChat(tokenizer, eos_token_ids=[1003,1004], transcript_cap=len(prompt) + len(script), timeout=3)
    ended, resume = Event(), Event()
    responses, errors = [], []

    def play():
        try:
            response = client.chat(messages)
            responses.append(response)
            ended.set()
            assert resume.wait(3)
            client.chat([*messages, dict(role='assistant', content=response['content']),
                         dict(role='user', content='Second pick')])
        except BaseException as error:
            errors.append(error)
        finally:
            client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    cursor = 0
    while cursor < len(script):
        if client.advance(script[cursor], -.3, -.5).target != -100:
            cursor += 1
    assert ended.wait(3)
    client.exhaust()
    resume.set()
    worker.join(3)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TranscriptLimit)
    record = client.record()
    assert len(record['tokens']) == client.transcript_cap
    assert len(record['turns']) == 1 and record['turns'][0]['stop_condition'] == 'action_line'
    assert sum(record['loss_mask']) == len(script)
    assert responses[0]['content'] == 'Action: aA'


def test_bad_native_feedback_fails_instead_of_mutating_prefix():
    tokenizer = ChatTokenizer()
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096)
    with pytest.raises(ValueError, match='initial message'):
        client.chat([dict(role='user', content='No system message')])



def test_vllm_sample_routing_preserves_forced_mask_and_skips_discarded_prefill():
    from types import SimpleNamespace
    import torch

    from prime_rl.inference.vllm.ps_interactive import TokenDecision, route_samples

    calls = []
    chats = {
        'sampled': SimpleNamespace(advance=lambda token, raw, tempered, **kwargs: (
            calls.append((token, raw, tempered, kwargs['entropy'])) or TokenDecision(token, token, 'assistant'))),
        'forced': SimpleNamespace(advance=lambda *args, **kwargs: TokenDecision(6, -100, 'environment_template')),
        'prefill': SimpleNamespace(advance=lambda *args, **kwargs: pytest.fail('Discarded partial prefill advanced game')),
    }
    logits = torch.tensor([[1., 2., -1., 3., 0., .5, 4.]]).repeat(3, 1)
    proposals = torch.tensor([[3], [4], [5]])
    emitted, labels = route_samples(['sampled-random', 'forced-random', 'prefill-random'],
        proposals, logits, chats, [False, False, True], temperature=.6)
    assert emitted[:, 0].tolist() == [3, 6, 5]
    assert labels[:, 0].tolist() == [3, -100, -100]
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(float(logits[0].log_softmax(-1)[3]))
    assert calls[0][2] == pytest.approx(float((logits[0] / .6).log_softmax(-1)[3]))

    assert calls[0][3] == pytest.approx(float(torch.distributions.Categorical(logits=logits[0] / .6).entropy()))


def test_forced_semantic_transition_excludes_addendum_from_128_token_budget_and_loss():
    tokenizer = ChatTokenizer()
    cue = '\n\nOkay, time to give the answer.</think>\n\n'
    client = PersistentChat(tokenizer, eos_token_ids=[1003, 1004], transcript_cap=4096,
        timeout=3, thinking_token_limit=128, thinking_budget_transition=cue)
    messages = [dict(role='system', content='Rules'), dict(role='user', content='Board')]
    responses = []

    def play():
        response = client.chat(messages)
        responses.append(response)
        responses.append(client.chat([*messages, dict(role='assistant', content=response['content']),
                                      dict(role='user', content='Next board')]))
        client.finish()

    worker = Thread(target=play, daemon=True)
    worker.start()
    client.initial_prompt()
    script = tokenizer.encode('x' * 128 + 'Action: aA\n' + 'y</think>\nAction: bB\n')
    cursor = 0
    while cursor < len(script):
        decision = client.advance(script[cursor], -.3, -.5, entropy=.75)
        if decision.target != -100:
            cursor += 1
        elif decision.role == 'thinking_budget':
            assert decision.target == -100
    worker.join(3)
    assert not worker.is_alive()
    record = client.record()
    first, second = record['turns']
    assert [response['content'] for response in responses] == ['Action: aA', 'Action: bB']
    assert [first['thinking_tokens'], second['thinking_tokens']] == [128, 1]
    assert first['thinking_budget_forced'] and not second['thinking_budget_forced']
    assert first['stop_condition'] == second['stop_condition'] == 'action_line'
    assert record['action_token_limit'] is None
    assert sum(record['loss_mask']) == record['entropy_tokens'] == len(script)
    forced = [i for i, role in enumerate(record['roles']) if role == 'thinking_budget']
    assert [record['tokens'][i] for i in forced] == tokenizer.encode(cue)
    assert all(not record['loss_mask'][i] and record['old_logps'][i] is None
               and record['old_tempered_logps'][i] is None for i in forced)
    assert record['positions'] == list(range(len(record['tokens'])))
    for turn in record['turns']:
        assert [record['tokens'][i] for i in turn['sampled_positions']] == turn['sampled_ids']
        assert all(record['loss_mask'][i] for i in turn['sampled_positions'])
    assert tokenizer.decode(record['tokens'][first['start']:first['end']]) == 'x' * 128 + cue + 'Action: aA\n'
