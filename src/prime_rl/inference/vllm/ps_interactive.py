"""Persistent native-game chat over one autoregressive vLLM request."""

from collections import deque
from dataclasses import dataclass
import re
from threading import Condition, Event
import time


class TranscriptLimit(BaseException):
    """Exit native runner Exception handlers while preserving resource censoring."""


@dataclass(frozen=True)
class TokenDecision:
    token: int
    target: int
    role: str


class PersistentChat:
    def __init__(self, tokenizer, *, eos_token_ids, transcript_cap, timeout=60., on_turn=None):
        self.tokenizer, self.eos = tokenizer, set(eos_token_ids)
        self.transcript_cap, self.timeout, self.on_turn = transcript_cap, timeout, on_turn
        self.condition = Condition()
        self.ready = Event()
        self.queue = deque()
        self.tokens, self.roles, self.old_logps, self.old_tempered_logps, self.turns = [], [], [], [], []
        self.prefix, self.system, self.close = None, None, None
        self.request, self.seen = None, 0
        self.sampled, self.sample_logps = [], []
        self.final_start, self.turn_start = None, 0
        self.done, self.error, self.resource_reason = False, None, None
        self.generated_tokens = 0
        self.entropy_sum, self.entropy_tokens = 0., 0
        self.started = time.perf_counter()
        thinking_end = tokenizer.encode('</think>', add_special_tokens=False)
        if len(thinking_end) != 1:
            raise ValueError('The native thinking delimiter must have one token')
        self.thinking_end = thinking_end[0]
        newline = tokenizer.encode('\n', add_special_tokens=False)
        if len(newline) != 1:
            raise ValueError('The action-line terminator must have one token')
        self.newline = newline[0]

    @staticmethod
    def _text(message):
        content = message['content']
        if isinstance(content, str):
            return content
        if any(part['type'] != 'text' for part in content):
            raise ValueError('Persistent textual games require text-only observations')
        return '\n'.join(part['text'] for part in content)

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Native game failed') from self.error

    def _capacity(self, count):
        if len(self.tokens) + count > self.transcript_cap:
            raise TranscriptLimit('Native context/transcript capacity reached')

    def _append(self, token, role, raw_logp=None, tempered_logp=None):
        self._capacity(1)
        self.tokens.append(token)
        self.roles.append(role)
        self.old_logps.append(raw_logp)
        self.old_tempered_logps.append(tempered_logp)
        if role == 'assistant':
            self.generated_tokens += 1

    def chat(self, messages):
        with self.condition:
            self._check()
            if self.done or self.request is not None:
                raise RuntimeError('Native chat calls must be serial within a game')
            new = [dict(role=m['role'], content=self._text(m)) for m in messages[self.seen:]]
            if self.prefix is None:
                if not new or new[0]['role'] != 'system':
                    raise ValueError('The initial message must be the immutable system prompt')
                self.system = new[0]
                prefix = self.tokenizer.apply_chat_template([self.system], tokenize=True, add_generation_prompt=False)
                if not 0 < len(prefix) <= 512:
                    raise ValueError('The unpadded system prefix must contain 1..512 tokens')
                self.prefix = tuple(prefix)
                for token in prefix:
                    self._append(token, 'prefix')
                marker = 'UNIQUE_CONTENT_MARKER_8037'
                closing = self.tokenizer.apply_chat_template([dict(role='assistant', content=marker)],
                    tokenize=False, add_generation_prompt=False)
                self.close = self.tokenizer.encode(closing.split(marker, 1)[1], add_special_tokens=False)
                leading = []
            else:
                if not new or any(m['role'] != 'user' for m in new):
                    raise ValueError('Only new native feedback may be appended between responses')
                new = [self.system, *new]
                leading = self.close[1:] if self.tokens[-1] == self.close[0] else self.close
            rendered = self.tokenizer.apply_chat_template(new, tokenize=True, add_generation_prompt=True)
            if tuple(rendered[:len(self.prefix)]) != self.prefix:
                raise ValueError('The system prefix changed within a game')
            fragment = rendered[len(self.prefix):]
            self._capacity(len(leading) + len(fragment))
            self.queue.extend((token, 'template') for token in leading)
            self.queue.extend((token, 'environment_template') for token in fragment)
            request = dict(event=Event(), messages=len(messages), response=None, error=None)
            self.request = request
            self.sampled, self.sample_logps, self.final_start = [], [], None
            self.turn_start = len(self.tokens) + len(self.queue)
            self.turn_started = time.perf_counter()
            self.ready.set()
            self.condition.notify_all()
        request['event'].wait()
        if request['error'] is not None:
            raise request['error']
        return request['response']

    def initial_prompt(self):
        if not self.ready.wait(self.timeout):
            raise TimeoutError('Native game did not produce its initial prompt')
        with self.condition:
            self._check()
            if self.prefix is None:
                raise RuntimeError('Native game ended without an initial prompt')
            return list(self.prefix)

    def _end_turn(self, stop):
        text = self.tokenizer.decode(self.sampled, skip_special_tokens=True)
        reasoning, final = text.rsplit('</think>', 1) if '</think>' in text else (text, '')
        turn = dict(start=self.turn_start, end=len(self.tokens), sampled_ids=self.sampled.copy(),
            old_logps=self.sample_logps.copy(), stop_condition=stop, reasoning=reasoning,
            content=final.strip(), seconds=time.perf_counter() - self.turn_started)
        self.turns.append(turn)
        if self.on_turn is not None:
            self.on_turn(turn)
        request, self.request = self.request, None
        self.seen = request['messages'] + 1
        request['response'] = dict(content=final.strip(), reasoning=reasoning)
        request['event'].set()

    def _exhaust(self, error):
        self.resource_reason = str(error)
        self.queue.clear()
        if self.request is not None:
            if self.sampled:
                self.turns.append(dict(start=self.turn_start, end=len(self.tokens),
                    sampled_ids=self.sampled.copy(), old_logps=self.sample_logps.copy(),
                    stop_condition='transcript_capacity', reasoning=self.tokenizer.decode(self.sampled,
                    skip_special_tokens=False), content=''))
            self.request['error'] = error
            self.request['event'].set()
            self.request = None
        self.done = True
        self.condition.notify_all()

    def advance(self, proposed, raw_logp, tempered_logp, *, entropy=None):
        """Forced feedback changes tokens, while only actual samples receive labels."""
        with self.condition:
            if not self.condition.wait_for(lambda: self.done or self.queue or self.request is not None,
                                           timeout=self.timeout):
                raise TimeoutError('Native environment did not provide feedback after a response')
            self._check()
            if self.done:
                return TokenDecision(self.tokenizer.eos_token_id, -100, 'transport')
            if len(self.tokens) == self.transcript_cap:
                self._exhaust(TranscriptLimit('Native context/transcript capacity reached'))
                return TokenDecision(self.tokenizer.eos_token_id, -100, 'transport')
            if self.queue:
                token, role = self.queue.popleft()
                self._append(token, role)
                return TokenDecision(token, -100, role)
            self._append(proposed, 'assistant', raw_logp, tempered_logp)
            if entropy is not None:
                self.entropy_sum += entropy
                self.entropy_tokens += 1
            self.sampled.append(proposed)
            self.sample_logps.append(raw_logp)
            if proposed == self.thinking_end:
                self.final_start = len(self.sampled)
            stop = 'eos' if proposed in self.eos else None
            if stop is None and proposed == self.newline and self.final_start is not None:
                final = self.tokenizer.decode(self.sampled[self.final_start:], skip_special_tokens=True)
                if re.search(r'(?:^|\n)Action:\s*[a-z][A-Z][ \t]*\n', final):
                    stop = 'action_line'
            if stop is not None:
                self._end_turn(stop)
            return TokenDecision(proposed, proposed, 'assistant')

    def finish(self, error=None):
        with self.condition:
            self.error, self.done = error, True
            if self.request is not None:
                self.request['error'] = error or RuntimeError('Engine ended before the native response finished')
                self.request['event'].set()
                self.request = None
            self.ready.set()
            self.condition.notify_all()

    def exhaust(self):
        with self.condition:
            if not self.done:
                self._exhaust(TranscriptLimit('Native context/transcript capacity reached'))

    def status(self):
        with self.condition:
            self._check()
            return dict(done=self.done, raw_transcript_tokens=len(self.tokens),
                generated_tokens=self.generated_tokens, turns=len(self.turns), resource_reason=self.resource_reason)

    def record(self):
        with self.condition:
            self._check()
            if not self.done:
                raise RuntimeError('A complete token record requires a terminal native game')
            return dict(tokens=self.tokens.copy(), roles=self.roles.copy(), positions=list(range(len(self.tokens))),
                loss_mask=[role == 'assistant' for role in self.roles], old_logps=self.old_logps.copy(),
                old_tempered_logps=self.old_tempered_logps.copy(), turns=self.turns.copy(),
                prefix_length=len(self.prefix), thinking_retained=True, entropy_tokens=self.entropy_tokens,
                mean_policy_entropy_at_temperature=self.entropy_sum / self.entropy_tokens if self.entropy_tokens else None)


def route_samples(request_ids, sampled_token_ids, logits, chats, discarded, *, temperature):
    import torch

    if sampled_token_ids.shape != (len(request_ids), 1):
        raise ValueError('Interactive games require one autoregressive token per request')
    samples = sampled_token_ids[:, 0].long()
    raw = logits.float()
    selected = raw.gather(1, samples[:, None])[:, 0]
    raw_logps = selected - raw.logsumexp(-1)
    scaled = raw / temperature
    tempered = scaled.gather(1, samples[:, None])[:, 0] - scaled.logsumexp(-1)
    probabilities = scaled.softmax(-1)
    entropy = scaled.logsumexp(-1) - torch.where(probabilities > 0, probabilities * scaled, 0.).sum(-1)
    values = torch.stack((samples.float(), raw_logps, tempered, entropy), dim=-1).tolist()
    emitted, targets = [], []
    for request, (token, raw_logp, tempered_logp, entropy), discard in zip(request_ids, values, discarded, strict=True):
        matches = [key for key in chats if request == key or request.startswith(key + '-')]
        if len(matches) != 1:
            raise ValueError(f'Request {request} must identify exactly one native game')
        if discard:
            emitted.append(int(token))
            targets.append(-100)
            continue
        decision = chats[matches[0]].advance(int(token), raw_logp, tempered_logp, entropy=entropy)
        emitted.append(decision.token)
        targets.append(decision.target)
    return (torch.tensor(emitted, device=sampled_token_ids.device, dtype=sampled_token_ids.dtype)[:, None],
            torch.tensor(targets, device=sampled_token_ids.device, dtype=torch.long)[:, None])


_cohort = None
_sampling_installed = False


def install_sampling_hook():
    global _sampling_installed
    if _sampling_installed:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_sample = GPUModelRunner._sample

    def sample(runner, logits, spec_decode_metadata):
        raw = logits.clone() if _cohort is not None else None
        result = original_sample(runner, logits, spec_decode_metadata)
        if _cohort is not None:
            if spec_decode_metadata is not None or result.logprobs_tensors is not None:
                raise ValueError('Interactive games use ordinary sampling and their own token likelihood records')
            tokens, labels = route_samples(runner.input_batch.req_ids, result.sampled_token_ids, raw,
                _cohort.chats, runner.discard_request_mask.np[:runner.input_batch.num_reqs],
                temperature=_cohort.temperature)
            result.sampled_token_ids.copy_(tokens)
            result.ps_loss_targets = labels
        return result

    GPUModelRunner._sample = sample
    _sampling_installed = True


class InteractiveWorkerMixin:
    def ps_game_prepare(self, factory, options):
        import importlib

        global _cohort
        if _cohort is not None:
            raise RuntimeError('Finish and release the previous native-game cohort before preparing another')
        module, name = factory.rsplit('.', 1)
        build = getattr(importlib.import_module(module), name)
        _cohort = build(**options, rank=self.vllm_config.parallel_config.data_parallel_rank)
        return _cohort.initial_requests()

    def ps_game_status(self):
        if _cohort is None:
            raise RuntimeError('No native-game cohort is active')
        return _cohort.status()

    def ps_game_engine_finished(self, request_ids):
        if _cohort is None:
            raise RuntimeError('No native-game cohort is active')
        for request in request_ids:
            if request in _cohort.chats:
                _cohort.chats[request].exhaust()

    def ps_game_abort(self, reason):
        if _cohort is not None:
            for chat in _cohort.chats.values():
                chat.finish(error=RuntimeError(reason))

    def ps_game_finish(self):
        if _cohort is None:
            raise RuntimeError('No native-game cohort is active')
        return _cohort.finish()

    def ps_game_release(self):
        global _cohort
        if _cohort is None or not all(row['done'] for row in _cohort.status().values()):
            raise RuntimeError('Native games must be terminal before releasing their controller')
        _cohort.finish()
        _cohort = None
