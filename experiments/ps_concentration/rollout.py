from collections import deque
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .config import system_prompt
from .environment import completed_action, make_env, parse_action, serialize
from .policy import StreamingQwen


class ChatCodec:
    def __init__(self, tokenizer, cards=52, limit=104, prompted=False):
        self.tokenizer = tokenizer
        self.prefix = tokenizer.apply_chat_template(
            [dict(role='system', content=system_prompt(cards, limit, prompted))],
            tokenize=True, add_generation_prompt=False)
        assert len(self.prefix) <= 512
        # Derive the closing delimiter from the native rendered assistant message.
        marker = 'UNIQUE_CONTENT_MARKER_8037'
        rendered = tokenizer.apply_chat_template([dict(role='assistant', content=marker)], tokenize=False,
                                                 add_generation_prompt=False)
        self.close = tokenizer.encode(rendered.split(marker, 1)[1], add_special_tokens=False)
        self.eos = tokenizer.eos_token_id

    def observation(self, text):
        return self.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                                                  add_generation_prompt=True)


def save_episodes(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'at') as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')


def sample(logits, rngs, temperature, top_p):
    raw = logits.log_softmax(-1)
    if temperature == 0:
        tokens = logits.argmax(-1)
        old = raw.gather(1, tokens[:, None])[:, 0]
        return tokens.cpu().tolist(), old.cpu().tolist(), old.cpu().tolist()
    logps = (logits / temperature).log_softmax(-1)
    sorted_probs, indices = logps.exp().sort(-1, descending=True)
    cdf = sorted_probs.cumsum(-1)
    sorted_probs.masked_fill_(cdf - sorted_probs >= top_p, 0.)
    sorted_probs /= sorted_probs.sum(-1, keepdim=True)
    uniforms = torch.tensor([r.random() for r in rngs], device=logits.device, dtype=logits.dtype)
    cumulative = sorted_probs.cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:]
    chosen = (cumulative < uniforms[:, None]).sum(-1).clamp_max(logits.shape[-1] - 1)
    tokens = indices.gather(1, chosen[:, None])[:, 0]
    return (tokens.cpu().tolist(), raw.gather(1, tokens[:, None])[:, 0].cpu().tolist(),
            logps.gather(1, tokens[:, None])[:, 0].cpu().tolist())


@torch.inference_mode()
@torch.autocast('cuda', dtype=torch.bfloat16)
def run_batch(base, tokenizer, seeds, generation_seeds, config, *, greedy=False, prompted=False, before_sample=None):
    codec = ChatCodec(tokenizer, config.cards, config.limit, prompted)
    native_eos = base.generation_config.eos_token_id
    eos_ids = set(native_eos if isinstance(native_eos, list) else [native_eos])
    stream = StreamingQwen(base, len(seeds), len(codec.prefix), config.window, config.attention,
                           config.transcript_cap)
    lanes = []
    for seed, rng in zip(seeds, generation_seeds, strict=True):
        env = make_env(config.cards == 104)
        obs, _ = env.reset(seed=int(seed))
        record = dict(board_seed=int(seed), generation_seed=int(rng), attention=config.attention, cards=config.cards,
                      queued_prefill_tokens=int(os.environ.get('PS_CONCENTRATION_PREFILL_TOKENS','0')),
                      full_decode_splits=int(os.environ.get('PS_CONCENTRATION_FULL_SPLITS','1')),
                      gpu_model=torch.cuda.get_device_name(base.device),source_sha256=os.environ.get('PS_CONCENTRATION_SOURCE_SHA',''),
                      window=config.window, prefix_length=len(codec.prefix), prompted=prompted,
                      tokens=[], loss_mask=[], roles=[], old_logps=[], old_tempered_logps=[],
                      turns=[], known_match=[], native_return=0., format_errors=0, invalid_actions=0,
                      completed=False, matched_fraction=0., flips=0)
        lane = dict(env=env, record=record, queue=deque(), sampled=[], done=False,
                    rng=np.random.default_rng(rng), reveals={}, known={})
        def append(tokens, role, sample_after=False, lane=lane):
            start = len(lane['record']['tokens'])
            lane['record']['tokens'].extend(tokens)
            lane['record']['roles'].extend([role] * len(tokens))
            lane['record']['loss_mask'].extend([role == 'assistant'] * len(tokens))
            lane['record']['old_logps'].extend([None] * len(tokens))
            lane['record']['old_tempered_logps'].extend([None] * len(tokens))
            lane['queue'].extend((token, sample_after and j == len(tokens)-1, start+j)
                                  for j, token in enumerate(tokens))
        lane['append'] = append
        append(codec.prefix, 'prefix')
        append(codec.observation(serialize(obs, env.facedown_card, limit=config.limit)), 'environment', True)
        lanes.append(lane)
    started = time.perf_counter()
    warm_tokens = min(len(lane['queue']) for lane in lanes) - 1
    first = [list(lane['queue'])[:warm_tokens] for lane in lanes]
    stream.prefill(torch.tensor([[token for token, _, _ in row] for row in first], device=base.device))
    for lane in lanes:
        for _ in range(warm_tokens):
            lane['queue'].popleft()
    ticks = warm_tokens
    chunk = int(os.environ.get('PS_CONCENTRATION_PREFILL_TOKENS','0'))
    while not all(lane['done'] for lane in lanes):
        if chunk and (config.attention=='full' or stream.consumed>=stream.capacity):
            waiting = min(len(lane['queue']) for lane in lanes if not lane['done'])-1
            if waiting >= chunk:
                from .chunk_prefill import append_tokens
                block=[]
                for lane in lanes:
                    queued=[lane['queue'].popleft() for _ in range(chunk)] if not lane['done'] else []
                    assert all(not sample_after and position==ticks+i for i,(_,sample_after,position) in enumerate(queued))
                    block.append([item[0] for item in queued] if queued else [codec.eos]*chunk)
                append_tokens(stream,torch.tensor(block,device=base.device))
                ticks += chunk
                continue
        inputs, requesting = [], []
        for index, lane in enumerate(lanes):
            if lane['done']:
                inputs.append(codec.eos)
                continue
            token, sample_after, position = lane['queue'].popleft()
            assert position == ticks
            inputs.append(token)
            if sample_after:
                requesting.append(index)
        logits = stream.step(torch.tensor(inputs, device=base.device))
        ticks += 1
        if not requesting:
            continue
        if before_sample is not None:
            for index in requesting:
                if not lanes[index]['sampled']:
                    before_sample(lanes[index], stream, index, logits[index])
        tokens, logps, tempered = sample(logits[requesting], [lanes[i]['rng'] for i in requesting],
                                         0. if greedy else config.temperature, config.top_p)
        for index, token, logp, temp_logp in zip(requesting, tokens, logps, tempered, strict=True):
            lane = lanes[index]
            rec, env = lane['record'], lane['env']
            if not lane['sampled']:
                lane['turn_start'] = len(rec['tokens'])
            lane['sampled'].append(token)
            text = tokenizer.decode(lane['sampled'], skip_special_tokens=True)
            stop = ('eos' if token in eos_ids else 'action_line' if completed_action(text)
                    else 'length' if len(lane['sampled']) == config.max_new_tokens else None)
            position = len(rec['tokens'])
            lane['append']([token], 'assistant', stop is None)
            rec['old_logps'][position], rec['old_tempered_logps'][position] = logp, temp_logp
            if stop is None:
                continue
            action, error = parse_action(text, config.cards, stop_condition=stop)
            first = env.deck['in_play_idx'][0] if env.deck['in_play_idx'] else None
            if first is not None:
                rank = int(env.obs[first])
                partners = {p: position - lane['reveals'][p] for p, r in lane['known'].items()
                            if p != first and p not in env.deck['face_up_idx'] and r == rank}
                if partners:
                    rec['known_match'].append(dict(step=rec['flips'], action=action, partners=partners,
                                                   correct=action in partners, decision_position=position,
                                                   target_rank=rank, first=first))
            invalid = action in env.deck['face_up_idx']
            obs, reward, terminated, truncated, _ = env.step(action)
            rec['flips'] += 1
            rec['native_return'] += float(reward)
            rec['format_errors'] += int(error)
            rec['invalid_actions'] += int(invalid)
            turn = dict(step=rec['flips'], text=text, start=lane['turn_start'], end=position+1,
                        action=action, format_error=error, stop_condition=stop, reward=float(reward),
                        stop_token_id=token if token in eos_ids else None,
                        terminated=bool(terminated), truncated=bool(truncated), observation=obs.tolist(),
                        next_pick='second' if reward == 0 else 'first')
            rec['turns'].append(turn)
            if index == 0 and rec['flips'] % 16 == 0:
                print(json.dumps(dict(event='rollout_progress', flips=rec['flips'], raw_ticks=ticks,
                                      seconds=time.perf_counter()-started, native_return=rec['native_return'])), flush=True)
            lane['sampled'] = []
            rec['completed'] = bool(terminated)
            rec['matched_fraction'] = len(env.deck['face_up_idx']) / config.cards
            lane['done'] = bool(terminated or truncated)
            if lane['done']:
                continue
            closing = codec.close[1:] if token == codec.close[0] else codec.close
            lane['append'](closing, 'template')
            observation = codec.observation(serialize(obs, env.facedown_card, rec['flips'], config.limit,
                                                       action, reward))
            turn['observation_start'] = len(rec['tokens'])
            lane['append'](observation, 'environment', True)
            turn['observation_end'] = len(rec['tokens'])
            for p, rank in enumerate(obs):
                if rank != env.facedown_card:
                    lane['known'][p] = int(rank)
                    lane['reveals'][p] = len(rec['tokens']) - 1
            assert ('second' if reward == 0 else 'first') == ('second' if env.deck['in_play_idx'] else 'first')
            if len(rec['tokens']) > config.transcript_cap:
                raise RuntimeError('Serialized episode exceeded the common audited transcript cap')
    elapsed = time.perf_counter() - started
    records = []
    for lane in lanes:
        rec = lane['record']
        rec['native_return']=math.fsum(turn['reward'] for turn in rec['turns'])
        rec.update(generated_tokens=sum(rec['loss_mask']), raw_transcript_tokens=len(rec['tokens']),
                   positions=list(range(len(rec['tokens']))), inference_seconds=elapsed / len(lanes),
                   peak_kv_bytes=sum(k.numel()*k.element_size()*2 for k in stream.keys)/len(lanes),
                   active_kv_bytes=stream.cache_bytes / len(lanes),
                   first_eviction=rec['prefix_length'] + config.window if config.attention == 'ps' else None)
        assert len(rec['tokens']) == len(rec['loss_mask']) == len(rec['old_logps'])
        records.append(rec)
    print(json.dumps(dict(event='rollout_batch', attention=config.attention, window=config.window,
                          episodes=len(records), seconds=elapsed, tokens=sum(x['generated_tokens'] for x in records),
                          mean_return=float(np.mean([x['native_return'] for x in records])),
                          matched=float(np.mean([x['matched_fraction'] for x in records])))), flush=True)
    return records
