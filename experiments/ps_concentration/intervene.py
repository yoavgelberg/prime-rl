import argparse
import copy
import gzip
import json
from pathlib import Path
import re

import numpy as np
import torch

from .config import Config, SPLITS
from .control import handoff_requested
from .environment import completed_action, parse_action
from .policy import StreamingQwen, load_model
from .rollout import run_batch, sample, save_episodes


ASSOCIATION = re.compile(r'(?<![\w])(?:position\s+|pos\s+|card\s+)?(\d{1,3})\s*'
                         r'(?::|=|->|→)\s*(?:rank\s*)?(\d{1,2})(?!\d)', re.IGNORECASE)


def associations(record, tokenizer, suffix_start):
    result = []
    for turn in record['turns']:
        lo, hi = turn['start'], turn['end']
        if hi <= suffix_start:
            continue
        ids = record['tokens'][lo:hi]
        text = tokenizer.decode(ids)
        action = re.search(r'^FLIP\b', text, re.MULTILINE)
        scratch = text[:action.start()] if action else text
        boundaries = {len(tokenizer.decode(ids[:i])): i for i in range(len(ids)+1)}
        for match in ASSOCIATION.finditer(scratch):
            if re.search(r'\b(?:rank|r)\s*[=:]?\s*$',scratch[max(0,match.start()-20):match.start()],re.IGNORECASE):
                continue
            if match.start(1) not in boundaries or match.end(1) not in boundaries:
                continue
            start = lo + boundaries[match.start(1)]
            end = lo + boundaries[match.end(1)]
            position, rank = int(match[1]), int(match[2])
            if start < suffix_start or not 0 <= position < 52 or not 0 <= rank < 13:
                continue
            assert all(role == 'assistant' for role in record['roles'][start:end])
            result.append(dict(position=position, rank=rank, start=start, end=end,
                               text=match[0], recognition='explicit position:rank / position=rank / position->rank'))
    return result


def make_edits(items, p, q, target, state, unmatched, tokenizer, record):
    edits = []
    for item in items:
        if item['position'] == p and item['rank'] == target:
            replacement = q
        elif item['position'] == q and item['rank'] == int(state[q]):
            replacement = p
        else:
            continue
        old = record['tokens'][item['start']:item['end']]
        new = tokenizer.encode(str(replacement), add_special_tokens=False)
        if len(old) != len(new):
            return None
        edits.append(dict(start=item['start'], end=item['end'], old=old, new=new))
    irrelevant = [item for item in items if item['position'] not in (p, q) and item['position'] in unmatched
                  and item['rank'] != target]
    sham = []
    for item in irrelevant:
        old = record['tokens'][item['start']:item['end']]
        replacement = next((r for r in unmatched if r != item['position'] and int(state[r]) != target
                            and int(state[r]) != item['rank']
                            and len(tokenizer.encode(str(r), add_special_tokens=False)) == len(old)), None)
        if replacement is not None:
            sham.append(dict(start=item['start'], end=item['end'], old=old,
                             new=tokenizer.encode(str(replacement), add_special_tokens=False)))
        if len(sham) == len(edits):
            break
    if len(sham) != len(edits) or sum(len(e['new']) for e in sham) != sum(len(e['new']) for e in edits):
        return None
    return edits, sham


def clone_stream(original, row, copies):
    clone = StreamingQwen(original.base, copies, original.prefix, original.window, original.attention,
                          original.capacity)
    for target, source in zip(clone.keys, original.keys, strict=True):
        target.copy_(source[row:row+1].expand_as(target))
    for target, source in zip(clone.values, original.values, strict=True):
        target.copy_(source[row:row+1].expand_as(target))
    clone.position, clone.consumed = original.position, original.consumed
    return clone


@torch.inference_mode()
@torch.autocast('cuda', dtype=torch.bfloat16)
def continue_turn(stream, logits, tokenizer, samples, rng_seed):
    native_eos = stream.base.generation_config.eos_token_id
    eos_ids = set(native_eos if isinstance(native_eos, list) else [native_eos])
    rngs = [np.random.default_rng(rng_seed+i) for i in range(samples)]
    responses = [[] for _ in range(samples+1)]
    done = [False] * (samples+1)
    for _ in range(256):
        greedy = int(logits[0].argmax())
        sampled, _, _ = sample(logits[1:], rngs, .8, .95) if samples else ([], [], [])
        tokens = [greedy, *sampled]
        for i, token in enumerate(tokens):
            if done[i]:
                continue
            responses[i].append(token)
            text = tokenizer.decode(responses[i], skip_special_tokens=True)
            done[i] = token in eos_ids or completed_action(text)
        if all(done):
            break
        logits = stream.step(torch.tensor(tokens, device=stream.base.device))
    results = []
    for response in responses:
        text = tokenizer.decode(response, skip_special_tokens=True)
        stop='eos' if response[-1] in eos_ids else 'action_line' if completed_action(text) else 'length'
        action, error = parse_action(text,stop_condition=stop)
        results.append(dict(action=action, format_error=error, tokens=response, text=text))
    return results


class Collector:
    def __init__(self, base, tokenizer, config, output, samples=32, maximum=128):
        self.base, self.tokenizer, self.config, self.output = base, tokenizer, config, output
        self.samples, self.maximum = samples, maximum
        self.states = []
        self.attempted = set()

    def __call__(self, lane, stream, row, logits):
        if len(self.states) >= self.maximum:
            return
        env, record = lane['env'], lane['record']
        key = record['board_seed'], record['flips']
        if key in self.attempted or len(env.deck['in_play_idx']) != 1:
            return
        self.attempted.add(key)
        first = env.deck['in_play_idx'][0]
        target = int(env.obs[first])
        start = len(record['tokens'])
        suffix_start = max(stream.prefix, start-stream.window)
        unmatched = [i for i in range(52) if i not in env.deck['face_up_idx'] and i != first]
        partners = [p for p in unmatched if lane['known'].get(p) == target
                    and start - lane['reveals'][p] > 2*stream.window]
        if not partners:
            return
        notes = associations(record, self.tokenizer, suffix_start)
        p = next((p for p in partners if any(a['position'] == p and a['rank'] == target for a in notes)), None)
        if p is None:
            return
        # A direct reveal older than 2W cannot be present in the retained W-token suffix.
        assert lane['reveals'][p] < suffix_start
        q = next((q for q in unmatched if int(env.state[q]) != target
                  and len(self.tokenizer.encode(str(q), add_special_tokens=False))
                  == len(self.tokenizer.encode(str(p), add_special_tokens=False))), None)
        state = dict(board_seed=key[0], step=key[1], p=p, q=q, target_rank=target,
                     decision_position=start, suffix_start=suffix_start, last_direct_reveal=lane['reveals'][p],
                     qualifying_associations=[n for n in notes if n['position'] == p and n['rank'] == target],
                     recognition='conservative explicit position-rank regex, no LLM judge')
        self.states.append(state)
        edit_pair = make_edits(notes, p, q, target, env.state, unmatched, self.tokenizer, record) if q is not None else None
        if edit_pair is None:
            state['untestable_reason'] = 'No equal-token-count irrelevant sham available'
            save_episodes(self.output/'states.jsonl', [state])
            return
        edits, sham = edit_pair
        state.update(edits=edits, sham_edits=sham)
        indices = list(range(stream.prefix)) + list(range(suffix_start, start))
        original_ids = [record['tokens'][i] for i in indices]
        state.update(retained_positions=indices, retained_token_ids=original_ids)
        ordinary = clone_stream(stream, row, self.samples+1)
        state['streaming'] = continue_turn(ordinary, logits[None].expand(self.samples+1, -1).clone(),
                                           self.tokenizer, self.samples, key[0]*104+key[1])
        del ordinary
        for name, branch_edits in [('unedited', []), ('swap', edits), ('sham', sham)]:
            edited = list(record['tokens'][:start])
            for edit in branch_edits:
                assert edited[edit['start']:edit['end']] == edit['old']
                assert len(edit['old']) == len(edit['new'])
                assert all(r == 'assistant' for r in record['roles'][edit['start']:edit['end']])
                edited[edit['start']:edit['end']] = edit['new']
            ids = [edited[i] for i in indices]
            assert len(ids) == len(original_ids) and ids[:stream.prefix] == original_ids[:stream.prefix]
            cold = StreamingQwen(self.base, self.samples+1, stream.prefix, stream.window, 'ps', self.config.transcript_cap)
            cold_logits = cold.prefill(torch.tensor([ids]*(self.samples+1), device=self.base.device),
                                       torch.tensor(indices, device=self.base.device))
            state[name] = continue_turn(cold, cold_logits, self.tokenizer, self.samples, key[0]*104+key[1])
            del cold
        game = copy.deepcopy(env)
        correct = [i for i in unmatched if int(game.state[i]) == target]
        state['correct_partners'] = correct
        for branch in ('streaming', 'unedited', 'swap', 'sham'):
            for outcome in state[branch]:
                outcome['correct'] = outcome['action'] in correct
                outcome['follows_false'] = outcome['action'] == q
                clone = copy.deepcopy(game)
                outcome['native_reward'] = float(clone.step(outcome['action'])[1])
        save_episodes(self.output/'states.jsonl', [state])
        print(json.dumps(dict(event='intervention', qualified=len(self.states), seed=key[0], step=key[1], p=p, q=q)), flush=True)


def summarize_states(states):
    tested = [s for s in states if 'unedited' in s]
    rows = {}
    for branch in ('streaming', 'unedited', 'swap', 'sham'):
        draws = [o for s in tested for o in s[branch][1:]]
        conditional = [o for s in tested if s['unedited'][0]['action'] == s['p'] for o in s[branch][1:]]
        rows[branch] = dict(tested_states=len(tested), greedy_correct_fraction=float(np.mean([s[branch][0]['correct'] for s in tested])) if tested else None,
                            stochastic_draws=len(draws), correct_probability=float(np.mean([o['correct'] for o in draws])) if draws else None,
                            false_association_probability=float(np.mean([o['follows_false'] for o in draws])) if draws else None,
                            conditional_draws=len(conditional), conditional_correct_probability=float(np.mean([o['correct'] for o in conditional])) if conditional else None)
    return dict(qualified_states=len(states), tested_states=len(tested), branches=rows,
                textual_memory_conclusion='INSUFFICIENT_STATES' if len(tested) < 32 else 'Inspect swap-versus-sham paired effects; qualification count alone is not causal evidence',
                recognition='explicit position:rank / position=rank / position->rank only; irregular notes require manual annotation')


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--window', type=int, default=1024)
    parser.add_argument('--output')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--maximum', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output or f'{args.checkpoint}/interventions')
    output.mkdir(parents=True, exist_ok=True)
    metadata=dict(checkpoint=args.checkpoint,window=args.window,samples=args.samples,maximum=args.maximum)
    metadata_path=output/'config.json'
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text())==metadata
    else:
        metadata_path.write_text(json.dumps(metadata,indent=2))
    base, tokenizer = load_model(args.checkpoint), AutoTokenizer.from_pretrained(args.checkpoint)
    from .audit import token_audit
    config = Config(model=args.checkpoint, window=args.window, transcript_cap=token_audit(tokenizer)['52']['required_cap'])
    collector = Collector(base, tokenizer, config, output, args.samples, args.maximum)
    states_path=output/'states.jsonl'
    if states_path.exists():
        collector.states=[json.loads(line) for line in states_path.read_text().splitlines() if line.strip()]
        collector.attempted={(s['board_seed'],s['step']) for s in collector.states}
    completed=set()
    collection=output/'collection.jsonl.gz'
    if collection.exists():
        with gzip.open(collection,'rt') as handle:
            for line in handle:
                completed.add(json.loads(line)['board_seed'])
    for index, seed in enumerate(SPLITS['intervention']):
        if len(collector.states)>=args.maximum and len(completed)>=20:
            break
        if seed in completed:
            continue
        records = run_batch(base, tokenizer, [seed], [seed], config, greedy=True, before_sample=collector)
        save_episodes(output/'collection.jsonl.gz', records)
        completed.add(seed)
        (output/'summary.json').write_text(json.dumps(summarize_states(collector.states), indent=2))
        if handoff_requested():
            return
        if len(collector.states) >= args.maximum and len(completed)>=20:
            break
    print(json.dumps(summarize_states(collector.states), indent=2), flush=True)


if __name__ == '__main__':
    main()
