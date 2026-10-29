"""Full-context base competence on native POPGym state-tracking tasks."""
import argparse
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import popgym.envs.count_recall as count_recall
import torch

from .rng_screen import LocalPolicy


def prompt(env, variant, budget=None):
    letters = ', '.join(chr(65+i) for i in range(env.num_distinct_cards))
    text = f"""You are playing Count Recall with card types {letters}.
This is ONE continuous game across all user messages. Every message adds exactly one new card. For the queried type, answer its TOTAL number of appearances since the game began, including the new card. Queries and your own words are not new cards. Counts start at zero; do not reset between turns.
Reason about the current count before answering. Use this format every turn:
Thought: <your reasoning or scratch work>
Action: <the integer count>
The Action is a count, not a card type. End immediately after its line."""
    if variant == 'icl':
        text += """

Six consecutive turns from a SEPARATE example game:
User: New card A. Query B.
Assistant:
Thought: The stream is A. No B has appeared.
Action: 0
User: New card B. Query A.
Assistant:
Thought: The stream is A, B. A has appeared once.
Action: 1
User: New card A. Query A.
Assistant:
Thought: The stream is A, B, A. A has appeared twice.
Action: 2
User: New card B. Query A.
Assistant:
Thought: The stream is A, B, A, B. A still has two appearances.
Action: 2
User: New card B. Query B.
Assistant:
Thought: The stream is A, B, A, B, B. B has appeared three times.
Action: 3
User: New card A. Query B.
Assistant:
Thought: The stream is A, B, A, B, B, A. B still has three appearances.
Action: 3
End of example. The ACTUAL game starts below with ALL counts reset to zero."""
    elif variant == 'notebook_control':
        text += '\nBefore answering, write all current dealt-value counts compactly in your scratch work.'
    return text


def parse(text, maximum):
    import re
    matches = list(re.finditer(r'^\**Action:[ \t]+\**(\d+)\**[ \t]*$',text,re.MULTILINE|re.IGNORECASE))
    valid = len(matches)==1 and not text[matches[0].end():].strip() and int(matches[0][1])<=maximum
    return (int(matches[0][1]),False) if valid else (0,True)


def run(args):
    torch.manual_seed(11)
    args.stop_at_answer = True
    args.answer_marker = 'Action:'
    client = LocalPolicy(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    constructor = getattr(count_recall, args.task)
    native_source = Path(inspect.getfile(count_recall))
    source_sha = hashlib.sha256(native_source.read_bytes()).hexdigest()
    rows = []
    for seed in range(1_000_000, 1_000_000+args.count):
        env = constructor()
        observation, _ = env.reset(seed=seed)
        counts = np.zeros(env.num_distinct_cards, dtype=np.int64)
        messages = [dict(role='system', content=prompt(env,args.variant,args.max_new_tokens))]
        (output/'resolved_config.json').write_text(json.dumps(dict(model=args.model,task=args.task,
            variant=args.variant,prompt=messages[0]['content'],
            max_new_tokens=None if args.unlimited_response else args.max_new_tokens,
            sample=args.sample,native_source_sha256=source_sha,hidden_info_serialized=False,
            unlimited_response=args.unlimited_response,encoding='class IDs mapped bijectively to letters',
            examples_only_in_system_prompt=True),indent=2))
        client.calls = []
        turns, rewards = [], []
        for step in range(env.max_episode_length):
            dealt, query = map(int, observation)
            counts[dealt] += 1
            expected = int(counts[query])
            content = ('ACTUAL GAME START: counts were zero before this card.\n' if step==0 else '')
            content += f'New card {chr(65+dealt)}. Query {chr(65+query)}.'
            messages.append(dict(role='user', content=content))
            response = client.chat(messages)['content']
            if client.calls[-1]['context_exhausted']:
                raise RuntimeError('Context exhausted before a completed response; censored, not a native game failure')
            action, error = parse(response, env.max_card_count)
            messages.append(dict(role='assistant', content=response))
            observation, reward, terminated, truncated, _ = env.step(action)
            correct = action == expected
            assert math.isclose(reward, (1 if correct else -1)/env.max_episode_length)
            rewards.append(float(reward))
            turns.append(dict(step=step,dealt=dealt,query=query,action=action,expected=expected,
                              correct=correct,format_error=error,reward=reward,response=response))
            (output/'progress.json').write_text(json.dumps(dict(seed=seed,task=args.task,
                completed_queries=len(turns),total_queries=env.max_episode_length,
                accuracy=sum(t['correct'] for t in turns)/len(turns),
                format_error_rate=sum(t['format_error'] for t in turns)/len(turns),
                generated_tokens=sum(len(c['sampled_ids']) for c in client.calls),
                inference_seconds=sum(c['seconds'] for c in client.calls),
                latest_response=response),indent=2))
            if terminated or truncated:
                break
        assert len(turns) == env.max_episode_length and terminated and not truncated
        record = dict(seed=seed,task=args.task,variant=args.variant,native_return=math.fsum(rewards),
                      accuracy=sum(t['correct'] for t in turns)/len(turns),
                      format_error_rate=sum(t['format_error'] for t in turns)/len(turns),
                      perfect=all(t['correct'] for t in turns),turns=turns,calls=client.calls)
        with gzip.open(output/f'seed-{seed}.json.gz', 'wt') as handle:
            json.dump(record,handle)
        rows.append({k:v for k,v in record.items() if k not in ('turns','calls')})
        summary = dict(episodes=len(rows),task=args.task,variant=args.variant,
                       native_return=sum(r['native_return'] for r in rows)/len(rows),
                       accuracy=sum(r['accuracy'] for r in rows)/len(rows),
                       format_error_rate=sum(r['format_error_rate'] for r in rows)/len(rows),
                       perfect_episode_rate=sum(r['perfect'] for r in rows)/len(rows),
                       native_source_sha256=source_sha,model=args.model,
                       sample=args.sample,max_new_tokens=None if args.unlimited_response else args.max_new_tokens,
                       unlimited_response=args.unlimited_response,
                       notebook_control_only=args.variant=='notebook_control',
                       scope='Full-context qualification only; no RL or SFT',episodes_detail=rows)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k!='episodes_detail'}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', choices=['CountRecallEasy','CountRecallMedium','CountRecallHard'],default='CountRecallMedium')
    parser.add_argument('--variant', choices=['neutral','icl','notebook_control'],default='icl')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--unlimited-response', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
