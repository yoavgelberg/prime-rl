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


def prompt(env, variant, budget):
    text = f"""You are playing Count Recall. Answer the current query; do not explain the rules.
There are {env.num_distinct_cards} possible values, numbered 0 through {env.num_distinct_cards-1}. Counts start at zero before the first observation. Each observation contains ONE newly dealt value and a queried value. Count the newly dealt value once, then answer how many times the queried value has been DEALT so far, INCLUDING this observation. The queried value is not an additional dealt value. Count only dealt values, never queries or values mentioned in your own response. There are {env.max_episode_length} queries. Each correct answer earns +1/{env.max_episode_length}, and each incorrect answer earns -1/{env.max_episode_length}. Maximize total reward.
Use brief scratch work if helpful. Respond to each observation within {budget} tokens and end with exactly one line:
ANSWER N
Replace N with the count. Issue your answer now, without restating the task."""
    if variant == 'icl':
        text += """
Example on a separate stream:
dealt=0 query=1
The queried value has not appeared.
ANSWER 0
dealt=1 query=0
0 has appeared once.
ANSWER 1
dealt=0 query=0
0 has appeared twice.
ANSWER 2
End of example. All counts are reset to zero before the actual first observation below."""
    elif variant == 'notebook_control':
        text += '\nBefore answering, write all current dealt-value counts compactly in your scratch work.'
    return text


def parse(text, maximum):
    import re
    matches = re.findall(r'^ANSWER (\d+)[ \t]*$', text, re.MULTILINE)
    valid = len(matches) == 1 and text.rstrip().splitlines()[-1] == f'ANSWER {matches[0]}' and int(matches[0]) <= maximum
    return (int(matches[0]), False) if valid else (0, True)


def run(args):
    torch.manual_seed(11)
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
            variant=args.variant,prompt=messages[0]['content'],max_new_tokens=args.max_new_tokens,
            sample=args.sample,native_source_sha256=source_sha,hidden_info_serialized=False),indent=2))
        client.calls = []
        turns, rewards = [], []
        for step in range(env.max_episode_length):
            dealt, query = map(int, observation)
            counts[dealt] += 1
            expected = int(counts[query])
            content = f'step={step} limit={env.max_episode_length} dealt={dealt} query={query} last_reward={rewards[-1] if rewards else 0.:.8f}\nAnswer the current query now.'
            messages.append(dict(role='user', content=content))
            response = client.chat(messages)['content']
            action, error = parse(response, env.max_card_count)
            messages.append(dict(role='assistant', content=response))
            observation, reward, terminated, truncated, _ = env.step(action)
            correct = action == expected
            assert math.isclose(reward, (1 if correct else -1)/env.max_episode_length)
            rewards.append(float(reward))
            turns.append(dict(step=step,dealt=dealt,query=query,action=action,expected=expected,
                              correct=correct,format_error=error,reward=reward,response=response))
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
                       sample=args.sample,max_new_tokens=args.max_new_tokens,
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
    run(parser.parse_args())


if __name__ == '__main__':
    main()
