import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import popgym.envs.concentration as concentration

from .config import SOURCE_SHA256, system_prompt
from .environment import make_env, parse_action, scripted_episode


def native_checks():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 52 and env.episode_length == 104 and np.all(obs == 13)
    assert sorted(np.bincount(env.state).tolist()) == [4] * 13
    p = 0
    q = next(i for i in range(1, 52) if env.state[i] != env.state[p])
    assert env.step(p)[1:4] == (0, False, False)
    obs, reward, _, _, _ = env.step(q)
    assert reward == -2 / 104 and obs[p] == env.state[p] and obs[q] == env.state[q]
    assert env.deck['in_play_idx'] == []
    obs, reward, _, _, _ = env.step(51)
    assert reward == 0 and obs[p] == 13 and obs[q] == 13
    env.reset(seed=0)
    env.step(p)
    assert env.step(p)[1] == -2 / 104
    env.reset(seed=0)
    q = next(i for i in range(1, 52) if env.state[i] == env.state[p])
    env.step(p)
    assert env.step(q)[1] == 1 / 26
    assert env.step(p)[1] == -1 / 104
    assert not env.deck['in_play_idx']
    env.step(next(i for i in range(52) if i not in (p, q)))
    assert env.step(q)[1] == -2 / 104
    env.reset(seed=0)
    for i in range(104):
        _, reward, terminated, truncated, _ = env.step(0)
        assert not terminated and truncated == (i == 103)
        assert ('second' if reward == 0 else 'first') == ('second' if env.deck['in_play_idx'] else 'first')
    env.reset(seed=0)
    pairs = []
    for rank in range(13):
        positions = np.where(env.state == rank)[0].tolist()
        pairs.extend(positions)
    total = 0
    for i, p in enumerate(pairs):
        _, r, terminated, truncated, _ = env.step(p)
        total += r
        assert terminated == (i == 51) and not truncated
    assert np.isclose(total, 1.)
    for text, expected in [('FLIP 17', (17, False)), ('notes\nFLIP 51\n', (51, False)),
                           ('FLIP 1\nFLIP 2', (0, True)), ('FLIP 52', (0, True)),
                           ('FLIP 2\nmore', (0, True)), ('hello', (0, True))]:
        assert parse_action(text) == expected
    big = make_env(True)
    assert big.action_space.n == 104 and big.episode_length == 208
    return dict(native_rules='PASS', transient_observation='PASS', phase='PASS', parser='PASS', larger='PASS')


def token_audit(tokenizer):
    records = {}
    for cards, limit, cap in [(52, 104, 65536), (104, 208, 262144)]:
        prefix = tokenizer.apply_chat_template([dict(role='system', content=system_prompt(cards, limit))],
                                               tokenize=True, add_generation_prompt=False)
        prompted_prefix = tokenizer.apply_chat_template([dict(role='system', content=system_prompt(cards, limit, True))],
                                                        tokenize=True, add_generation_prompt=False)
        assert len(prefix) <= 512
        reset_text = f'step=0 limit={limit} next_pick=first last_action=none reward=0.00000000\nvisible={{}}'
        complete = tokenizer.apply_chat_template([dict(role='system', content=system_prompt(cards, limit)),
                                                   dict(role='user', content=reset_text)], tokenize=True,
                                                  add_generation_prompt=True)
        assert complete[:len(prefix)] == prefix, 'Native template must preserve the fixed system prefix'
        # Byte bound is conservative for Qwen's byte-level tokenizer, including all
        # possible observations and delimiters; generated tokens are counted directly.
        observation = (f'step={limit} limit={limit} next_pick=second last_action={cards-1} reward=-0.01923077\n'
                       + 'visible={' + ','.join(f'{i}:12' for i in range(cards)) + '}')
        template_bytes = len('<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n'.encode())
        bound = max(len(prefix), len(prompted_prefix)) + (limit + 1) * (len(observation.encode()) + template_bytes) + limit * 256
        records[str(cards)] = dict(prefix_tokens=len(prefix), conservative_bound=bound,
                                   prompted_prefix_tokens=len(prompted_prefix),
                                   initial_cap=cap, required_cap=max(cap, bound),
                                   tokenizer_basis='byte-level UTF-8 upper bound')
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/concentration/audit')
    parser.add_argument('--seeds', type=int, default=1000)
    parser.add_argument('--tokenizer')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    versions = {x: importlib.metadata.version(x) for x in ('popgym', 'gymnasium', 'numpy')}
    assert versions['popgym'] == '1.0.7' and versions['gymnasium'] == '1.0.0'
    sha = hashlib.sha256(Path(inspect.getfile(concentration)).read_bytes()).hexdigest()
    assert sha == SOURCE_SHA256, f'Inspect source discrepancy before comparisons: {sha}'
    checks = native_checks()
    summary = {}
    for policy, random in [('random', True), ('memory', False)]:
        rows = [scripted_episode(seed, random) for seed in range(args.seeds)]
        (out / f'{policy}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary[policy] = {key: float(np.mean([row[key] for row in rows]))
                           for key in ('native_return', 'matched_fraction', 'completed', 'flips')}
        summary[policy]['flips_p95'] = float(np.percentile([row['flips'] for row in rows], 95))
    if args.seeds == 1000:
        assert abs(summary['memory']['native_return'] - .69965) < .00001, summary
        assert summary['memory']['completed'] == 1 and summary['memory']['flips_p95'] == 88
        assert abs(summary['random']['native_return'] + .82156) < .015
    tokens = {}
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokens = token_audit(AutoTokenizer.from_pretrained(args.tokenizer))
    result = dict(status='GO', scope='CPU environment and scripted-policy audit only; GPU parity pending',
                  environment=versions, source_sha256=sha, checks=checks, calibration=summary,
                  token_audit=tokens, gate_values=dict(memory_completion=1., source_hash_equal=True))
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'dependencies.lock.txt').write_text(''.join(sorted(
        f'{d.metadata["Name"]}=={d.version}\n' for d in importlib.metadata.distributions())))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
