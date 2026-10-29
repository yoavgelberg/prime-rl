"""CPU calibration on the pinned native game; never supplies model training data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys


def run_episode(env_class, board_seed, policy, max_responses=96):
    env = env_class(rows=4, cols=4, seed=board_seed)
    obs = env.reset()
    coords = [(r, c) for r in range(4) for c in range(4)]
    known, matched = {}, set()
    digest = hashlib.sha256(f'rng-calibration-v1:{board_seed}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'little'))
    attempts = 0
    while not obs.done and attempts < max_responses:
        available = [p for p in coords if p not in matched and p != obs.first_flip_coord]
        if policy == 'random':
            action = rng.choice(available)
        else:
            partners = []
            if obs.phase == 'flip_second':
                partners = [p for p in available if known.get(p) == obs.first_flip_face]
            else:
                groups = defaultdict(list)
                for p, face in known.items():
                    if p not in matched:
                        groups[face].append(p)
                partners = next((sorted(group) for group in groups.values() if len(group) >= 2), [])
            unseen = [p for p in available if p not in known]
            action = min(partners) if partners else (unseen or available)[0]
        first = obs.first_flip_coord
        obs = env.step(action)
        attempts += 1
        if obs.phase == 'flip_second':
            known[action] = obs.first_flip_face
        else:
            known[action] = obs.last_result['face2']
            if obs.last_result['matched']:
                matched.update((first, action))
                del known[first], known[action]
    return dict(seed=board_seed, policy=policy, pairs=obs.score, total_pairs=8,
                attempts=attempts, completed=obs.done,
                efficiency_return=obs.score / 8 - attempts / max_responses)


def calibrate(source, count=1000, start_seed=0, max_responses=96):
    sys.path.insert(0, str(Path(source) / '1_matching_pairs_new'))
    from env.matching_env import MatchingEnv
    result = dict(source=str(source), max_responses=max_responses, start_seed=start_seed,
                  boards=count, scope='Scripted native-game CPU calibration, not LLM results', policies={})
    for policy in ('random', 'perfect_recall'):
        rows = [run_episode(MatchingEnv, seed, policy, max_responses)
                for seed in range(start_seed, start_seed + count)]
        result['policies'][policy] = dict(mean_pairs=statistics.mean(r['pairs'] for r in rows),
            mean_attempts=statistics.mean(r['attempts'] for r in rows),
            completion_rate=statistics.mean(r['completed'] for r in rows),
            mean_efficiency_return=statistics.mean(r['efficiency_return'] for r in rows),
            max_attempts=max(r['attempts'] for r in rows), episodes=rows)
        if policy == 'perfect_recall':
            assert all(r['pairs'] == 8 for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--max-responses', type=int, default=96)
    args = parser.parse_args()
    result = calibrate(args.source, args.count, args.start_seed, args.max_responses)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'policies'}))
    for policy, values in result['policies'].items():
        print(json.dumps(dict(policy=policy, **{k: v for k, v in values.items() if k != 'episodes'})))


if __name__ == '__main__':
    main()
