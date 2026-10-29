"""Base-model competence diagnostics; not the original frozen experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import Config, SPLITS
from .environment import make_env, serialize
from .metrics import read_episodes, summarize


CONCISE = """You are the player. Play the current turn; do not explain the rules.
There are 52 positions, 0 through 51, and 13 ranks, 0 through 12, with four cards per rank. Each action flips ONE position. Two successive valid flips form a pair. Distinct positions of equal rank match, earn 1/26 reward, and stay revealed permanently. A mismatch earns -2/104. The observation shows the unsuccessful pair before those two cards turn face down. Permanently matched cards cannot be selected again. The game ends after all cards match or 104 flips. Maximize total reward.
Each observation gives the step, remaining phase (next_pick=first or second), last action, last reward, and visible cards as position:rank. A missing position is face down. The first observation starts a fresh independent board.
Issue exactly ONE move for each observation. You may use brief scratch work if helpful. Your entire response must fit in 256 tokens, including the action. Do not restate the game or discuss general strategy. End with a single action line, replacing N by your chosen position:
FLIP N"""


def demonstrations():
    for seed in range(1000):
        env = make_env()
        obs, _ = env.reset(seed=seed)
        turns = [(serialize(obs, env.facedown_card), 17)]
        obs, reward, _, _, _ = env.step(17)
        first_rank = int(obs[17])
        turns.append((serialize(obs, env.facedown_card, 1, last_action=17, reward=reward), 23))
        obs, reward, _, _, _ = env.step(23)
        second_rank = int(obs[23])
        if first_rank == second_rank:
            continue
        turns.append((serialize(obs, env.facedown_card, 2, last_action=23, reward=reward), 8))
        obs, reward, _, _, _ = env.step(8)
        if int(obs[8]) != second_rank:
            continue
        turns.append((serialize(obs, env.facedown_card, 3, last_action=8, reward=reward), 23))
        obs, reward, _, _, _ = env.step(23)
        assert reward == 1/26 and len(env.deck['face_up_idx']) == 2
        ending = serialize(obs, env.facedown_card, 4, last_action=23, reward=reward)
        return seed, turns, ending
    raise RuntimeError('No valid illustrative trace found')


def prompts():
    seed, turns, ending = demonstrations()
    def trace(selected):
        return '\n\n'.join(f'Observation:\n{obs}\nResponse:\nFLIP {action}' for obs, action in selected)
    start = '\n\nIllustrative turns from a DIFFERENT board:\n'
    end = '\nEnd of example. The actual game starts below on an independent board.'
    reason = CONCISE.replace('You may use brief scratch work if helpful.',
                            'Briefly reason about this specific move before the action.')
    reasoning = ['The board is unknown, so explore a position.',
                 'Choose a different, unexplored position.',
                 'These ranks differ; explore a new position.',
                 'This rank matches the card previously seen at 23.']
    explained = '\n\n'.join(f'Observation:\n{obs}\nResponse:\n{why}\nFLIP {action}'
                             for (obs, action), why in zip(turns[1:], reasoning[1:], strict=True))
    strategy = ('\nFor a first pick, take a known unmatched pair if one is available; otherwise explore an unseen position. '
                'For a second pick, choose a known unmatched partner of the revealed rank if available; otherwise explore. '
                'Use the available game history to choose your move.')
    return seed, dict(concise=CONCISE,
                     format_icl=CONCISE + start + trace(turns[:2]) + end,
                     game_icl=CONCISE + start + trace(turns) + '\nObservation:\n' + ending + end,
                     reason_icl=reason + start + explained + end,
                     strategy=reason + strategy,
                     notebook_control=reason + strategy +
                     '\nKeep an updated compact list of revealed unmatched card positions and ranks in your scratch work.')


def run(args):
    from transformers import AutoTokenizer
    from .policy import load_model
    from .audit import token_audit
    from . import rollout
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_bound = token_audit(tokenizer)['52']
    example_seed, candidates = prompts()
    base = load_model(args.model)
    original_prompt = rollout.system_prompt
    seeds = list(SPLITS['screening'])[:args.count]
    results = {}
    for name in args.conditions:
        prompt = candidates[name].replace('256 tokens', f'{args.max_new_tokens} tokens')
        rollout.system_prompt = lambda cards, limit, prompted, prompt=prompt: prompt
        codec = rollout.ChatCodec(tokenizer)
        cap = original_bound['required_cap'] + max(0, len(codec.prefix)-original_bound['prefix_tokens']) + 104*(args.max_new_tokens-256)
        config = Config(model=args.model, attention=args.attention, window=args.window,
                        max_new_tokens=args.max_new_tokens, transcript_cap=cap)
        metadata = dict(config=asdict(config), prompt=prompt, prefix_tokens=len(codec.prefix),
                        example_seed=example_seed, count=args.count, variant=name,
                        scope='User-authorized prompt/ICL diagnostic; original experiment remains separate',
                        memory_note_demonstration=False, notebook_control_only=name == 'notebook_control',
                        action_history_matching_demonstration=name in ('game_icl','reason_icl'),
                        strategy_instruction=name in ('strategy','notebook_control'))
        (output/f'{name}.config.json').write_text(json.dumps(metadata, indent=2))
        path = output/f'{name}-{args.attention}-w{args.window}.jsonl.gz'
        records = read_episodes(path) if path.exists() else []
        completed = {r['board_seed'] for r in records}
        remaining = [s for s in seeds if s not in completed]
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start+args.batch_size]
            fresh = rollout.run_batch(base, tokenizer, batch, batch, config, greedy=True)
            for record in fresh:
                record.update(prompt_variant=name, diagnostic=True)
            rollout.save_episodes(path, fresh)
            records.extend(fresh)
            result = summarize(records)
            result.update(status='GO' if len(records) == 64 and result['matched_fraction'] >= .80 and
                          result['format_error_rate'] < .01 else 'HOLD',
                          variant=name, attention=args.attention, window=args.window,
                          scope='Full competence requires64 screening boards; no W selection from this smoke',
                          required_matched_fraction=.80, required_format_error_rate=.01)
            path.with_suffix('.summary.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        results[name] = summarize(records)
    rollout.system_prompt = original_prompt
    (output/'summary.json').write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', nargs='+', choices=['concise','format_icl','game_icl','reason_icl','strategy','notebook_control'],
                        default=['concise','format_icl','game_icl'])
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--attention', choices=['full','ps'], default='full')
    parser.add_argument('--window', type=int, default=1024)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
