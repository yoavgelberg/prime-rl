import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import Config, SPLITS
from .control import handoff_requested
from .metrics import read_episodes, summarize


def decide(results):
    full = results['neutral_full']['matched_fraction']
    gaps = {str(w): full - results[f'neutral_ps{w}']['matched_fraction'] for w in (1024, 512)}
    window = next((w for w in (1024, 512) if gaps[str(w)] >= .10), None)
    prompted = results['prompted_full']['matched_fraction']
    return dict(status='GO' if window is not None else 'HOLD', window=window,
                full_competent=full >= .80, skip_full_rl=full >= .80,
                full_pilot_required=prompted < .80, neutral_full_matched_fraction=full,
                prompted_full_matched_fraction=prompted, matched_fraction_gaps=gaps,
                required_gap=.10, required_full_competence=.80,
                interpretation='Window selection only; implementation and RL pilot gates remain separate')


def run(args):
    from transformers import AutoTokenizer
    from .audit import token_audit
    from .policy import load_model
    from .rollout import run_batch, save_episodes
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_bounds = token_audit(tokenizer)
    (out / 'token_bounds.json').write_text(json.dumps(token_bounds, indent=2))
    cap = token_bounds['52']['required_cap']
    base = load_model(args.model)
    seeds = list(SPLITS[args.split])[:args.count]
    results = {}
    conditions = [('neutral_full', 'full', 1024, False),
                  ('neutral_ps1024', 'ps', 1024, False), ('neutral_ps512', 'ps', 512, False),
                  ('neutral_ps2048', 'ps', 2048, False), ('prompted_full', 'full', 1024, True),
                  ('prompted_ps1024', 'ps', 1024, True), ('prompted_ps512', 'ps', 512, True)]
    if args.conditions:
        conditions = [c for c in conditions if c[0] in args.conditions]
    for name, attention, window, prompted in conditions:
        config = Config(model=args.model, attention=attention, window=window, transcript_cap=cap)
        path = out / f'{name}.jsonl.gz'
        existing = read_episodes(path) if path.exists() else []
        completed = {r['board_seed'] for r in existing}
        remaining = [s for s in seeds if s not in completed]
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start+args.batch_size]
            records = run_batch(base, tokenizer, batch, batch, config, greedy=True, prompted=prompted)
            save_episodes(path, records)
            existing.extend(records)
            (out / f'{name}.summary.json').write_text(json.dumps(summarize(existing), indent=2))
            if handoff_requested():
                return
        results[name] = summarize(existing)
        print(json.dumps(dict(condition=name, **results[name])), flush=True)
    all_summaries = {p.name.removesuffix('.summary.json'): json.loads(p.read_text())
                     for p in out.glob('*.summary.json')}
    ready = all(name in all_summaries and all_summaries[name]['episodes'] == args.count
                for name in ('neutral_full', 'neutral_ps1024', 'neutral_ps512', 'neutral_ps2048', 'prompted_full',
                             'prompted_ps1024', 'prompted_ps512'))
    gate = decide(all_summaries) if ready else dict(status='HOLD', scope='Screening incomplete')
    if args.count != 64:
        gate.update(status='HOLD', scope='SMOKE ONLY: fewer than 64 screening boards; do not freeze W')
    (out / ('screening.json' if ready else f'partial-{"-".join(args.conditions or ["all"])}.json')).write_text(
        json.dumps(dict(gate=gate, conditions=all_summaries), indent=2))
    print(json.dumps(gate, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--split', choices=['screening'], default='screening')
    parser.add_argument('--output', default='runs/concentration/screening')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--conditions', nargs='+')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
