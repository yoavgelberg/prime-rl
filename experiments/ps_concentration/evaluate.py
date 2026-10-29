import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .config import Config, SPLITS
from .control import handoff_requested
from .metrics import read_episodes, summarize


def evaluate(args):
    from transformers import AutoTokenizer
    from .audit import token_audit
    from .policy import load_model
    from .rollout import run_batch, save_episodes
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group('nccl')
    out = Path(args.output or f'{args.checkpoint}/evaluation/{args.split}')
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_model(args.checkpoint, f'cuda:{rank}')
    cards, limit = (104, 208) if args.split == 'larger' else (52, 104)
    cap = token_audit(tokenizer)[str(cards)]['required_cap']
    config = Config(model=args.checkpoint, cards=cards, limit=limit, transcript_cap=cap, window=args.window,
                    rollout_prefill_tokens=int(os.environ.get('PS_CONCENTRATION_PREFILL_TOKENS','0')))
    metadata = dict(checkpoint=args.checkpoint, split=args.split, world=world, prompted=args.prompted,
                    config=asdict(config))
    experiment = Path(args.checkpoint) / 'experiment.json'
    metadata['experiment'] = json.loads(experiment.read_text()) if experiment.exists() else None
    metadata_path = out / 'evaluation.json'
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text()) == metadata, 'Do not change a partially evaluated condition'
    elif rank == 0:
        metadata_path.write_text(json.dumps(metadata, indent=2))
    conditions = [(a, args.window, None) for a in args.attention]
    if args.window_sweep:
        conditions.extend(('ps', w, 128) for w in (256, 512, 1024, 2048, 4096) if w != args.window)
    for attention, window, count in conditions:
        config.attention, config.window = attention, window
        seeds = list(SPLITS[args.split])[:count or len(SPLITS[args.split])][rank::world]
        path = out / f'{attention}-w{window}-rank{rank}.jsonl.gz'
        records = read_episodes(path) if path.exists() else []
        done = {r['board_seed'] for r in records}
        remaining = [s for s in seeds if s not in done]
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start+args.batch_size]
            fresh = run_batch(model, tokenizer, batch, batch, config, greedy=True, prompted=args.prompted)
            save_episodes(path, fresh)
            records.extend(fresh)
            if handoff_requested():
                if world > 1:
                    dist.destroy_process_group()
                return
        summary = summarize(records)
        (out / f'{attention}-w{window}-rank{rank}.summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(condition=attention, window=window, rank=rank, **summary)), flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=list(SPLITS), default='test')
    parser.add_argument('--window', type=int, default=1024)
    parser.add_argument('--attention', nargs='+', choices=['full', 'ps'], default=['full', 'ps'])
    parser.add_argument('--window-sweep', action='store_true')
    parser.add_argument('--prompted', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
