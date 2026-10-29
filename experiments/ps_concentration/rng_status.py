"""Read-only progress and throughput summaries from saved native-game metrics."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import time

from .rng_metrics import score_prefix


def read_live_json(path):
    for _ in range(10):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            time.sleep(.02)
    return None


def collect(root, previous=None, gpu_counts=None):
    now=datetime.datetime.now(datetime.timezone.utc)
    result=dict(utc=now.isoformat(),epoch=now.timestamp(),conditions={})
    for attention in ('full','ps'):
        directory=Path(root)/attention
        rows=[value for path in directory.glob('seed-*-result.json') if (value:=read_live_json(path)) is not None]
        by_seed={row['seed']:row for row in rows}
        pairs=[row['score'] for row in rows if row.get('score') is not None and not row.get('censored',False)]
        tokens={str(row['seed']):row['generated_tokens'] for row in rows}
        active=[]
        for path in directory.glob('progress-*-batch.json'):
            row=read_live_json(path)
            if row is None or row['seed'] in by_seed:
                continue
            game=read_live_json(directory/f'seed-{row["seed"]}/game.json')
            if game is None:
                continue
            observed=score_prefix(game,96)
            tokens[str(row['seed'])]=row['generated_tokens']
            active.append(dict(seed=row['seed'],pairs=observed['score'],attempts=observed['response_count'],
                               generated_tokens=row['generated_tokens'],seconds=row['seconds'],
                               tok_s=row['generated_tokens']/row['seconds']))
        rates=[row['tok_s'] for row in active]
        condition=dict(finished=len(rows),scored_endpoints=len(pairs),
                       context_censored=sum(bool(row.get('censored')) for row in rows),
                       mean_pairs=sum(pairs)/len(pairs) if pairs else None,
                       finished_pairs=pairs,active_games=len(active),
                       active_mean_pairs=statistics.mean(row['pairs'] for row in active) if active else None,
                       per_game_tok_s_mean=statistics.mean(rates) if rates else None,
                       per_game_tok_s_median=statistics.median(rates) if rates else None,
                       per_game_tok_s_range=[min(rates),max(rates)] if rates else None,
                       observed_policy_token_counters=tokens,active=active)
        if previous and attention in previous.get('conditions',{}):
            old=previous['conditions'][attention]['observed_policy_token_counters']
            seconds=now.timestamp()-previous['epoch']
            delta=sum(tokens.values())-sum(old.values())
            monotonic=all(tokens.get(seed,0)>=count for seed,count in old.items())
            if seconds>0 and monotonic and delta>=0:
                condition.update(interval_seconds=seconds,interval_policy_tok_s=delta/seconds)
                if gpu_counts:
                    condition['interval_policy_tok_s_per_gpu']=delta/seconds/gpu_counts[attention]
        result['conditions'][attention]=condition
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output')
    parser.add_argument('--previous')
    parser.add_argument('--full-gpus',type=int,default=3)
    parser.add_argument('--ps-gpus',type=int,default=1)
    args=parser.parse_args()
    previous=json.loads(Path(args.previous).read_text()) if args.previous else None
    result=collect(args.run_dir,previous,dict(full=args.full_gpus,ps=args.ps_gpus))
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2))
    brief=dict(utc=result['utc'],conditions={key:{k:v for k,v in value.items() if k not in
               ('active','observed_policy_token_counters','finished_pairs')} for key,value in result['conditions'].items()})
    print(json.dumps(brief,indent=2))


if __name__=='__main__':
    main()
