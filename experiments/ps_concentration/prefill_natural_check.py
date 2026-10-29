import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from .audit import token_audit
from .check_policy import parity, toy_reference
from .config import Config
from .metrics import summarize
from .policy import load_model
from .rollout import run_batch, save_episodes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--attention',nargs='+',choices=['ps','full'],default=['ps','full'])
    parser.add_argument('--prefill-tokens',type=int,choices=[0,32],default=32)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    os.environ['PS_CONCENTRATION_PREFILL_TOKENS']=str(args.prefill_tokens)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    base=load_model(args.model)
    results={}
    toy=toy_reference()
    for attention in args.attention:
        config=Config(model=args.model,attention=attention,
                      transcript_cap=token_audit(tokenizer)['52']['required_cap'],rollout_prefill_tokens=args.prefill_tokens,
                      full_decode_splits=int(os.environ.get('PS_CONCENTRATION_FULL_SPLITS','1')))
        records=run_batch(base,tokenizer,[1_000_000,1_000_001],[11,22],config,greedy=True)
        save_episodes(output/f'{attention}.jsonl.gz',records)
        results[attention]=dict(parity=parity(base,records),metrics=summarize(records))
    result=dict(status='GO' if all(v['parity']['status']=='GO' for v in results.values()) else 'HOLD',
                scope='Two-board systems diagnostic only; no training, window selection, or final results',
                toy=toy,conditions=results,prefill_tokens=args.prefill_tokens,
                full_decode_splits=int(os.environ.get('PS_CONCENTRATION_FULL_SPLITS','1')))
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
