import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .audit import token_audit
from .check_policy import accept_parity, parity, toy_reference
from .config import Config
from .policy import load_model
from .rollout import run_batch, save_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-numerical-target-miss', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cap = token_audit(tokenizer)['52']['required_cap']
    model = load_model(args.model)
    config = Config(model=args.model, transcript_cap=cap)
    records = run_batch(model, tokenizer, [1_000_000, 1_000_001], [11, 22], config, greedy=True)
    save_episodes(output / 'ps.jsonl.gz', records)
    checks = dict(toy=toy_reference(), parity=parity(model, records))
    (output / 'checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if not accept_parity(checks['parity'], args.allow_numerical_target_miss):
        raise RuntimeError('Parity gate failed')


if __name__ == '__main__':
    main()
