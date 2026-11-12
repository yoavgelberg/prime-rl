import argparse
import json
from pathlib import Path

import torch

from .check_policy import parity
from .metrics import read_episodes
from .policy import load_model, StreamingQwen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = load_model(args.model)
    records = read_episodes(args.episodes)[:2]
    for rec in records:
        stream = StreamingQwen(base, 1, rec['prefix_length'], rec['window'], 'ps', 73186)
        values = [None] * len(rec['tokens'])
        with torch.inference_mode():
            for i, token in enumerate(rec['tokens'][:-1]):
                logits = stream.step(torch.tensor([token], device=base.device))
                if rec['loss_mask'][i+1]:
                    values[i+1] = float(logits.log_softmax(-1)[0, rec['tokens'][i+1]])
        rec['old_logps'] = values
    results = {b: parity(base, records, b) for b in ('fa4', 'dense')}
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
