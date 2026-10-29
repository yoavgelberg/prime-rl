import argparse
import json
from pathlib import Path
import time

import torch

from .policy import load_model, StreamingQwen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = load_model(args.model)
    results = {}
    for attention in ('ps', 'full'):
        beginning, cap = (20, 256) if attention == 'ps' else (2020, 4096)
        eager = StreamingQwen(base, 2, 4, 16, attention, cap)
        eager.use_graphs = False
        graph = StreamingQwen(base, 2, 4, 16, attention, cap)
        graph.use_graphs = True
        torch.manual_seed(11)
        tokens = torch.randint(0, base.config.vocab_size, (2, beginning+64), device=base.device)
        eager.prefill(tokens[:, :beginning])
        graph.prefill(tokens[:, :beginning])
        prefix = [k[:, :, :4].clone() for k in graph.keys]
        errors = []
        started = time.perf_counter()
        for i in range(beginning, beginning+64):
            expected, actual = eager.step(tokens[:, i]), graph.step(tokens[:, i])
            errors.append(float((expected-actual).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        for want, got in zip(prefix, graph.keys, strict=True):
            torch.testing.assert_close(got[:, :, :4], want, rtol=0, atol=0)
        for want, got in zip(eager.keys, graph.keys, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        results[attention] = dict(max_logit_error=max(errors), steps=64, seconds=time.perf_counter()-started,
                                  graphs=len(graph.graphs), immutable_prefix=True,
                                  cache_tokens=min(graph.consumed, graph.capacity))
        del eager, graph
    Path(args.output).write_text(json.dumps(dict(status='GO', conditions=results), indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
