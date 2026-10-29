"""Export exact native-thinking streams with offline eviction annotations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .rng_memory_metrics import action_query, known_match_decisions, summarize
from .rng_metrics import score_prefix


def export(record_path, game_path, tokenizer, output):
    record_path, game_path, output = Path(record_path), Path(game_path), Path(output)
    with gzip.open(record_path, 'rt') as handle:
        record = json.load(handle)
    game = json.loads(game_path.read_text())
    size = len(record['tokens'])
    assert record['positions'] == list(range(size))
    assert size == len(record['roles']) == len(record['loss_mask']) == len(record['old_logps'])
    assert record['loss_mask'] == [role == 'assistant' for role in record['roles']]
    decisions = known_match_decisions(record, game, tokenizer)
    by_response = {row['response_index']: row for row in decisions}
    complete = [turn for turn in record['turns'] if turn['stop_condition'] != 'transcript_capacity']
    scored = score_prefix(game, len(complete))
    header = ['Actual native-thinking model token stream: Matching Pairs',
        f'Attention={record["attention"]}; W={record["window"]}; P={record["prefix_length"]}.',
        f'Observed pairs={scored["score"]}/{scored["total_pairs"]}; responses={len(complete)}; context_censored={record.get("censored", False)}.',
        'All annotations below are offline analysis, never model input. No hidden board is shown.',
        'Token ranges are original half-open positions. The action query precedes its sampled coordinate token.', '']
    full, compact, cursor = header.copy(), header.copy(), 0
    for index, turn in enumerate(record['turns']):
        native = game['turns'][index] if index < len(complete) else None
        query = action_query(tokenizer, record['tokens'][turn['start']:turn['end']], turn['start'],
                             native['parse']['coord'] if native and native['parse']['ok'] else None)
        annotations = [f'RESPONSE {index + 1}: sampled [{turn["start"]},{turn["end"]}); stop={turn["stop_condition"]}']
        if record['attention'] == 'ps':
            start = max(record['prefix_length'], query - record['window'] + 1)
            annotations.append(f'At action query {query}: prefix [0,{record["prefix_length"]}) plus suffix [{start},{query + 1}); earlier suffix text evicted.')
        else:
            annotations.append(f'At action query {query}: full causal history available; no evictions.')
        if index in by_response:
            annotations.append('Known-partner opportunity from direct environment history: ' + json.dumps(by_response[index]))
        context = tokenizer.decode(record['tokens'][cursor:turn['start']], skip_special_tokens=False)
        for lines in (full, compact):
            lines.extend(['\n' + '=' * 72, *annotations, '\nNEW ENVIRONMENT/TEMPLATE TOKENS', context])
        full.extend(['\nSAMPLED ASSISTANT TOKENS, INCLUDING ALL NATIVE THINKING',
                     tokenizer.decode(record['tokens'][turn['start']:turn['end']], skip_special_tokens=False)])
        compact.extend(['\nFINAL RESPONSE', turn.get('content', '')])
        cursor = turn['end']
    if cursor < size:
        full.extend(['\nTRAILING OR UNFINISHED STREAM', tokenizer.decode(record['tokens'][cursor:], skip_special_tokens=False)])
        compact.append(f'\nUnfinished trailing stream: tokens [{cursor},{size}); see full thinking trace.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'readable-trace.txt').write_text('\n'.join(compact) + '\n')
    (output / 'full-trace-with-thinking.txt').write_text('\n'.join(full) + '\n')
    audit = dict(record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        raw_tokens=size, generated_tokens=sum(record['loss_mask']), known_match=summarize(decisions),
        decisions=decisions, native_score=scored, context_censored=record.get('censored', False),
        scope='Exact base/trained model stream with offline annotations; generated notes do not refresh direct-reveal age')
    (output / 'trace-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    parser.add_argument('--game', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(args.record, args.game, AutoTokenizer.from_pretrained(args.tokenizer), args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
