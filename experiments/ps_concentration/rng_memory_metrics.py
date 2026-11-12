"""Offline known-partner scoring from actual environment text and sampled IDs."""
import argparse
import gzip
import json
from pathlib import Path
import re


ANALYSIS_VERSION = 'native-known-match-v2-first-coordinate'


def decoded_offsets(tokenizer, tokens):
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(encoded['input_ids']) == list(tokens), 'Token round-trip differs; investigate before assigning reveal ages'
    return text, encoded['offset_mapping']


def token_at(offsets, character):
    matches = [index for index, (start, end) in enumerate(offsets) if start <= character < end]
    assert matches, character
    return matches[-1]


def sampled_token_at(tokenizer, tokens, character, text):
    low, high = 1, len(tokens)
    while low < high:
        middle = (low + high) // 2
        prefix = tokenizer.decode(tokens[:middle], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(prefix) > character:
            high = middle
        else:
            low = middle + 1
    prefix = tokenizer.decode(tokens[:low], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert prefix[character] == text[character]
    return low - 1


def direct_reveals(text, offsets, absolute_start, rows, cols):
    reveals = []
    pattern = r'(?m)^([a-z])[ \t]+((?:\[[^\]\n]*\])+)[ \t]*$'
    for row in re.finditer(pattern, text):
        if ord(row[1]) - ord('a') >= rows:
            continue
        cells = list(re.finditer(r'\[([^\]]*)\]', row[2]))
        assert len(cells) == cols
        for column, cell in enumerate(cells):
            raw, value = cell[1], cell[1].strip()
            if value in ('', '*'):
                continue
            character = row.start(2) + cell.start(1) + len(raw) - len(raw.lstrip()) + len(value) - 1
            position = absolute_start + token_at(offsets, character)
            reveals.append((row[1] + chr(ord('A') + column), value, position))
    return reveals


def action_query(tokenizer, tokens, absolute_start, native_coord):
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    final_start = text.rfind('</think>')
    final_start = final_start + len('</think>') if final_start >= 0 else 0
    final = text[final_start:]
    markers = list(re.finditer(r'(?:^|[^A-Za-z])\**action\**\s*:', final, flags=re.IGNORECASE))
    if markers:
        marker = markers[-1]
        coordinate = re.match(r'\s*\**\s*([a-z])\s*([A-Z])(?![a-zA-Z])', final[marker.end():])
        if coordinate:
            if native_coord is not None:
                assert coordinate[1] + coordinate[2] == native_coord
            character = final_start + marker.end() + coordinate.start(1)
            return absolute_start + sampled_token_at(tokenizer, tokens, character, text) - 1
    assert native_coord is None
    return absolute_start + len(tokens) - 2


def known_match_decisions(record, game, tokenizer):
    first, last_round, previous_end = None, None, record['prefix_length']
    known, matched, decisions = {}, set(), []
    complete = [turn for turn in record['turns'] if turn['stop_condition'] != 'transcript_capacity']
    assert len(game['turns']) >= len(complete)
    rows, cols = game['config']['rows'], game['config']['cols']
    for number, (turn, native) in enumerate(zip(complete, game['turns'], strict=False)):
        assert turn['end'] <= len(record['tokens'])
        if native['round'] != last_round:
            first = None
        last_round = native['round']
        text, offsets = decoded_offsets(tokenizer, record['tokens'][previous_end:turn['start']])
        for position, rank, reveal in direct_reveals(text, offsets, previous_end, rows, cols):
            if position not in matched:
                known[position] = (rank, reveal)
        chosen = native['parse']['coord'] if native['parse']['ok'] else None
        query = action_query(tokenizer, record['tokens'][turn['start']:turn['end']], turn['start'], chosen)
        if first is not None:
            first_position, target = first
            assert first_position in known and known[first_position][0] == target
            partners = {position: query - reveal for position, (rank, reveal) in known.items()
                        if position != first_position and position not in matched and rank == target}
            assert all(age >= 0 for age in partners.values())
            if partners:
                age = min(partners.values())
                window = record['window']
                age_bin = '<=W' if age <= window else '(W,2W]' if age <= 2 * window else '>2W'
                decisions.append(dict(response_index=number, turn_start=turn['start'], query_position=query,
                    first_position=first_position, target_rank=target, partners=partners, direct_reveal_age=age,
                    known_unmatched={position: dict(rank=rank, last_direct_reveal=reveal)
                        for position, (rank, reveal) in known.items() if position != first_position and position not in matched},
                    age_bin=age_bin, chosen=chosen,
                    correct=bool(native['env_result']['applied'] and chosen in partners),
                    retained_suffix_start=max(record['prefix_length'], query - window + 1)))
        result = native['env_result']
        if result['applied']:
            rank = result['revealed_face']
            if result['phase_after'] == 'flip_second':
                assert first is None
                first = chosen, rank
            else:
                assert first is not None
                if first[1] == rank:
                    matched.update((first[0], chosen))
                    known.pop(first[0], None)
                    known.pop(chosen, None)
                first = None
        previous_end = turn['end']
    return decisions


def summarize(decisions):
    bins = {}
    for name in ('<=W', '(W,2W]', '>2W'):
        rows = [row for row in decisions if row['age_bin'] == name]
        correct = sum(row['correct'] for row in rows)
        bins[name] = dict(opportunities=len(rows), correct=correct,
                          utilization=correct / len(rows) if rows else None)
    return bins


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    parser.add_argument('--game', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with gzip.open(args.record, 'rt') as handle:
        record = json.load(handle)
    game = json.loads(Path(args.game).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    decisions = known_match_decisions(record, game, tokenizer)
    result = dict(known_match=summarize(decisions), decisions=decisions,
        analysis_version=ANALYSIS_VERSION,
        age_definition='Original action-query token position minus latest direct environment rank-token position; notes never reset age',
        scope='Completed native responses only; unfinished context-censored responses are excluded',
        record=str(args.record), game=str(args.game))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['known_match'], indent=2))


if __name__ == '__main__':
    main()
