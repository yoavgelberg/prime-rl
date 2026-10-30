"""Collect conservative, auditable text-memory interventions from held-out games."""

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re

from .rng_memory_metrics import ANALYSIS_VERSION, known_match_decisions


ASSOCIATION = re.compile(r'(?<![A-Za-z])([a-d][A-D])\s*(?::|=|->|→|\bis\b)\s*(?:rank\s*)?(\d+)(?!\d)', re.I)


def action_line(text):
    return bool(re.search(r'(?:^|[^A-Za-z])\**action\**\s*:', text, flags=re.I)
                or re.match(r'\s*\**FLIP\b', text, flags=re.I))


def boundary(tokenizer, tokens, character):
    low, high = 0, len(tokens)
    while low < high:
        middle = (low + high) // 2
        prefix = tokenizer.decode(tokens[:middle], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(prefix) < character:
            low = middle + 1
        else:
            high = middle
    prefix = tokenizer.decode(tokens[:low], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return low if len(prefix) == character else None


def associations(record, tokenizer, start, end, manual=()):
    notes = []
    position = start
    while position < end:
        if record['roles'][position] != 'assistant':
            position += 1
            continue
        finish = position + 1
        while finish < end and record['roles'][finish] == 'assistant':
            finish += 1
        tokens = record['tokens'][position:finish]
        text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for match in ASSOCIATION.finditer(text):
            line_end = text.find('\n', match.end())
            line = text[text.rfind('\n', 0, match.start()) + 1:line_end if line_end >= 0 else len(text)]
            if action_line(line):
                continue
            lo, hi = [boundary(tokenizer, tokens, character) for character in match.span(1)]
            coordinate = match[1]
            if lo is None or hi is None or not re.fullmatch(r'[a-d][A-D]', coordinate):
                continue
            notes.append(dict(position=coordinate, surface=coordinate, rank=match[2], start=position + lo, end=position + hi,
                recognition='Explicit generated coordinate:rank, coordinate=rank, coordinate->rank, or coordinate is rank'))
        position = finish
    for item in manual:
        lo, hi = item['start'], item['end']
        surface = item.get('surface', item['position'])
        if not re.fullmatch(r'[a-d][A-D]', item['position']) or not surface or '\n' in surface:
            raise ValueError('Manual annotations need a native board position and a single-line coordinate surface')
        if not start <= lo < hi <= end or not all(role == 'assistant' for role in record['roles'][lo:hi]):
            raise ValueError('Manual associations must be inside retained sampled scratch work')
        if tokenizer.decode(record['tokens'][lo:hi], skip_special_tokens=False) != surface:
            raise ValueError('Manual coordinate annotation does not match the original token span')
        for turn in record['turns']:
            if turn['start'] <= lo < turn['end']:
                before = tokenizer.decode(record['tokens'][turn['start']:lo], skip_special_tokens=False)
                after = tokenizer.decode(record['tokens'][lo:turn['end']], skip_special_tokens=False)
                if action_line(before.rsplit('\n', 1)[-1] + after.split('\n', 1)[0]):
                    raise ValueError('A manual note edit cannot alter a past action line')
        notes = [note for note in notes if (note['start'], note['end']) != (lo, hi)]
        notes.append(dict(item, surface=surface, rank=str(item['rank']),
                          recognition='Manual association annotation; exact sampled surface mapped to native position'))
    unique = {(item['start'], item['end'], item['position'], item['rank']): item for item in notes}
    return sorted(unique.values(), key=lambda item: item['start'])


def swap_edits(notes, first, second, record, tokenizer):
    edits = []
    surfaces = {position:{item.get('surface', position) for item in notes if item['position'] == position}
                for position in (first, second)}
    for item in notes:
        if item['position'] not in (first, second):
            continue
        replacement = second if item['position'] == first else first
        lo, hi = item['start'], item['end']
        old = record['tokens'][lo:hi]
        surface = item.get('surface', item['position'])
        if surface == item['position']:
            replacement_surface = replacement
        elif replacement in item.get('replacement_surfaces', {}):
            replacement_surface = item['replacement_surfaces'][replacement]
        elif len(surfaces[replacement]) == 1:
            replacement_surface = next(iter(surfaces[replacement]))
        else:
            return None
        if not replacement_surface or '\n' in replacement_surface:
            raise ValueError('A replacement coordinate surface must fit one scratch-work line')
        new = tokenizer.encode(replacement_surface, add_special_tokens=False)
        if len(new) != len(old) or tokenizer.decode(old, skip_special_tokens=False) != surface:
            return None
        edits.append(dict(start=lo, end=hi, old=old, new=new, association=item,
                          replacement_position=replacement, replacement_surface=replacement_surface))
    return edits or None


def edit_size(edits):
    return (sum(len(item['old']) for item in edits),
            sum(sum(a != b for a, b in zip(item['old'], item['new'], strict=True)) for item in edits))


def find_edits(notes, partner, target, known, record, tokenizer):
    for false in sorted(known):
        if false == partner or known[false]['rank'] == target:
            continue
        relevant = [item for item in notes if item['position'] in (partner, false)]
        if any(item['rank'] != known[item['position']]['rank'] for item in relevant):
            continue
        edits = swap_edits(relevant, partner, false, record, tokenizer)
        if edits is None:
            continue
        irrelevant = [position for position in sorted(known) if position not in (partner, false)
                      and known[position]['rank'] != target]
        for left, right in itertools.combinations(irrelevant, 2):
            choices = [item for item in notes if item['position'] in (left, right)]
            if any(item['rank'] != known[item['position']]['rank'] for item in choices):
                continue
            sham = swap_edits(choices, left, right, record, tokenizer)
            if sham is not None and edit_size(sham) == edit_size(edits):
                return false, edits, sham
    return None


def apply_edits(state, edits):
    tokens = list(state['retained_token_ids'])
    positions = {position: index for index, position in enumerate(state['retained_positions'])}
    changed = set()
    for edit in edits:
        if len(edit['old']) != len(edit['new']):
            raise ValueError('Interventions must preserve token count')
        original = list(range(edit['start'], edit['end']))
        if changed.intersection(original) or any(position not in positions for position in original):
            raise ValueError('Edits must be disjoint and fully inside retained text')
        indices = [positions[position] for position in original]
        if tokens[indices[0]:indices[-1] + 1] != edit['old']:
            raise ValueError('Edited note tokens differ from the exact original record')
        if not all(state['retained_roles'][index] == 'assistant' for index in indices):
            raise ValueError('An intervention cannot change forced observations or template tokens')
        tokens[indices[0]:indices[-1] + 1] = edit['new']
        changed.update(original)
    if len(tokens) != len(state['retained_positions']):
        raise ValueError('An intervention changed context length')
    return tokens


def collect_states(record, game, tokenizer, *, maximum=128, manual=None):
    if record['attention'] != 'ps' or not record['greedy']:
        raise ValueError('Memory interventions require greedy PS trajectories')
    states = []
    for decision in known_match_decisions(record, game, tokenizer):
        if decision['chosen'] is None:
            continue
        query, window = decision['query_position'], record['window']
        start = max(record['prefix_length'], query - window + 1)
        end = query + 1
        annotations = (manual or {}).get(decision['response_index'], [])
        notes = associations(record, tokenizer, start, end, manual=annotations)
        for partner, age in sorted(decision['partners'].items()):
            if age <= 2 * window:
                continue
            target = decision['target_rank']
            relevant = [item for item in notes if item['position'] == partner and item['rank'] == target]
            if not relevant:
                continue
            positions = list(range(record['prefix_length'])) + list(range(start, end))
            known = decision['known_unmatched']
            reveal = known[partner]['last_direct_reveal']
            if reveal in positions or not reveal < start:
                raise ValueError('The direct partner reveal remains in the retained context')
            state = dict(board_seed=record['board_seed'], response_index=decision['response_index'],
                decision_position=end, query_position=query, first_position=decision['first_position'],
                target_rank=target, partner=partner, last_direct_reveal=reveal, direct_reveal_age=age,
                correct_partners=sorted(decision['partners']), streaming_action=decision['chosen'],
                window=window, prefix_length=record['prefix_length'], retained_positions=positions,
                retained_token_ids=[record['tokens'][index] for index in positions],
                retained_roles=[record['roles'][index] for index in positions], qualifying_associations=relevant,
                recognition=sorted({item['recognition'] for item in relevant}),
                analysis_version=ANALYSIS_VERSION,
                state_boundary='Immediately before the first coordinate token of the pending second-pick action')
            pair = find_edits(notes, partner, target, known, record, tokenizer)
            if pair is None:
                state['untestable_reason'] = 'No truthful unrelated sham with equal token-span size and equal changed-token count'
            else:
                false, edits, sham = pair
                state.update(false_partner=false, edits=edits, sham_edits=sham,
                    edited_token_span_count=edit_size(edits)[0], changed_token_count=edit_size(edits)[1])
                for changes in (edits, sham):
                    apply_edits(state, changes)
            states.append(state)
            break
        if len(states) >= maximum:
            break
    return states


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--annotations', help='Optional manual {board_seed,response_index,associations:[...]} JSONL')
    parser.add_argument('--maximum', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.maximum <= 128:
        parser.error('Collect at most the precommitted 128 states')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line]
    if len({row['board_seed'] for row in rows}) != len(rows):
        raise ValueError('Intervention collection requires one trajectory per board')
    manual = {}
    if args.annotations:
        for line in Path(args.annotations).read_text().splitlines():
            item = json.loads(line)
            manual.setdefault(item['board_seed'], {})[item['response_index']] = item['associations']
    states = []
    for row in sorted(rows, key=lambda item: item['board_seed']):
        if not 4000000 <= row['board_seed'] <= 4000511:
            raise ValueError('Collection must use the separate intervention seed split')
        path = Path(row['token_record'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['token_record_sha256']:
            raise ValueError('The intervention token record checksum changed')
        with gzip.open(path, 'rt') as handle:
            record = json.load(handle)
        game_path = Path(record['native_game_json'])
        game = json.loads(game_path.read_text())
        candidates = collect_states(record, game, tokenizer, maximum=args.maximum - len(states),
                                    manual=manual.get(row['board_seed']))
        for state in candidates:
            state.update(token_record=str(path), token_record_sha256=row['token_record_sha256'],
                         native_game_json=str(game_path), state_id=f'b{row["board_seed"]}-r{state["response_index"]}')
        states.extend(candidates)
        if len(states) == args.maximum:
            break
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / 'states.jsonl.gz', 'wt', compresslevel=1) as handle:
        for state in states:
            handle.write(json.dumps(state) + '\n')
    result = dict(qualified_states=len(states), testable_states=sum('edits' in state for state in states),
        minimum_for_textual_memory_conclusion=32, intervention_branches_executed=False,
        status='COLLECTED' if len(states) >= 32 else 'INSUFFICIENT QUALIFYING STATES',
        recognition='Conservative explicit coordinate/rank patterns plus exact manual token spans when provided; no LLM judge',
        causal_textual_memory_established=False)
    (output / 'collection-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
