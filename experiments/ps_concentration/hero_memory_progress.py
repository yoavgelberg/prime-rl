"""Measure old-partner successes per opportunity in complete sampled training blocks."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .hero_v2 import read, utc, write
from .rng_intervention_states import ASSOCIATION_VERSION, associations
from .rng_memory_metrics import ANALYSIS_VERSION, known_match_decisions, summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_notes(record, tokenizer, turn_start):
    end = turn_start
    start = max(record['prefix_length'], end - record['window'])
    return associations(record, tokenizer, start, end)


def valid_choices_by_turn(game):
    total = game['config']['rows'] * game['config']['cols']
    if total != 16 or game['config']['num_cards'] != 8:
        raise ValueError('This training diagnostic uses the fixed4x4two-copies-per-rank board')
    first, last_round, matched, choices = None, None, set(), {}
    for index, turn in enumerate(game['turns']):
        if turn['round'] != last_round:
            first = None
        last_round = turn['round']
        if first is not None:
            choices[index] = total - len(matched) - 1
        result = turn['env_result']
        if result['applied']:
            position, rank = turn['parse']['coord'], result['revealed_face']
            if result['phase_after'] == 'flip_second':
                assert first is None
                first = position, rank
            else:
                assert first is not None
                if first[1] == rank:
                    assert first[0] != position
                    matched.update((first[0], position))
                first = None
    return choices


def summarize_opportunities(rows, window):
    groups = {}
    for age, include in [('<=W', lambda n:n <= window),
            ('(W,2W]', lambda n:window < n <= 2*window),
            ('>2W', lambda n:n > 2*window), ('>W', lambda n:n > window)]:
        population = [row for row in rows if include(row['age_before_thinking'])]
        groups[age] = {}
        for label, note in [('all', None), ('prior_note', True), ('no_recognized_prior_note', False)]:
            selected = [r for r in population if note is None or r['prior_note'] == note]
            groups[age][label] = {}
            for subset, values in [('all_states', selected),
                    ('nonforced_states', [r for r in selected if r['valid_choices'] > 1])]:
                count = len(values)
                successes = sum(r['correct'] for r in values)
                chance = sum(1/r['valid_choices'] for r in values)
                groups[age][label][subset] = dict(opportunities=count, successes=successes,
                    failures=count-successes, success_rate=successes/count if count else None,
                    expected_random_successes=chance, random_valid_rate=chance/count if count else None,
                    excess_success_rate=(successes-chance)/count if count else None)
    return groups


def analyze(run, first, last, output):
    from transformers import AutoTokenizer

    run, output = Path(run), Path(output)
    if not 321 <= first <= last <= read(run/'state.json')['policy_version']:
        raise ValueError('Analyze only complete committed continuation batches')
    config = read(run/'resolved_config.json')
    if config['window'] != 1024 or config['temperature'] != .8:
        raise ValueError('Keep the declared W1024temperature0.8 continuation')
    if output.exists():
        raise ValueError('Preserve the previous analysis; use a unique output directory')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    output.mkdir(parents=True)
    all_rows, decisions, sources, episodes, games_with_old_notes = [], [], {}, 0, 0
    metrics = []
    for update in range(first, last+1):
        plan_path = run/f'rollout-plans/update-{update:04d}.json'
        metric_path = run/f'update-metrics/update-{update:04d}.json'
        specs, point = read(plan_path), read(metric_path)['metrics']
        assert len(specs) == point['episodes'] == 128 and point['update'] == update
        assert len({s['board_seed'] for s in specs}) == 16
        sources[str(plan_path)], sources[str(metric_path)] = digest(plan_path), digest(metric_path)
        metrics.append(point)
        for spec in specs:
            result_path = run/f'rollouts/update-{update:04d}/actor-{spec["actor_rank"]}'/spec['episode_id']/'result.json'
            result = read(result_path)
            assert all(result.get(key) == value for key,value in spec.items())
            token_path = Path(result['token_record'])
            assert digest(token_path) == result['token_record_sha256']
            with gzip.open(token_path, 'rt') as handle:
                record = json.load(handle)
            assert not record['greedy'] and record['window'] == 1024 and record['temperature'] == .8
            assert not result['censored']
            game_path = Path(record['native_game_json'])
            sources[str(token_path)], sources[str(game_path)] = result['token_record_sha256'], digest(game_path)
            with (gzip.open if game_path.suffix == '.gz' else open)(game_path, 'rt') as handle:
                game = json.load(handle)
            choices = valid_choices_by_turn(game)
            values = known_match_decisions(record, game, tokenizer)
            decisions.extend(values)
            old_note = False
            for value in values:
                assert len(value['partners']) == 1
                partner, = value['partners']
                query = value['turn_start']-1
                reveal = value['known_unmatched'][partner]['last_direct_reveal']
                age = query-reveal
                notes = prior_notes(record, tokenizer, value['turn_start'])
                present = any(n['position'] == partner and n['rank'] == value['target_rank'] for n in notes)
                count = choices[value['response_index']]
                assert count in (1,3,5,7,9,11,13,15)
                all_rows.append(dict(update=update,episode_id=spec['episode_id'],board_seed=spec['board_seed'],
                    response=value['response_index']+1,age_before_thinking=age,
                    age_at_action=value['direct_reveal_age'],partner=partner,target_rank=value['target_rank'],
                    prior_note=present,correct=value['correct'],valid_choices=count,
                    retained_suffix=[max(record['prefix_length'],value['turn_start']-1024),value['turn_start']]))
                old_note |= bool(age > 1024 and present)
            episodes += 1
            games_with_old_notes += old_note
            sources[str(result_path)] = digest(result_path)
        print(json.dumps(dict(update=update,episodes=episodes,opportunities=len(all_rows))),flush=True)
    with gzip.open(output/'opportunities.jsonl.gz','wt',compresslevel=1) as handle:
        for row in all_rows:
            handle.write(json.dumps(row)+'\n')
    summary = dict(utc=utc(),first_update=first,last_update=last,episodes=episodes,
        groups=summarize_opportunities(all_rows,1024),games_with_retained_old_target_notes=games_with_old_notes,
        known_match_at_action=summarize(decisions),
        training_means={key:sum(p[key] for p in metrics)/len(metrics) for key in
            ('mean_return','mean_native_return','mean_pairs','mean_attempts','seconds','mean_per_game_tokens_per_second')},
        input_sha256=sources,opportunities_sha256=digest(output/'opportunities.jsonl.gz'),
        analysis_version=ANALYSIS_VERSION,association_version=ASSOCIATION_VERSION,
        implementation_sha256={name:digest(Path(__file__).with_name(name)) for name in
            ('hero_memory_progress.py','rng_intervention_states.py','rng_memory_metrics.py')},
        definition='Second-pick opportunities with a previously revealed still-unmatched matchingpartner. Age beforecurrentthinking;priorcorrectmapping must be in earlier sampledassistant scratch within previous1024rawtranscripttokens. Invalid/wrong moves countas failures. Currentthinking,prefix and actionlines are excluded frompriornotes.',
        scope='Descriptive sampled-training states; conservative literalrecognition may miss indirectnotes or includehypotheticaltext. State-matchedchance is uniformvalidchoices,not an independent randomgamepolicy. No causaltext/KVclaim or independent-training-seed variability.')
    write(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--first',type=int,required=True)
    parser.add_argument('--last',type=int,required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    analyze(args.run_dir,args.first,args.last,args.output)


if __name__ == '__main__':
    main()
