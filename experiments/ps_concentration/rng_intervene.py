"""Run equal-length scratch edits from identical cold PS caches at original positions."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from threading import Event
import time

import numpy as np
import torch

from prime_rl.inference.vllm.ps_interactive import PersistentChat

from .native_cold import NativeCold
from .native_teacher import NativeTeacher
from .rng_intervention_states import apply_edits, edit_size


def native_modules(source):
    source = Path(source)
    lock = json.loads((source / 'SOURCE.json').read_text())
    if lock['commit'] != 'e3d195e857727dfa3d4a96561be202098d67da83':
        raise ValueError('Native game revision changed')
    for row in lock['manifest']:
        if hashlib.sha256((source / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Native source differs from its manifest')
    sys.path.insert(0, str(source / '1_matching_pairs_new'))
    from common.parsing import parse_coord
    from env.matching_env import MatchingEnv, InvalidActionError
    return parse_coord, MatchingEnv, InvalidActionError


def coordinate(position):
    return ord(position[0]) - ord('a'), ord(position[1]) - ord('A')


def validate_source(record, state, metadata, checkpoint):
    if (record['policy_version'] != metadata['policy_version']
            or record['policy_identity'] != str(Path(checkpoint).resolve()) or not record['greedy']
            or record['attention'] != 'ps' or record['window'] != state['window']
            or record['window'] != metadata['window'] or record['board_seed'] != state['board_seed']):
        raise ValueError('Intervention source is not a greedy PS rollout of this exact checkpoint and board')
    keys = ('thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')
    protocol = {key: record.get(key) for key in keys}
    expected = {key: metadata.get(key) for key in keys}
    if metadata.get('variant') == 'hero-v2':
        from .hero_protocol import THINKING_TRANSITION

        if expected['thinking_budget_transition'] not in (None, THINKING_TRANSITION):
            raise ValueError('Hero checkpoint declares a different thinking controller')
        expected['thinking_budget_transition'] = THINKING_TRANSITION
    if protocol != state['continuation_protocol'] or protocol != expected:
        raise ValueError('Intervention source changed the checkpoint thinking controller')


def replay_state(state, game, env_type, invalid_type):
    config = game['config']
    env = env_type(rows=config['rows'], cols=config['cols'], num_cards=config['num_cards'], seed=config['seed'])
    env.reset()
    if env.render_ground_truth(mode='text') != config['ground_truth_text']:
        raise ValueError('Offline replay did not recreate the identical native board')
    previous_round = None
    for turn in game['turns'][:state['response_index'] + 1]:
        if turn['round'] != previous_round and env.phase == 'flip_second':
            env.abort_round()
        previous_round = turn['round']
        if turn is game['turns'][state['response_index']]:
            break
        if turn['parse']['ok']:
            try:
                env.step(coordinate(turn['parse']['coord']))
            except invalid_type as error:
                if turn['env_result']['applied'] or error.code != turn['env_result'].get('code'):
                    raise ValueError('Native invalid-action replay differs from recorded consequences') from error
            else:
                if not turn['env_result']['applied'] or env.phase != turn['env_result']['phase_after']:
                    raise ValueError('Native replay phase differs from the recorded game')
    if env.phase != 'flip_second' or env.first_flip_coord != coordinate(state['first_position']):
        raise ValueError('The saved intervention is not at the claimed native second-pick state')
    if env.first_flip_face != state['target_rank']:
        raise ValueError('Target rank differs from the native environment')
    partner = coordinate(state['partner'])
    if env.board.is_removed(partner) or env.board.get_face(partner) != state['target_rank']:
        raise ValueError('The qualifying partner is not an unmatched native partner')
    if 'false_partner' in state:
        false = coordinate(state['false_partner'])
        if env.board.is_removed(false) or env.board.get_face(false) == state['target_rank']:
            raise ValueError('The false association does not point to a different unmatched rank')
    return env


def score_choice(env, action, state, invalid_type):
    if action is None:
        return dict(action=None, correct=False, follows_false=False, native_invalid=False, format_error=True)
    clone = copy.copy(env)
    clone.board, clone._last_result = copy.deepcopy(env.board), copy.deepcopy(env.last_result)
    try:
        clone.step(coordinate(action))
    except invalid_type as error:
        return dict(action=action, correct=False, follows_false=action == state.get('false_partner'),
                    native_invalid=True, format_error=False, native_error=error.code)
    return dict(action=action, correct=bool(clone.last_result['matched']),
        follows_false=action == state.get('false_partner'), native_invalid=False, format_error=False)


@torch.inference_mode()
def continue_branch(base, tokenizer, eos_ids, state, tokens, final_prefix, *, samples, rng_seed,
                    parse_coord, environment, invalid_type, transcript_cap=262144):
    stream = NativeCold(base, state['prefix_length'], state['window'], copies=samples + 1)
    logits = stream.prefill(tokens, state['retained_positions'])
    generator = torch.Generator(device=base.device).manual_seed(rng_seed)
    responses, stops = [[] for _ in range(samples + 1)], [None] * (samples + 1)
    newline = tokenizer.encode('\n', add_special_tokens=False)
    if len(newline) != 1:
        raise ValueError('The native controller requires a single newline token')
    for _ in range(transcript_cap - state['decision_position']):
        proposals = torch.cat([logits[:1].argmax(-1), torch.multinomial(
            (logits[1:].float() / .6).softmax(-1), 1, generator=generator)[:, 0]]) if samples else logits.argmax(-1)
        ids = proposals.tolist()
        for index, token in enumerate(ids):
            if stops[index] is not None:
                continue
            responses[index].append(token)
            if token in eos_ids:
                stops[index] = 'eos'
            elif token == newline[0]:
                text = final_prefix + tokenizer.decode(responses[index], skip_special_tokens=True)
                if re.search(r'(?:^|\n)Action:\s*[a-z][A-Z][ \t]*\n', text):
                    stops[index] = 'action_line'
        if all(stop is not None for stop in stops):
            break
        logits = stream.step(proposals)
    rows = []
    for index, response in enumerate(responses):
        text = final_prefix + tokenizer.decode(response, skip_special_tokens=True)
        parsed = parse_coord(text, environment.rows, environment.cols) if stops[index] else None
        action = chr(parsed[0] + ord('a')) + chr(parsed[1] + ord('A')) if parsed is not None else None
        rows.append(dict(score_choice(environment, action, state, invalid_type),
            tokens=response, final_text=text, stop_condition=stops[index] or 'transcript_capacity', greedy=index == 0))
    sampled = rows[1:]
    return dict(greedy=rows[0], sampled=sampled, sampling_temperature=.6, top_p=1., top_k=-1,
        samples=samples, probability_correct_partner=sum(row['correct'] for row in sampled) / samples if samples else None,
        probability_following_false=sum(row['follows_false'] for row in sampled) / samples if samples else None,
        probability_definition='Monte Carlo native next-flip probability from the intervention cursor; greedy continuation excluded',
        rng_seed=rng_seed, peak_kv_bytes=stream.cache_bytes,
        resource_terminations=sum(row['stop_condition'] == 'transcript_capacity' for row in rows))


@torch.inference_mode()
def continue_turn(base, tokenizer, eos_ids, state, tokens, record, *, samples, rng_seed,
                  parse_coord, environment, invalid_type, transcript_cap=262144):
    start = state['decision_position']
    if start != record['turns'][state['response_index']]['start']:
        raise ValueError('The primary intervention must precede all current-turn thinking')
    protocol = {key: record.get(key) for key in (
        'thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')}
    if protocol != state['continuation_protocol']:
        raise ValueError('Intervention continuation changed the native thinking controller')
    stream = NativeCold(base, state['prefix_length'], state['window'], copies=samples + 1)
    logits = stream.prefill(tokens, state['retained_positions'])
    chats = []
    for _ in range(samples + 1):
        chat = PersistentChat(tokenizer, eos_token_ids=eos_ids, transcript_cap=transcript_cap, **protocol)
        chat.prefix = tuple(record['tokens'][:state['prefix_length']])
        chat.tokens = list(record['tokens'][:start])
        chat.roles = list(record['roles'][:start])
        chat.old_logps = [None] * start
        chat.old_tempered_logps = [None] * start
        chat.turn_start, chat.turn_started = start, time.perf_counter()
        chat.request = dict(event=Event(), messages=0, response=None, error=None)
        chats.append(chat)
    generator = torch.Generator(device=base.device).manual_seed(rng_seed)
    for _ in range(transcript_cap - start):
        proposals = torch.cat([logits[:1].argmax(-1), torch.multinomial(
            (logits[1:].float() / .6).softmax(-1), 1, generator=generator)[:, 0]]) if samples else logits.argmax(-1)
        actual = []
        for chat, token in zip(chats, proposals.tolist(), strict=True):
            actual.append(chat.advance(token, None, None).token if chat.request is not None else next(iter(eos_ids)))
        if all(chat.request is None for chat in chats):
            break
        logits = stream.step(torch.tensor(actual, device=base.device, dtype=torch.long))
    rows = []
    for index, chat in enumerate(chats):
        turn = chat.turns[-1] if chat.turns else None
        text = turn['content'] if turn is not None else ''
        parsed = parse_coord(text, environment.rows, environment.cols) if turn is not None else None
        action = chr(parsed[0] + ord('a')) + chr(parsed[1] + ord('A')) if parsed is not None else None
        rows.append(dict(score_choice(environment, action, state, invalid_type),
            tokens=chat.tokens[start:], roles=chat.roles[start:], sampled_ids=chat.sampled,
            final_text=text, thinking_tokens=chat.thinking_tokens,
            thinking_budget_forced=chat.thinking_budget_forced,
            stop_condition=turn['stop_condition'] if turn is not None else 'transcript_capacity', greedy=index == 0))
    sampled = rows[1:]
    return dict(greedy=rows[0], sampled=sampled, sampling_temperature=.6, top_p=1., top_k=-1,
        samples=samples, probability_correct_partner=sum(row['correct'] for row in sampled) / samples if samples else None,
        probability_following_false=sum(row['follows_false'] for row in sampled) / samples if samples else None,
        probability_definition='Monte Carlo next-flip probability including newly generated thinking with the native controller; greedy excluded',
        rng_seed=rng_seed, peak_kv_bytes=stream.cache_bytes,
        resource_terminations=sum(row['stop_condition'] == 'transcript_capacity' for row in rows))


def summarize_results(states, results):
    qualified = len(states)
    executed = [row for row in results if 'unedited' in row]
    result = dict(qualified_states=qualified, executed_states=len(executed),
        untestable_states=sum('untestable_reason' in state for state in states),
        pending_states=qualified - len(results), branches={}, conditional_unedited_selects_p={},
        minimum_states=32, sufficient_executed_states=len(executed) >= 32,
        state_boundaries=sorted({state.get('state_boundary', 'unspecified') for state in states}),
        causal_textual_memory_established=False,
        causal_action_cursor_text_usage_established=False,
        limitation='NativeCold uses native-effective BF16 arithmetic and exact FA4 PS attention at original positions; native streaming likelihood audits are reported separately')
    for subset, key in ((executed, 'branches'),
                        ([row for row in executed if row['unedited']['greedy']['action'] == row['partner']],
                         'conditional_unedited_selects_p')):
        result[key]['denominator'] = len(subset)
        for branch in ('unedited', 'swap', 'sham'):
            result[key][branch] = dict(
                greedy_correct=float(np.mean([row[branch]['greedy']['correct'] for row in subset])) if subset else None,
                greedy_follows_false=float(np.mean([row[branch]['greedy']['follows_false'] for row in subset])) if subset else None,
                probability_correct_partner=float(np.mean([row[branch]['probability_correct_partner'] for row in subset])) if subset else None,
                probability_following_false=float(np.mean([row[branch]['probability_following_false'] for row in subset])) if subset else None)
    if executed:
        streaming = [row['streaming_action'] in row['correct_partners'] for row in executed]
        cold = [row['unedited']['greedy']['correct'] for row in executed]
        result['streaming_vs_unedited_cold'] = dict(denominator=len(executed),
            streaming_greedy_correct=float(np.mean(streaming)), cold_greedy_correct=float(np.mean(cold)),
            cold_minus_streaming=float(np.mean(np.array(cold, dtype=float) - streaming)))
        differences = np.array([row['swap']['probability_following_false'] - row['sham']['probability_following_false']
                                for row in executed])
        boards = sorted({row['board_seed'] for row in executed})
        sums = np.array([sum(value for row, value in zip(executed, differences, strict=True)
                             if row['board_seed'] == board) for board in boards])
        counts = np.array([sum(row['board_seed'] == board for row in executed) for board in boards])
        interval = None
        if len(boards) > 1:
            rng = np.random.default_rng(0)
            draw = rng.integers(len(boards), size=(10000, len(boards)))
            means = sums[draw].sum(-1) / counts[draw].sum(-1)
            interval = [float(value) for value in np.quantile(means, [.025, .975])]
        result['false_association_effect_swap_minus_sham'] = dict(mean=float(differences.mean()),
            ci95=interval, board_clusters=len(boards),
            uncertainty='Bootstrap over board seeds retaining all collected states per sampled board; no between-board interval with one board; not independent training runs')
        errors = np.array([error for row in executed for error in
                           row['streaming_parity'].get('absolute_errors', [row['streaming_parity']['absolute_error']])])
        strict = bool(errors.mean() <= .01 and np.quantile(errors, .99) <= .05)
        result['streaming_teacher_parity'] = dict(mean_abs=float(errors.mean()), p99=float(np.quantile(errors, .99)),
            status='GO' if strict else 'HOLD', scope='All sampled tokens in each executed source turn')
        drop = np.mean([row['sham']['probability_correct_partner'] - row['swap']['probability_correct_partner'] for row in executed])
        effect = bool(len(executed) >= 32 and len(executed) == qualified
            and strict and interval is not None and interval[0] > 0 and drop > 0)
        result['causal_textual_memory_established'] = effect and result['state_boundaries'] == ['before_thinking']
        result['causal_action_cursor_text_usage_established'] = effect and result['state_boundaries'] == ['first_coordinate']
    return result


@torch.no_grad()
@torch.autocast('cuda', dtype=torch.bfloat16)
def audit_streaming(base, record, state):
    turn = record['turns'][state['response_index']]
    end = turn['end']
    ids = torch.tensor(record['tokens'][:end], device=base.device)
    positions = torch.arange(end, device=base.device)
    teacher = NativeTeacher(base, state['window'], backend='fa4')
    hidden = teacher.segment(ids, positions, positions, state['prefix_length'])
    sampled = [position for position in range(turn['start'], end) if record['roles'][position] == 'assistant']
    if not sampled or any(record['old_tempered_logps'][position] is None for position in sampled):
        raise ValueError('The intervention source turn has missing sampled token likelihoods')
    selected = torch.tensor(sampled, device=base.device)
    logits = teacher.linear(hidden[selected - 1], base.lm_head.weight).float() / .6
    actual = logits.log_softmax(-1).gather(1, ids[selected, None])[:, 0].cpu().tolist()
    expected = [record['old_tempered_logps'][position] for position in sampled]
    errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
    return dict(absolute_error=max(errors), absolute_errors=errors, sampled_positions=sampled,
                query_position=state['decision_position'] - 1,
                scope='Full-history native-effective PS teacher versus every recorded native-vLLM sample in the turn')


def main():
    from transformers import AutoTokenizer
    from .policy import load_model
    from .rng_overlap import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--states', required=True)
    parser.add_argument('--reference', required=True, help='Original pinned tokenizer and generation configuration snapshot')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary-output', help='Separate worker summary when disjoint states share a result directory')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('Use at least one stochastic sample in addition to the greedy branch')
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / 'experiment.json').read_text())
    if metadata['policy_version'] <= 0 or metadata['window'] != 1024:
        raise ValueError('Use a selected trained PS checkpoint at the frozen window')
    if Path(args.reference).name != '768f209d9ea81521153ed38c47d515654e938aea':
        raise ValueError('Use the immutable original tokenizer/EOS snapshot')
    if metadata['model_revision'] != Path(args.reference).name:
        raise ValueError('The selected checkpoint and reference tokenizer revisions differ')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output / 'summary.json'
    with gzip.open(args.states, 'rt') as handle:
        states = [json.loads(line) for line in handle if line.strip()]
    if len({state['state_id'] for state in states}) != len(states):
        raise ValueError('Intervention state IDs must be unique')
    results, pending = [], []
    for state in states:
        if not 4000000 <= state['board_seed'] <= 4000511:
            raise ValueError('Use only the separate intervention board split')
        identity = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        path = output / (state['state_id'] + '.json')
        if path.exists():
            row = json.loads(path.read_text())
            if (row['checkpoint'] != str(checkpoint.resolve()) or row['samples'] != args.samples
                    or row.get('state_sha256') != identity):
                raise ValueError('An existing intervention belongs to a different checkpoint or sample count')
            results.append(row)
        else:
            pending.append(state)
    parse_coord, env_type, invalid_type = native_modules(args.source)
    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    generation = json.loads((Path(args.reference) / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    base = load_model(str(checkpoint), args.device) if any('edits' in state for state in pending) else None
    for state in pending:
        started = time.perf_counter()
        row = dict(state, checkpoint=str(checkpoint.resolve()), samples=args.samples,
                   state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest())
        if 'edits' in state:
            if edit_size(state['edits']) != edit_size(state['sham_edits']):
                raise ValueError('The false-association and sham edits have different sizes')
            token_path = Path(state['token_record'])
            if hashlib.sha256(token_path.read_bytes()).hexdigest() != state['token_record_sha256']:
                raise ValueError('Intervention source tokens changed')
            with gzip.open(token_path, 'rt') as handle:
                record = json.load(handle)
            validate_source(record, state, metadata, checkpoint)
            end, prefix_length = state['decision_position'], state['prefix_length']
            positions = list(range(prefix_length)) + list(range(max(prefix_length, end - state['window']), end))
            if (state['retained_positions'] != positions
                    or state['retained_token_ids'] != [record['tokens'][position] for position in positions]
                    or state['retained_roles'] != [record['roles'][position] for position in positions]):
                raise ValueError('Cold branches differ from the exact immutable prefix and retained original suffix')
            with (gzip.open if Path(state['native_game_json']).suffix == '.gz' else open)(Path(state['native_game_json']), 'rt') as handle:
                game = json.load(handle)
            environment = replay_state(state, game, env_type, invalid_type)
            turn = record['turns'][state['response_index']]
            row['streaming_parity'] = audit_streaming(base, record, state)
            seed = int.from_bytes(hashlib.sha256(f'intervention-v1:{state["state_id"]}'.encode()).digest()[:8], 'little') % (2**63 - 1)
            for branch, changes in [('unedited', []), ('swap', state['edits']), ('sham', state['sham_edits'])]:
                tokens = apply_edits(state, changes)
                if state.get('state_boundary') == 'before_thinking':
                    row[branch] = continue_turn(base, tokenizer, eos, state, tokens, record,
                        samples=args.samples, rng_seed=seed, parse_coord=parse_coord,
                        environment=environment, invalid_type=invalid_type)
                    continue
                prefix_ids = list(record['tokens'][turn['start']:end])
                for edit in changes:
                    if edit['start'] >= turn['start']:
                        prefix_ids[edit['start'] - turn['start']:edit['end'] - turn['start']] = edit['new']
                prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                final_prefix = prefix.rsplit('</think>', 1)[-1]
                row[branch] = continue_branch(base, tokenizer, eos, state, tokens, final_prefix,
                    samples=args.samples, rng_seed=seed, parse_coord=parse_coord,
                    environment=environment, invalid_type=invalid_type)
        row['seconds'] = time.perf_counter() - started
        write_json(output / (state['state_id'] + '.json'), row)
        results.append(row)
        write_json(summary_path, summarize_results(states, results))
        print(json.dumps(dict(state_id=state['state_id'], completed=len(results), qualified=len(states))), flush=True)
    write_json(summary_path, summarize_results(states, results))


if __name__ == '__main__':
    main()
