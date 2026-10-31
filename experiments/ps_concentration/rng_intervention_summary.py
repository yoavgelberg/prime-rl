"""Summarize memory interventions without importing a CUDA inference backend."""

import numpy as np


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

