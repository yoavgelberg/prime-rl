import pytest

from .baseline_memory_report import summarize_rows


def records():
    rows = []
    for attention in ('full', 'ps'):
        for seed in range(1000000, 1000064):
            bins = {name:dict(opportunities=0, correct=0, utilization=None) for name in ('<=W', '(W,2W]', '>2W')}
            bins['>2W'] = dict(opportunities=10 if attention == 'full' else 20,
                correct=8 if attention == 'full' else 2, utilization=.8 if attention == 'full' else .1)
            rows.append(dict(attention=attention, board_seed=seed, prefix_sha256='fixed',
                board_identity_sha256=str(seed), known_match=bins, censored=False))
    return rows


def test_baseline_bootstrap_keeps_paired_boards_and_undefined_empty_bins():
    result = summarize_rows(list(reversed(records())))
    assert result['conditions']['full']['known_match']['>2W']['opportunities'] == 640
    assert result['conditions']['ps']['known_match']['>2W']['utilization'] == .1
    assert result['paired_utilization_difference']['>2W']['full_minus_ps'] == pytest.approx(.7)
    assert result['paired_utilization_difference']['>2W']['ci95'] == pytest.approx([.7, .7])
    assert result['paired_utilization_difference']['<=W']['defined_resamples'] == 0
    assert result['paired_utilization_difference']['<=W']['ci95'] is None


def test_baseline_aggregation_rejects_different_board_or_prefix_and_missing_boards():
    rows = records()
    rows[-1]['board_identity_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='identical native board'):
        summarize_rows(rows)
    rows = records(); rows[-1]['prefix_sha256'] = 'different'
    with pytest.raises(ValueError, match='system prefixes differ'):
        summarize_rows(rows)
    with pytest.raises(ValueError, match='all64 paired'):
        summarize_rows(records()[:-1])
