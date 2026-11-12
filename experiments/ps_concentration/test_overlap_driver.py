import asyncio
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.ps_concentration.overlap_config import BoardProtocol, RunConfig, board_protocol, cohorts, episode_metrics, evaluation_resources, evaluation_specs, permutation, training_specs
from experiments.ps_concentration import rng_overlap as driver


def config(tmp_path, **kwargs):
    model = tmp_path / '768f209d9ea81521153ed38c47d515654e938aea'
    model.mkdir(exist_ok=True)
    (model / 'config.json').write_text(json.dumps(dict(hidden_size=2560, num_attention_heads=32,
        head_dim=128, num_hidden_layers=36, num_key_value_heads=8)))
    return RunConfig(model=str(model), source=str(tmp_path / 'native'), output=str(tmp_path / 'run'),
        source_sha256='a' * 64, **kwargs)


def row(spec):
    return dict(spec, score=8, done=True, training_return=.75, response_count=24,
        generated_tokens=100, raw_transcript_tokens=200, invalid_actions=0, format_errors=0,
        seconds=2., censored=False, entropy_tokens=100, mean_policy_entropy_at_temperature=1.5)


def test_board_groups_and_generation_rngs_survive_physical_layout_changes(tmp_path):
    cfg = config(tmp_path)
    boards = permutation(11)
    baseline = training_specs(cfg, 17, boards)
    assert len({item['generation_seed'] for item in baseline}) == 64
    for gpus in (1, 2, 4, 8):
        current = training_specs(replace(cfg, gpus=gpus), 17, boards)
        assert [{key: value for key, value in item.items() if key != 'rank'} for item in current] == [
            {key: value for key, value in item.items() if key != 'rank'} for item in baseline]
        batches = list(cohorts(current, gpus, 4))
        assert sum(map(len, batches)) == 64
        for batch in batches:
            assert all(sum(item['rank'] == rank for item in batch) == 4 for rank in range(gpus))
        assert all(len({item['board_seed'] for item in current if item['board_index'] == group}) == 1
                   for group in range(8))
    assert not set(boards[:256 * 8]) & set(range(1000000, 1000064))
    with pytest.raises(ValueError, match='reserved split'):
        evaluation_specs(cfg, 'validation', 'bad', count=129)


def test_engine_memory_and_position_configuration(tmp_path):
    cfg = config(tmp_path)
    ps = driver.engine_options(cfg)
    assert ps['max_num_seqs'] == 4
    assert ps['kv_cache_memory_bytes'] == 2 * 1024**3
    assert ps['cudagraph_capture_sizes'] == [1, 2, 4]
    assert ps['additional_config']['ps_streaming']['max_requests'] == 4
    assert ps['additional_config']['ps_streaming']['loss_denominator'] == 26624
    assert ps['additional_config']['ps_streaming']['weight_gradient_backend'] == 'cublas'
    assert ps['hf_overrides'] == dict(use_sliding_window=True, sliding_window=1024)
    assert 'rope_scaling' not in ps['hf_overrides'] and 'max_position_embeddings' not in ps['hf_overrides']
    evaluation = driver.engine_options(cfg, evaluation=True)
    assert evaluation['max_num_seqs'] == 16
    assert evaluation['kv_cache_memory_bytes'] > ps['kv_cache_memory_bytes']
    full = driver.engine_options(replace(cfg, evaluation_concurrency=4), attention='full', evaluation=True)
    assert full['kv_cache_memory_bytes'] < 145 * 1024**3
    assert full['attention_backend'] == 'CUSTOM'
    assert full['hf_overrides']['sliding_window'] == cfg.transcript_cap
    assert full['additional_config']['ps_streaming']['window'] == cfg.transcript_cap
    with pytest.raises(ValueError, match='four live'):
        driver.engine_options(cfg, attention='full', evaluation=True)


def test_audited_full_resources_preserve_policy_and_seed_plan(tmp_path):
    cfg = config(tmp_path, evaluation_concurrency=4, thinking_token_limit=128,
        action_token_limit=32, thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n')
    current, bound = evaluation_resources(cfg, 4, mode='evaluate', attention='full',
        concurrency=16, full_kv_token_bound=53568)
    assert current.scientific_identity() == cfg.scientific_identity()
    original = evaluation_specs(cfg, 'test', 'full-u0', count=128)
    changed = evaluation_specs(current, 'test', 'full-u0', count=128)
    assert [{k: v for k, v in row.items() if k != 'rank'} for row in original] == [
        {k: v for k, v in row.items() if k != 'rank'} for row in changed]
    options = driver.engine_options(current, attention='full', evaluation=True, full_kv_token_bound=bound)
    assert options['max_num_seqs'] == 16
    assert options['kv_cache_memory_bytes'] == 126835752960
    assert options['max_num_batched_tokens'] == 1024
    assert options['max_model_len'] == cfg.transcript_cap == 262144
    assert options['hf_overrides']['sliding_window'] == cfg.transcript_cap
    assert options['additional_config']['ps_streaming']['window'] == cfg.transcript_cap
    assert options['cudagraph_capture_sizes'] == [1, 2, 4, 8, 16]
    assert options['kv_cache_memory_bytes'] < driver.engine_options(cfg, attention='full', evaluation=True)['kv_cache_memory_bytes']


def test_small_full_layout_still_supports_one_declared_model_context(tmp_path):
    cfg = config(tmp_path, evaluation_concurrency=1, thinking_token_limit=128,
        action_token_limit=32, thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n')
    options = driver.engine_options(cfg, attention='full', evaluation=True, full_kv_token_bound=53568)
    assert options['max_model_len'] == 262144 and options['max_num_seqs'] == 1
    assert options['kv_cache_memory_bytes'] == 38673580032


@pytest.mark.parametrize('change', [dict(mode='train'), dict(attention='ps'), dict(concurrency=32),
    dict(full_kv_token_bound=53569), dict(concurrency=None), dict(full_kv_token_bound=None)])
def test_full_resource_override_rejects_unaudited_layouts(tmp_path, change):
    cfg = config(tmp_path, thinking_token_limit=128, action_token_limit=32,
        thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n')
    options = dict(mode='evaluate', attention='full', concurrency=16, full_kv_token_bound=53568)
    with pytest.raises(ValueError, match='full-evaluation'):
        evaluation_resources(cfg, None, **(options | change))


@pytest.mark.parametrize('change', [dict(thinking_token_limit=None, thinking_budget_transition=None), dict(thinking_token_limit=129),
    dict(action_token_limit=64), dict(thinking_budget_transition=None)])
def test_full_resource_bound_cannot_be_used_with_another_controller(tmp_path, change):
    cfg = config(tmp_path, thinking_token_limit=128, action_token_limit=32,
        thinking_budget_transition='\n\nOkay, time to give the answer.</think>\n\n')
    with pytest.raises(ValueError, match='128/32-token controller'):
        evaluation_resources(replace(cfg, **change), None, mode='evaluate', attention='full',
            concurrency=16, full_kv_token_bound=53568)


def test_larger_split_doubles_cards_and_attempts_without_changing_training(tmp_path):
    cfg = config(tmp_path)
    larger = board_protocol('larger')
    assert larger.identity() == dict(board_rows=4, board_cols=8, distinct_ranks=8, total_pairs=16, max_responses=192)
    assert board_protocol('validation') == BoardProtocol()
    specs = evaluation_specs(cfg, 'larger', 'larger-test')
    assert [item['board_seed'] for item in specs] == list(range(3000000, 3000256))
    options = driver.worker_options(cfg, specs, tmp_path, greedy=True, board=larger)
    assert (options['rows'], options['cols'], options['num_cards'], options['max_responses']) == (4, 8, 8, 192)
    assert options['transcript_cap'] == cfg.transcript_cap and options['window'] == cfg.window
    assert driver.worker_options(cfg, [], tmp_path)['max_responses'] == 96
    assert cfg.scientific_identity()['reward'] == 'pairs/8-attempts/96; resource exhaustion -1'
    observed = dict(row(specs[0]), score=8, total_pairs=16, max_responses=192)
    assert episode_metrics([observed])['matched_fraction'] == .5
    with pytest.raises(ValueError, match='mix board sizes'):
        episode_metrics([observed, row(specs[1])])
    with pytest.raises(ValueError, match='frozen training board'):
        BoardProtocol(cols=8)


def test_larger_evaluation_records_effective_limits_and_complete_reserved_split(tmp_path):
    cfg = config(tmp_path)
    batches = []

    class Runner:
        async def collect(self, factory, options, **kwargs):
            assert options['greedy'] is True and options['max_responses'] == 192
            assert options['cols'] == 8 and options['num_cards'] == 8
            batches.append(options['specs'])
            return dict(episodes=[dict(row(spec), done=False, training_return=.375) for spec in options['specs']],
                        seconds=.1, devices=[])

        async def release(self):
            pass

    output = tmp_path / 'larger-evaluation'
    result = asyncio.run(driver.evaluate_games(cfg, Runner(), split='larger', output=output, label='larger'))
    assert result['episodes'] == 256 and result['total_pairs'] == 16 and result['max_responses'] == 192
    assert result['matched_fraction'] == .5 and result['mean_return'] == .375
    assert len(batches) == 2
    protocol = json.loads((output / 'evaluation-protocol.json').read_text())
    assert protocol['board_cols'] == 8 and protocol['max_responses'] == 192
    assert protocol['config']['max_responses'] == 96 and protocol['transcript_cap'] == 262144
    with pytest.raises(ValueError, match='Size generalization'):
        asyncio.run(driver.evaluate_games(cfg, Runner(), split='larger', output=tmp_path / 'bad-larger',
                                         label='larger-full', attention='full'))


@pytest.mark.parametrize('defer_evaluation', [False, True])
def test_one_fresh_update_saves_before_validation_and_uses_every_trajectory(tmp_path, monkeypatch, defer_evaluation):
    cfg = config(tmp_path, updates=1)
    output = Path(cfg.output)
    output.mkdir()
    calls, admitted, logged = [], [], []

    class Runner:
        def __init__(self, engine):
            pass

        async def collect(self, factory, options, *, backward=False, policy_version=0):
            specs = options['specs']
            calls.append(('collect', backward, policy_version))
            if backward:
                admitted.extend(specs)
            return dict(episodes=[row(spec) for spec in specs], seconds=.1,
                backward=[dict(peak_allocated_bytes=123, peak_reserved_bytes=456)] * cfg.gpus if backward else None)

        async def release(self):
            calls.append(('release',))

    async def rpc(engine, method, *, kwargs=None, timeout=600.):
        calls.append((method, kwargs))
        if method == 'ps_train_initialize':
            return [dict(world_size=cfg.gpus)] * cfg.gpus
        if method == 'ps_train_save':
            return [dict(policy_version=len(admitted) // 64)] * cfg.gpus
        if method == 'ps_train_materialize':
            return [dict(advantages=[-1., 1.] * 4, zero_advantage_groups=0,
                normalized_score_surrogate_loss=.01, clipped_grpo_surrogate_at_origin=0.)] * cfg.gpus
        if method == 'ps_train_save_pending':
            return [dict(policy_version=0)] * cfg.gpus
        if method == 'ps_train_reference_kl':
            return [dict(kl_sum=0., normalized_kl=0., generated_tokens=800,
                         parity=dict(mean_abs=.001, p99=.01, status='GO'))] * cfg.gpus
        if method == 'ps_train_step':
            assert len(admitted) == 64
            return [dict(policy_version=1, learning_rate=5e-6 / 8, gradient_norm=.5,
                         weight_update_norm=.001, gradient_clipped=False)] * cfg.gpus
        if method == 'ps_train_reduce':
            return [dict(episodes=4)] * cfg.gpus
        raise AssertionError(method)

    monkeypatch.setattr(driver, 'GameRolloutRunner', Runner)
    monkeypatch.setattr(driver, 'worker_rpc', rpc)
    monkeypatch.setattr(driver, 'handoff_requested', lambda: False)
    monitor = type('Monitor', (), dict(log=lambda self, value: logged.append(value)))()
    competence = None if defer_evaluation else dict(matched_fraction=.9, mean_return=.75, episodes=128, split='validation')
    asyncio.run(driver.train(cfg, object(), monitor, competence, defer_evaluation=defer_evaluation))
    step = next(index for index, call in enumerate(calls) if call[0] == 'ps_train_step')
    save = next(index for index in range(step + 1, len(calls)) if calls[index][0] == 'ps_train_save')
    assert step < save
    if defer_evaluation:
        assert not any(call[:2] == ('collect', False) for call in calls)
    else:
        val = next(index for index in range(step + 1, len(calls)) if calls[index][:2] == ('collect', False))
        assert save < val
    assert len({spec['episode_id'] for spec in admitted}) == 64
    assert sum(call[0] == 'ps_train_step' for call in calls) == 1
    assert sum(call[0] == 'ps_train_reduce' for call in calls) == 2
    final = json.loads((output / 'pilot-complete.json').read_text())
    assert final['episodes'] == 64 and final['best_update'] == (None if defer_evaluation else 0)
    assert final['evaluation_deferred'] == defer_evaluation
    assert len(logged) == (1 if defer_evaluation else 3)
    metric = logged[0 if defer_evaluation else 1]
    assert metric['train/generated_tokens'] == 6400
    assert metric['train/normalized_score_surrogate_loss'] == pytest.approx(.08)
    assert metric['train/advantage_std'] == pytest.approx(1.)


def test_orphan_future_rollouts_are_quarantined_on_resume(tmp_path):
    for name in ('rollouts', 'validation', 'update-metrics', 'parity', 'rollout-plans'):
        (tmp_path / name).mkdir()
        for update in (16, 17):
            (tmp_path / name / f'update-{update:04d}').mkdir()
    (tmp_path / 'checkpoints/policy-0017.interrupted-123').mkdir(parents=True)
    driver.recover_uncommitted(tmp_path, 16)
    assert (tmp_path / 'rollouts/update-0016').exists()
    assert not (tmp_path / 'rollouts/update-0017').exists()
    assert len(list(tmp_path.glob('recovered-*'))) == 1


def test_partial_or_repeated_batches_cannot_advance_an_update(tmp_path):
    cfg = config(tmp_path)
    specs = training_specs(cfg, 1, permutation(11))
    duplicate = [row(spec) for spec in specs[:-1]]
    duplicate.append(duplicate[-1])
    with pytest.raises(ValueError, match='64 distinct'):
        driver.aggregate_update([dict(episodes=duplicate)], [], [], [], update=1, seconds=1., gpus=8)


def test_pilot_reports_exact_gate_values_without_changing_window_or_hiding_hold():
    full = dict(matched_fraction=.95)
    initial = dict(matched_fraction=.2, mean_return=-.6)
    trained = dict(mean_return=-.5, format_error_rate=.005)
    gate = driver.pilot_gate(full, initial, trained)
    assert gate['status'] == 'GO'
    assert gate['initial_matched_fraction_gap'] == pytest.approx(.75)
    assert gate['ps_return_improvement'] == pytest.approx(.1)
    held = driver.pilot_gate(full, initial, dict(trained, mean_return=-.58))
    assert held['status'] == 'HOLD' and held['ps_return_improvement'] == pytest.approx(.02)
