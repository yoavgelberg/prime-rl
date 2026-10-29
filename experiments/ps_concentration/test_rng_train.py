import json
from dataclasses import asdict

import numpy as np
import torch

from .objective import leave_one_out, trajectory_loss
from .rng_train import board_requests, combine_metrics, reconcile_resume, restore_resume
from .rng_training_config import MatchingConfig


def test_global_rollout_plan_is_independent_of_gpu_layout():
    config = MatchingConfig(model='model', source='source')
    permutation = np.random.default_rng(11).permutation(1000000)
    expected = board_requests(permutation, config, 1, 0, 1)
    assert len(expected) == 64
    assert len({row['board_seed'] for row in expected}) == 8
    assert len({row['generation_seed'] for row in expected}) == 64
    for world in (2, 4, 8):
        actual = [row for rank in range(world) for row in board_requests(permutation, config, 1, rank, world)]
        assert actual == expected
    next_update = board_requests(permutation, config, 2, 0, 1)
    assert not {row['board_seed'] for row in expected}.intersection(row['board_seed'] for row in next_update)


def test_distributed_sum_preserves_fixed_all_token_objective():
    config = MatchingConfig(model='model', source='source')
    rewards = np.array([[0., 1., 2., 3., 4., 5., 6., 7.]] * 8)
    advantages = leave_one_out(rewards).ravel()
    parameters = torch.tensor([-1., -2., -3.], requires_grad=True)
    references = parameters.detach().clone()
    whole = sum(trajectory_loss(parameters * (index + 1), references * (index + 1), advantage, config)
                for index, advantage in enumerate(advantages))
    whole.backward()
    expected = parameters.grad.clone()
    combined = torch.zeros_like(expected)
    for rank in range(4):
        local = parameters.detach().clone().requires_grad_()
        loss = sum(trajectory_loss(local * (index + 1), references * (index + 1), advantages[index], config)
                   for index in range(rank * 16, (rank + 1) * 16))
        loss.backward()
        combined += local.grad
    assert torch.allclose(combined, expected, atol=1e-9)
    assert expected.abs().min() > 0
    local = parameters.detach().clone().requires_grad_()
    trajectory_loss(local, references, 2., config).backward()
    assert torch.allclose(local.grad, torch.full_like(local, -2 / (26624 * 64)), atol=1e-10)


def test_resource_endpoints_remain_separate_in_validation_metrics():
    workers = [dict(episodes=2, return_sum=-1., pairs_sum=3, attempts_sum=110, completed=0,
        context_terminations=1, invalid_actions=2, format_errors=1, generated_tokens=1000, raw_transcript_tokens=1200)] * 2
    result = combine_metrics(workers)
    assert result['mean_pairs'] == 1.5
    assert result['mean_return'] == -.5
    assert result['context_terminations'] == 2
    assert result['format_error_rate'] == 2 / 220


def test_resume_archives_uncommitted_work_and_rebuilds_metrics(tmp_path):
    for update in (1, 2):
        folder = tmp_path / f'rollouts/update-{update:04d}'
        folder.mkdir(parents=True)
        (folder / 'tokens.json').write_text('preserve')
        metrics = tmp_path / 'update-metrics'
        metrics.mkdir(exist_ok=True)
        (metrics / f'update-{update:04d}.json').write_text(json.dumps(dict(update=update)))
        (tmp_path / f'checkpoint-{update:04d}').mkdir()
    reconcile_resume(tmp_path, 1)
    assert (tmp_path / 'rollouts/update-0001/tokens.json').read_text() == 'preserve'
    assert not (tmp_path / 'rollouts/update-0002').exists()
    archived = list(tmp_path.glob('recovered-*/rollouts/update-0002/tokens.json'))
    assert len(archived) == 1 and archived[0].read_text() == 'preserve'
    assert (tmp_path / 'checkpoint-0001').exists()
    assert not (tmp_path / 'checkpoint-0002').exists()
    assert json.loads((tmp_path / 'updates.jsonl').read_text()) == dict(update=1)


def test_resume_restores_exact_adam_update(tmp_path):
    config = MatchingConfig(model='model', source='source', context_exhaustion_reward=-1.)
    original = torch.nn.Linear(3, 2)
    first = torch.optim.AdamW(original.parameters(), lr=5e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    inputs = torch.tensor([[1., 2., 3.], [3., 1., -2.]])

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().sum().backward()
        optimizer.step()

    step(original, first)
    path = tmp_path / 'resume.pt'
    torch.save(dict(masters=[p.detach().clone() for p in original.parameters()], optimizer=first.state_dict(),
        reference=config.model, config=asdict(config), update=1, best_return=.3, best_update=0,
        rngs=[dict(torch=torch.get_rng_state(), cuda={})]), path)
    restored = torch.nn.Linear(3, 2)
    restored.device = torch.device('cpu')
    second = torch.optim.AdamW(restored.parameters(), lr=5e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    assert restore_resume(path, restored, second, config, 0, []) == (1, .3, 0)
    for p, q in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(p, q)
    step(original, first)
    step(restored, second)
    for p, q in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(p, q)
