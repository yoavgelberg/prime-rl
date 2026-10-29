import pytest
import torch

from prime_rl.trainer.ps_backward.delayed_reward import RewardGradientGroup


@pytest.mark.parametrize("order", [range(8), reversed(range(8))])
def test_delayed_cpu_rloo_equals_direct_episodic_objective(order):
    rng = torch.Generator().manual_seed(11)
    features = torch.randn(8, 13, dtype=torch.float64, generator=rng)
    rewards = torch.tensor([-.8, -.8, 0., .25, -.4, 1., .5, -.1], dtype=torch.float64)
    parameter = torch.randn(13, dtype=torch.float64, generator=rng, requires_grad=True)
    denominator, global_batch = 26624, 64
    advantage = rewards - (rewards.sum() - rewards) / 7
    losses = (features @ parameter).square()
    expected, = torch.autograd.grad((advantage * losses).sum() / (denominator * global_batch), parameter)
    group = RewardGradientGroup(8, 3, baseline="leave_one_out", device="cpu")
    for index in order:
        # Each gradient is a full trajectory sum; length is not a denominator.
        value = 2 * features[index] * (features[index] @ parameter.detach())
        group.add(index, {"weight": value}, float(rewards[index]), policy_version=3)
    actual = group.finish(policy_version=3, scale=1 / (denominator * global_batch))
    torch.testing.assert_close(actual["weight"], expected, atol=1e-17, rtol=1e-12)
    assert actual["weight"].device.type == "cpu"
    assert not group.gradient_sum


def test_zero_reward_spread_cancels_all_task_gradients():
    group = RewardGradientGroup(8, 0, baseline="leave_one_out", device="cpu")
    for index in range(8):
        group.add(index, {"w": torch.tensor([index, index ** 2], dtype=torch.float64)}, .5, policy_version=0)
    actual = group.finish(policy_version=0)
    torch.testing.assert_close(actual["w"], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)


def test_rloo_rejects_weighted_and_stale_gradients():
    with pytest.raises(ValueError, match="two rollouts"):
        RewardGradientGroup(1, 0, baseline="leave_one_out")
    group = RewardGradientGroup(2, 0, baseline="leave_one_out")
    gradient = {"w": torch.ones(3)}
    with pytest.raises(ValueError, match="unweighted"):
        group.add(0, gradient, 1, policy_version=0, baseline_weight=13)
    with pytest.raises(ValueError, match="policy versions"):
        group.add(0, gradient, 1, policy_version=1)
    for index in range(2):
        group.add(index, gradient, 1, policy_version=0)
    with pytest.raises(ValueError, match="finite"):
        group.finish(policy_version=0, scale=float("nan"))


@pytest.mark.parametrize("temperature", [1., .6])
@pytest.mark.parametrize("all_masked", [False, True])
def test_recorded_head_matches_direct_tempered_objective_with_environment_mask(temperature, all_masked):
    from types import SimpleNamespace

    from prime_rl.trainer.ps_backward.recorded_graphs import RecordedGraphCache

    torch.manual_seed(93)
    norm = torch.nn.RMSNorm(8, eps=1e-6)
    norm.variance_epsilon = norm.eps
    head = torch.nn.Linear(8, 17, bias=False)
    model = SimpleNamespace(
        base=SimpleNamespace(model=SimpleNamespace(norm=norm), lm_head=head),
        recorded_score_temperature=temperature,
        _weight_cast_cache={id(head.weight): (None, head.weight.detach())},
    )
    raw = torch.randn(7, 8, requires_grad=True)
    labels = torch.full((7,), -100) if all_masked else torch.tensor([3, -100, 7, 9, -100, 2, 5])
    normalized = norm(raw)
    logits = head(normalized)
    expected_loss = torch.nn.functional.cross_entropy(logits / temperature, labels, reduction="sum")
    expected = torch.autograd.grad(expected_loss, (raw, norm.weight, head.weight))
    cache = RecordedGraphCache(model, 8)
    loss, dx = cache.head(raw.detach(), normalized.detach(), logits.detach(), labels)
    torch.testing.assert_close(loss, expected_loss.detach(), atol=1e-6, rtol=1e-6)
    for actual, reference in zip((dx, norm.weight.grad, head.weight.grad), expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dx[labels == -100], torch.zeros_like(dx[labels == -100]), rtol=0, atol=0)


def test_fixed_gradient_denominator_does_not_depend_on_generated_length():
    from prime_rl.trainer.ps_backward.recorded_stream import RecordedStream

    outputs = []
    for length in (128, 262144):
        stream = object.__new__(RecordedStream)
        stream.loss_denominator, stream.loss_tokens = 26624, length
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([8., -4.])
        stream.parameters = {"w": parameter}
        stream.normalize_gradients()
        outputs.append(parameter.grad.clone())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], torch.tensor([8., -4.]) / 26624)


@pytest.mark.parametrize("order", [range(8), reversed(range(8))])
def test_delayed_grpo_matches_direct_clipped_surrogate_gradient(order):
    from prime_rl.orchestrator.grpo import normalized_advantages

    rng = torch.Generator().manual_seed(22)
    features = torch.randn(8, 13, dtype=torch.float64, generator=rng)
    rewards = torch.tensor([[-.8, -.8, 0., .25, -.4, 1., .5, -.1]], dtype=torch.float64)
    parameter = torch.randn(13, dtype=torch.float64, generator=rng, requires_grad=True)
    logps = -(features @ parameter).square()
    advantages = normalized_advantages(rewards)[0]
    ratio = (logps - logps.detach()).exp()
    clipped = ratio.clamp(.8, 1.2)
    loss = -torch.minimum(ratio * advantages, clipped * advantages).sum() / (26624 * 64)
    expected, = torch.autograd.grad(loss, parameter)
    group = RewardGradientGroup(8, 3, baseline="grpo", device="cpu")
    for index in order:
        score = 2 * features[index] * (features[index] @ parameter.detach())
        group.add(index, {"w": score}, float(rewards[0, index]), policy_version=3)
    actual = group.finish(policy_version=3, scale=1 / (26624 * 64))
    torch.testing.assert_close(actual['w'], expected, atol=1e-17, rtol=1e-12)
    assert group.reward_std == pytest.approx(float(rewards.std(correction=1)), abs=1e-15)


def test_identical_rewards_have_exactly_zero_grpo_advantage_and_gradient():
    from prime_rl.orchestrator.grpo import normalized_advantages

    rewards = torch.full((2, 8), .1, dtype=torch.float64)
    assert not normalized_advantages(rewards).count_nonzero()
    group = RewardGradientGroup(8, 0, baseline="grpo")
    for index in range(8):
        group.add(index, {"w": torch.tensor([index * .13], dtype=torch.float64)}, .1, policy_version=0)
    assert not group.finish(policy_version=0)['w'].count_nonzero()
