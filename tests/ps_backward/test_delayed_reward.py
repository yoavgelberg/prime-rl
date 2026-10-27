import pytest
import torch

from prime_rl.trainer.ps_backward.delayed_reward import RewardGradientGroup


@pytest.mark.parametrize("length_weighted", [False, True])
@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2]])
def test_completed_rollouts_can_be_reduced_before_grpo_baseline_is_known(length_weighted, order):
    torch.manual_seed(5)
    rewards = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    lengths = torch.tensor([13.0, 37.0, 7.0, 101.0], dtype=torch.float64)
    gradients = [
        {"weight": torch.randn(5, 7, dtype=torch.float64), "bias": torch.randn(5, dtype=torch.float64)} for _ in rewards
    ]
    baseline_weights = lengths if length_weighted else torch.ones_like(lengths)
    baseline = (rewards * baseline_weights).sum() / baseline_weights.sum()
    expected = {
        name: sum((rewards[i] - baseline) * g[name] for i, g in enumerate(gradients)) / 4 for name in gradients[0]
    }
    group = RewardGradientGroup(4, 17)
    for i in order:
        group.add(i, gradients[i], rewards[i].item(), policy_version=17, baseline_weight=baseline_weights[i].item())
    actual = group.finish(policy_version=17, scale=0.25)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-12, rtol=1e-12)


def test_delayed_gradients_reject_duplicate_and_changed_policy_and_missing_rewards():
    group = RewardGradientGroup(2, 7)
    gradient = {"x": torch.ones(5)}
    with pytest.raises(ValueError, match="policy versions"):
        group.add("a", gradient, 1.0, policy_version=8)
    group.add("a", gradient, 1.0, policy_version=7)
    with pytest.raises(ValueError, match="Duplicate"):
        group.add("a", gradient, 1.0, policy_version=7)
    with pytest.raises(RuntimeError, match="All rewards"):
        group.finish(policy_version=7)
    group.add("b", gradient, 0.0, policy_version=7)
    with pytest.raises(ValueError, match="optimizer policy"):
        group.finish(policy_version=8)
