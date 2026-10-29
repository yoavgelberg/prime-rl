import numpy as np
import torch

from .config import Config
from .objective import leave_one_out, trajectory_loss


def test_leave_one_out_uses_other_seven_without_standardization():
    returns = np.array([[1., 0., 0., 0., 0., 0., 0., 0.], [2.]*8])
    advantage = leave_one_out(returns)
    np.testing.assert_allclose(advantage[0], [1., *([-1/7]*7)])
    assert np.array_equal(advantage[1], np.zeros(8))
    assert np.array_equal(leave_one_out([[-1.0000000000000009]*8]),np.zeros((1,8)))


def test_early_scratch_gradient_and_fixed_token_denominator():
    config = Config(kl_coefficient=0.)
    # First and last entries represent sampled early scratch and a late action.
    parameters = torch.zeros(4, requires_grad=True)
    selected = parameters[[0, 3]]
    loss = trajectory_loss(selected, selected.detach(), 1., config)
    loss.backward()
    expected = -1/(26624*64)
    torch.testing.assert_close(parameters.grad, torch.tensor([expected, 0., 0., expected]))


def test_zero_advantage_reference_equal_policy_has_zero_gradient():
    logps = torch.tensor([-.1, -1.2, -3.5], requires_grad=True)
    loss = trajectory_loss(logps, logps.detach().clone(), 0., Config())
    loss.backward()
    torch.testing.assert_close(logps.grad, torch.zeros(3))
