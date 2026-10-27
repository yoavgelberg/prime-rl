import pytest
import torch

from prime_rl.trainer.ps_backward.linear import fp32grad_linear


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_cast_supports_independent_backwards_with_fp32_weight_and_input_gradients():
    torch.manual_seed(816)
    weight = torch.randn(64, 128, device="cuda", requires_grad=True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    cast = weight.detach().bfloat16()
    expected_weight = torch.zeros_like(weight, dtype=torch.float64)
    expected_bias = torch.zeros_like(bias, dtype=torch.float64)
    for rows in (93, 128, 13):
        x = torch.randn(1, rows, 128, device="cuda", requires_grad=True)
        dy = torch.randn(1, rows, 64, device="cuda", dtype=torch.bfloat16)
        actual = fp32grad_linear(x, weight, bias, cast)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native = torch.nn.functional.linear(x, weight, bias)
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        actual.backward(dy)
        torch.testing.assert_close(x.grad.double(), dy.double() @ cast.double(), rtol=2e-5, atol=2e-5)
        expected_weight += dy.squeeze(0).double().T @ x.detach().bfloat16().squeeze(0).double()
        expected_bias += dy.squeeze(0).double().sum(0)
    torch.testing.assert_close(weight.grad.double(), expected_weight, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(bias.grad.double(), expected_bias, rtol=2e-5, atol=2e-5)
