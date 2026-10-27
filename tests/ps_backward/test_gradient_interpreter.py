import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="Run in a separate Triton interpreter process")
@pytest.mark.parametrize("widths", [(7,), (3, 2, 2), (65, 31, 31)])
def test_direct_kernel_pointer_indexing_in_cpu_interpreter(widths):
    from prime_rl.trainer.ps_backward.gradient_accumulation_kernel import accumulate

    torch.manual_seed(7)
    x, dy = torch.randn(37, 79), torch.randn(37, sum(widths))
    buffers = [torch.randn(width, 79) for width in widths]
    initial = [value.clone() for value in buffers]
    pointers = torch.tensor([value.data_ptr() for value in buffers], dtype=torch.int64)
    accumulate(dy, x, pointers, tuple(range(len(widths))), widths)
    expected = (dy.T @ x).split(widths)
    for actual, base, delta in zip(buffers, initial, expected, strict=True):
        torch.testing.assert_close(actual, base + delta, atol=2e-5, rtol=2e-5)
