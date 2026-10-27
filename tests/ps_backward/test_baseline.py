import pytest
import torch
from helpers import assert_grads, gradients, make_trace, tiny_model

from prime_rl.trainer.ps_backward.baseline import backward_full, backward_truncated
from prime_rl.trainer.ps_backward.loss import LossConfig


@pytest.mark.parametrize("kind", ["qwen2", "qwen3"])
def test_checkpointed_full_baseline_preserves_gradients(kind):
    model = tiny_model(kind=kind)
    trace = make_trace(clipped=True, masked=True)
    expected = backward_full(model, trace, LossConfig(head_block_tokens=3))
    reference = gradients(model)
    model.zero_grad(set_to_none=True)
    model.checkpoint_layers, model.retain_last_layers = True, 1
    actual = backward_full(model, trace, LossConfig(head_block_tokens=3))
    assert actual.loss == pytest.approx(expected.loss, abs=1e-12)
    assert_grads(gradients(model), reference)


def test_truncation_is_exact_when_the_whole_completion_fits_one_window():
    model = tiny_model(window=8)
    trace = make_trace(length=5)
    expected = backward_full(model, trace)
    reference = gradients(model)
    model.zero_grad(set_to_none=True)
    actual = backward_truncated(model, trace)
    assert actual.loss == pytest.approx(expected.loss, abs=1e-12)
    assert_grads(gradients(model), reference)


def test_truncation_omits_earlier_labels_on_a_long_trace():
    model = tiny_model(window=4)
    trace = make_trace(length=31)
    full = backward_full(model, trace)
    reference = gradients(model)
    model.zero_grad(set_to_none=True)
    truncated = backward_truncated(model, trace)
    assert (full.loss_tokens, truncated.loss_tokens) == (31, 4)
    assert any(not torch.allclose(p.grad, reference[n]) for n, p in model.named_parameters())
