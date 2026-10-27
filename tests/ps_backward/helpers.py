import types

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from prime_rl.trainer.ps_backward.model import QwenPS
from prime_rl.trainer.ps_backward.types import Trace


def tiny_model(kind="qwen3", depth=3, window=3, dtype=torch.float64, checkpoint_layers=False):
    torch.manual_seed(310)
    config_type, model_type = (Qwen3Config, Qwen3ForCausalLM) if kind == "qwen3" else (Qwen2Config, Qwen2ForCausalLM)
    config = config_type(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        attention_dropout=0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    base = model_type(config).to(dtype)
    if dtype == torch.float64:
        # HF RMSNorm unconditionally casts to FP32, even on a double model.
        # Keep the algebra reference entirely FP64 instead of relaxing its tolerance.
        def double_norm(self, hidden):
            return self.weight * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.variance_epsilon)

        for module in base.modules():
            if hasattr(module, "variance_epsilon"):
                module.forward = types.MethodType(double_norm, module)
    return QwenPS(base, window, backend="dense", checkpoint_layers=checkpoint_layers)


def make_trace(prefix=4, length=19, *, clipped=False, masked=False):
    generator = torch.Generator().manual_seed(170)
    ids = torch.randint(0, 41, (prefix + length,), generator=generator)
    advantages = torch.randn(length, generator=generator, dtype=torch.float64)
    mask = torch.arange(length) % 3 != 1 if masked else None
    old = torch.full((length,), -3.8, dtype=torch.float64) if clipped else None
    # An offset verifies that replay/cropping does not rebase RoPE positions.
    return Trace(ids, prefix, torch.arange(prefix + length) + 123, advantages, old, mask)


def gradients(model):
    return {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}


def assert_grads(actual, expected, rtol=1e-8, atol=1e-10):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name], expected[name], rtol=rtol, atol=atol, msg=lambda msg: name + ": " + msg
        )
