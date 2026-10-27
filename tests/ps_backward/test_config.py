import pytest

from prime_rl.trainer.ps_backward.config import StreamConfig


@pytest.mark.parametrize(
    "options",
    [
        dict(window=0),
        dict(chunk_tokens=-1),
        dict(max_requests=True),
        dict(ring_tokens=1000),
        dict(tail_block_tokens=512),
        dict(fp32_attention="false"),
        dict(joint_qkv="false"),
        dict(copy_mode="automatic"),
        dict(copy_programs=-1),
        dict(gpu_stages_in_flight=0),
        dict(backward_batch_tokens=0),
        dict(weight_gradient_backend="automatic"),
    ],
)
def test_invalid_storage_and_precision_configs_fail_early(options):
    with pytest.raises(ValueError):
        StreamConfig(**options)


def test_private_fa4_api_is_pinned(monkeypatch):
    import prime_rl.trainer.ps_backward.config as config

    monkeypatch.setattr(config, "version", lambda _: "future-release")
    with pytest.raises(ValueError, match="pinned"):
        StreamConfig().validate_hardware("cuda", query_heads=4, kv_heads=2, head_dim=64)


@pytest.mark.parametrize("major", [9, 10, 11])
def test_fp32_attention_accepts_supported_accumulator_architectures(monkeypatch, major):
    import torch

    import prime_rl.trainer.ps_backward.config as config

    monkeypatch.setattr(config, "version", lambda _: "4.0.0b36")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (major, 0))
    StreamConfig().validate_hardware("cuda", query_heads=4, kv_heads=2, head_dim=128)
