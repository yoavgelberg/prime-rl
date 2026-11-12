from dataclasses import asdict, replace

import pytest

from .hero_config import MODEL_REVISION, HeroConfig


def config():
    return HeroConfig(
        model=f"/models/{MODEL_REVISION}",
        source="/native",
        output="/run",
        node_dir="/node/control",
        source_sha256="a" * 64,
        deadline=10000,
    )


def test_config_round_trip_preserves_existing_checkpoint_schema():
    original = config()
    assert HeroConfig(**asdict(original)) == original
    from .hero_v2 import HeroConfig as compatible_import

    assert compatible_import is HeroConfig


@pytest.mark.parametrize(
    "name",
    [
        "updates",
        "denominator",
        "transcript_cap",
        "warmup_updates",
        "checkpoint_interval",
        "export_interval",
        "head_block_tokens",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_counts_fail_before_workers_start(name, value):
    with pytest.raises(ValueError, match="positive integers"):
        replace(config(), **{name: value})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_temperature_is_rejected(value):
    with pytest.raises(ValueError, match="temperature"):
        replace(config(), temperature=value)


def test_training_cannot_overrun_the_board_pool():
    assert replace(config(), updates=62500).updates == 62500
    with pytest.raises(ValueError, match="board pool"):
        replace(config(), updates=62501)


def test_source_digest_must_be_hexadecimal():
    with pytest.raises(ValueError, match="captured implementation"):
        replace(config(), source_sha256="z" * 64)
