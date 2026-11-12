import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .hero_config import MODEL_REVISION, NATIVE_REVISION
from .launch import configure, main


@pytest.fixture
def launch_args(tmp_path):
    model = tmp_path / "models" / MODEL_REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    source = tmp_path / "native"
    native = source / "1_matching_pairs_new" / "game.py"
    native.parent.mkdir(parents=True)
    native.write_text("# native source fixture\n")
    (source / "SOURCE.json").write_text(
        json.dumps(
            {
                "commit": NATIVE_REVISION,
                "manifest": [
                    {"path": str(native.relative_to(source)), "sha256": hashlib.sha256(native.read_bytes()).hexdigest()}
                ],
            }
        )
    )
    return SimpleNamespace(
        model=model,
        source=source,
        output=tmp_path / "output",
        node_dir=tmp_path / "node" / "control",
        source_sha256="a" * 64,
        deadline=10000.0,
        gpus=8,
        wandb_id="test-only",
    )


def test_preparation_preserves_parent_recipe_and_does_not_start_workers(launch_args):
    config, path = configure(launch_args)
    assert config.batch_size == 128 and config.group_size == 8
    assert config.updates == 128 and config.window == 1024 and config.temperature == 0.6
    assert config.exact_kl_start_update is None
    assert config.thinking_token_limit == 128 and config.action_token_limit == 32
    assert json.loads(path.read_text())["wandb_id"] == "test-only"
    assert not Path(config.node_dir).exists()
    assert configure(launch_args) == (config, path)


def test_prepare_only_cli_runs_without_an_allocation(launch_args, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_PARTITION", raising=False)
    values = vars(launch_args)
    argv = [item for key, value in values.items() for item in ("--" + key.replace("_", "-"), str(value))]
    main([*argv, "--prepare-only"])
    assert (launch_args.output / "launch-config.json").exists()
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_prepare_does_not_overwrite_another_run(launch_args):
    _, path = configure(launch_args)
    before = path.read_bytes()
    launch_args.wandb_id = "another-run"
    with pytest.raises(ValueError, match="differs"):
        configure(launch_args)
    assert path.read_bytes() == before


def test_prepare_rejects_changed_native_source(launch_args):
    (launch_args.source / "1_matching_pairs_new" / "game.py").write_text("changed")
    with pytest.raises(ValueError, match="Native source changed"):
        configure(launch_args)
    assert not launch_args.output.exists()


def test_launch_uses_training_cli_with_worker_cleanup(launch_args, monkeypatch):
    from . import hero_v2

    calls = []
    monkeypatch.setenv("SLURM_JOB_PARTITION", "yejin")
    monkeypatch.setattr(hero_v2, "main", calls.append)
    argv = [item for key, value in vars(launch_args).items() for item in ("--" + key.replace("_", "-"), str(value))]
    main(argv)
    assert calls == [["--config", str(launch_args.output / "launch-config.json")]]
