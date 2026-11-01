from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from .hero_pair_plan import prepare
from .overlap_config import RunConfig, evaluation_specs
from .rng_select import argument
from .test_hero_v2 import config


def test_paired_plan_keeps_five_windows_and_identical_validation_boards(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'resolved_config.json').write_text(json.dumps(asdict(config())))
    out = tmp_path / 'paired'
    plan = prepare(parent, out, source_root='/captured', source_sha256='b' * 64)
    assert plan['parent_barrier'] == 128
    assert [(r['window'], r['additional_updates'], r['additional_episodes']) for r in plan['branches']] == [
        (1024, 64, 8192), (256, 64, 8192)]
    assert len(plan['initial_sweep']['8']) == 7
    assert {r['window'] for r in plan['initial_sweep']['8'] if r['attention'] == 'ps'} == {64, 128, 256, 512, 1024}
    for gpus, entries in plan['initial_sweep'].items():
        for row in entries:
            path = Path(argument(row['argv'], '--config'))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['config_sha256']
            cfg = RunConfig(**json.loads(path.read_text()))
            specs = evaluation_specs(cfg, 'validation', 'fixture', count=32)
            assert [x['board_seed'] for x in specs] == list(range(1100000, 1100032))
            assert cfg.gpus == int(gpus) and cfg.window == row['window']
            assert cfg.thinking_token_limit == 128 and cfg.action_token_limit == 32
    assert prepare(parent, out, source_root='/captured', source_sha256='b' * 64) == plan
    with pytest.raises(ValueError, match='frozen paired plan'):
        prepare(parent, out, source_root='/changed', source_sha256='c' * 64)
