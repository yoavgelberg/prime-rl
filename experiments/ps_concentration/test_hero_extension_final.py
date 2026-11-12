from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from .hero_extension import extension_config
from .hero_extension_final import prepare
from .hero_protocol import inference_config
from .hero_v2 import identity, write
from .overlap_config import SPLITS, board_protocol, evaluation_specs
from .test_hero_v2 import config


def test_final320_inference_preserves_controller_and_fresh_board_splits():
    hero = replace(config(4), updates=320, window=512, exact_kl_start_update=215)
    inference = inference_config(hero, output='/evaluation', gpus=8)
    assert inference.updates == 320 and inference.window == 512
    assert inference.thinking_token_limit == 128 and inference.action_token_limit == 32
    assert inference.thinking_budget_transition == '\n\nOkay, time to give the answer.</think>\n\n'
    reserved = [set(range(start, start+count)) for start, count in SPLITS.values()]
    assert sum(map(len, reserved)) == len(set.union(*reserved))
    for split in ('extension_test', 'extension_larger'):
        rows = evaluation_specs(inference, split, 'fixed')
        assert [r['board_seed'] for r in rows] == list(range(SPLITS[split][0], sum(SPLITS[split])))
        four = evaluation_specs(replace(inference, gpus=4), split, 'fixed')
        assert [{k:v for k,v in r.items() if k != 'rank'} for r in rows] == [
            {k:v for k,v in r.items() if k != 'rank'} for r in four]
    assert board_protocol('extension_larger') == board_protocol('larger')
    assert board_protocol('extension_test').identity()['total_pairs'] == 8


def test_legacy_driver_rejects320_training_before_engine_creation(tmp_path, monkeypatch):
    from . import rng_overlap
    hero = replace(config(4), updates=320)
    path = tmp_path/'config.json'
    write(path, inference_config(hero, output=tmp_path/'output', gpus=4).model_dump())
    monkeypatch.setattr('sys.argv', ['rng_overlap', 'train', '--config', str(path)])
    with pytest.raises(SystemExit) as error:
        rng_overlap.main()
    assert error.value.code == 2
    assert not (tmp_path/'output').exists()


def test_extension_final_freezes_common_parent_fixed320_and_primary_pair(tmp_path):
    parent_run = tmp_path/'old/arms/w1024'
    parent = replace(config(4), output=str(parent_run), updates=192, schedule_updates=128)
    write(parent_run/'resolved_config.json', asdict(parent))
    output = tmp_path/'extension'
    branches = [extension_config(parent, window=w, output=output/f'arms/w{w}',
        node_dir=tmp_path/f'node/w{w}/control', source_sha256='b'*64, deadline=0.)
        for w in (1024,512)]
    plan = dict(output=str(output), parent_run=str(parent_run), parent_policy_version=192,
        target_update=320, source_sha256='b'*64, branches=[asdict(c) for c in branches])
    path = output/'plan.json';write(path, plan)
    options = dict(gpus=8, source_root=tmp_path/'source', source_sha256='c'*64)
    final = prepare(path, output/'final', **options)
    entries = final['conditions']
    assert [r['condition'] for r in entries[:2]] == final['primary']
    for entry in entries:
        inference = json.loads(Path(entry['argv'][entry['argv'].index('--config')+1]).read_text())
        assert inference['evaluation_concurrency'] == (16 if entry['attention']=='full' else 64)
        expected = 0 if entry['arm']=='base' else 192 if entry['arm']=='parent' else 320
        assert entry['policy_version'] == expected
        assert entry['count'] == (128 if entry['condition'].endswith('first128') else
            256 if entry['split']=='extension_larger' else 512)
        if expected:
            owner = parent if expected==192 else next(c for c in branches if c.window==int(entry['arm'][1:]))
            assert entry['policy_training_identity'] == identity(owner)
        if entry['split']=='extension_test':
            assert entry['first_board_seed']==5000000
        if entry['split']=='extension_larger':
            assert entry['attention']=='ps' and entry['window']==512
            assert entry['first_board_seed']==6000000
    assert prepare(path, output/'final', **options) == final
    assert not list(output.rglob('episodes.jsonl'))
    changed = json.loads(path.read_text());changed['target_update']=319;write(path,changed)
    with pytest.raises(ValueError, match='final320'):
        prepare(path, output/'another-final', **options)
