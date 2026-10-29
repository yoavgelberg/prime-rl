import gzip
import json

import pytest

from .rng_resume import prepare


def test_live_predecessor_cannot_archive(tmp_path):
    folder=tmp_path/'ps/seed-1000000';folder.mkdir(parents=True)
    (folder/'game.json').write_text('{"turns":[]}')
    with pytest.raises(RuntimeError,match='unverified predecessor'):
        prepare(tmp_path,'RUNNING','next')
    assert (folder/'game.json').exists()


def test_completed_results_preserved_partial_games_archived(tmp_path):
    directory=tmp_path/'ps';directory.mkdir()
    good=dict(seed=10,done=True,error=None)
    (directory/'seed-10-result.json').write_text(json.dumps(good))
    (directory/'seed-10').mkdir()
    (directory/'seed-11').mkdir()
    (directory/'seed-11/game.json').write_text('{"turns":[1]}')
    (directory/'seed-11-turns.jsonl.gz').write_bytes(b'partial')
    result=prepare(tmp_path,'TIMEOUT','next')
    assert (directory/'seed-10-result.json').read_text()==json.dumps(good)
    assert (directory/'seed-10').exists()
    assert not (directory/'seed-11').exists()
    assert not (directory/'seed-11-turns.jsonl.gz').exists()
    assert result['archived'][0]['seed']==11
    assert list(tmp_path.glob('interrupted-before-next-*/ps/seed-11/game.json'))


def test_complete_token_record_recovers_receipt(tmp_path):
    directory=tmp_path/'full';directory.mkdir()
    record=dict(seed=12,score=8,total_pairs=8,done=True,error=None,censored=False,
                response_count=30,max_responses=96,tokens=[1,2,3],loss_mask=[False,True,True],
                old_logps=[None,-1.,-2.],generated_tokens=2)
    with gzip.open(directory/'seed-12-tokens.json.gz','wt') as handle:
        json.dump(record,handle)
    result=prepare(tmp_path,'TIMEOUT','next')
    assert result['recovered']==[dict(attention='full',seed=12)]
    assert json.loads((directory/'seed-12-result.json').read_text())['score']==8
    assert not result['archived']
