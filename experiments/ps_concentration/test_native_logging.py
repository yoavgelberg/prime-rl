from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
from threading import local

import pytest

from .native_logging import buffered_logger


@pytest.fixture
def native_logger():
    source = os.environ.get('PS_CONCENTRATION_NATIVE_SOURCE')
    if source is None:
        pytest.skip('Set PS_CONCENTRATION_NATIVE_SOURCE to the pinned RNG-Bench source')
    root = Path(source)
    assert json.loads((root/'SOURCE.json').read_text())['commit'] == 'e3d195e857727dfa3d4a96561be202098d67da83'
    spec = importlib.util.spec_from_file_location('pinned_turn_logger', root/'1_matching_pairs_new/common/turn_logger.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TurnLogger


def turns(count):
    messages = [dict(role='system', content='Pinned rules')]
    for i in range(count):
        messages.append(dict(role='user',content='a [*][*][*][*]\nb [*][*][*][*]'))
        response = dict(content='Action: aA' if i%3 else 'invalid output',reasoning='reasoning '*16)
        yield dict(call_type='flip_first' if i%2==0 else 'flip_second', round_idx=i//2+1,
            player=None,messages_sent=messages,response=response,parse=dict(ok=bool(i%3),coord='aA'),
            env_result=dict(applied=bool(i%3),revealed_face='0',phase_after='flip_second'),retry_of=None)
        messages.append(dict(role='assistant',content=response['content'],reasoning=response['reasoning']))


@pytest.mark.parametrize('count',[0,3,12,96])
@pytest.mark.parametrize('finalize',[False,True])
def test_buffered_native_log_preserves_every_final_byte_and_partial_score_state(tmp_path,native_logger,count,finalize):
    state=local()
    reference=native_logger(tmp_path/'reference.json',dict(seed=123,rows=4,cols=4,total_pairs=8))
    buffered=buffered_logger(native_logger,state)(tmp_path/'buffered.json',dict(seed=123,rows=4,cols=4,total_pairs=8))
    initial=(tmp_path/'buffered.json').read_bytes()
    for number,turn in enumerate(turns(count),1):
        assert reference.log_call(**turn)==buffered.log_call(**turn)
        assert reference.response_count==buffered.response_count==number
        assert reference._data==buffered._data
        assert (tmp_path/'buffered.json').read_bytes()==initial
    if finalize:
        reference.finalize(dict(score=3,done=False))
        buffered.finalize(dict(score=3,done=False))
    else:
        buffered.flush_pending()
    assert (tmp_path/'reference.json').read_bytes()==(tmp_path/'buffered.json').read_bytes()
    before=(tmp_path/'buffered.json').stat().st_mtime_ns
    buffered.flush_pending()
    assert (tmp_path/'buffered.json').stat().st_mtime_ns==before


def test_buffered_logger_isolates_concurrent_episode_flushes(tmp_path,native_logger):
    state=local()
    factory=buffered_logger(native_logger,state)

    def run(seed):
        logger=factory(tmp_path/f'{seed}.json',dict(seed=seed))
        for turn in turns(4):
            logger.log_call(**turn)
        assert state.logger is logger
        state.logger.flush_pending()
        return json.loads(logger.output_path.read_text())

    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(run,[11,22]))
    assert [v['config']['seed'] for v in values]==[11,22]
    assert all(len(v['turns'])==4 for v in values)
