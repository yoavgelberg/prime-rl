import asyncio
from types import SimpleNamespace

from prime_rl.orchestrator.ps_games import GameRolloutRunner, worker_rpc


class Engine:
    def __init__(self):
        self.engine_core = SimpleNamespace(core_engines=[b'0', b'1'], core_engine=b'0',
                                           _call_utility_async=self.utility)
        self.native_done = [False, False]
        self.abort_events = [asyncio.Event(), asyncio.Event()]
        self.calls, self.params, self.exhausted, self.aborted = [], {}, set(), set()

    async def utility(self, operation, method, timeout, args, kwargs, *, engine):
        rank = int(engine)
        self.calls.append((rank, method))
        if method == 'ps_game_prepare':
            result = [dict(request_id=f'game{rank}', prompt_token_ids=[1, 2, 3, 4],
                           generation_seed=11 + rank, data_parallel_rank=rank, max_tokens=100)]
        elif method == 'ps_game_status':
            result = {f'game{rank}': dict(done=self.native_done[rank])}
        elif method == 'ps_game_engine_finished':
            if f'game{rank}' in kwargs['request_ids']:
                self.native_done[rank] = True
                self.exhausted.add(rank)
            result = None
        elif method == 'ps_game_finish':
            result = [dict(episode_id=f'game{rank}', generated_tokens=rank + 2)]
        elif method == 'ps_game_device_metrics':
            result = dict(rank=rank, kv_pool_allocated_bytes=2048)
        elif method in ('ps_game_release', 'ps_game_abort'):
            result = None
        else:
            raise AssertionError(method)
        return [result]

    async def generate(self, prompt, params, identity, *, data_parallel_rank):
        rank = data_parallel_rank
        assert prompt == dict(prompt_token_ids=[1, 2, 3, 4])
        self.params[rank] = params
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[17])])
        if rank == 1:
            # Engine length limit arrives while the native game is unfinished.
            return
        self.native_done[rank] = True
        await self.abort_events[rank].wait()

    async def abort(self, identities):
        for identity in identities:
            rank = int(identity[-1])
            self.aborted.add(rank)
            self.abort_events[rank].set()


def test_driver_retains_all_dp_replies_and_distinguishes_native_end_from_length_end():
    async def run():
        engine = Engine()
        factory = lambda request, **kwargs: SimpleNamespace(**kwargs, seed=request['generation_seed'])
        runner = GameRolloutRunner(engine, poll_seconds=.001, params_factory=factory)
        result = await asyncio.wait_for(runner.collect('native.Factory', dict(temperature=.6)), 3)
        assert [row['episode_id'] for row in result['episodes']] == ['game0', 'game1']
        assert result['engine_output_tokens'] == 2
        assert engine.exhausted == {1}
        assert engine.aborted == {0, 1}
        assert engine.params[0].seed == 11 and engine.params[1].seed == 12
        for rank in (0, 1):
            assert (rank, 'ps_game_prepare') in engine.calls
            assert (rank, 'ps_game_finish') in engine.calls
        await runner.release()
        assert engine.calls[-2:] == [(0, 'ps_game_release'), (1, 'ps_game_release')]
    asyncio.run(run())


def test_worker_rpc_does_not_silently_accept_tensor_parallel_replies():
    async def run():
        engine = Engine()
        async def invalid(*args, **kwargs):
            return [dict(rank=0), dict(rank=1)]
        engine.engine_core._call_utility_async = invalid
        import pytest
        with pytest.raises(ValueError, match='TP=PP=1'):
            await worker_rpc(engine, 'anything')
    asyncio.run(run())
