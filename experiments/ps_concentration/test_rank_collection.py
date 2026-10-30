import asyncio
import json
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.ps_games import GameRolloutRunner
from experiments.ps_concentration.overlap_config import RunConfig, permutation, training_specs
from experiments.ps_concentration.rank_collection import RankCollector, RankEngine


class Engine:
    def __init__(self, gpus, *, failure=False):
        self.gpus, self.failure = gpus, failure
        self.engine_core = SimpleNamespace(core_engines=[str(rank).encode() for rank in range(gpus)],
            _call_utility_async=self.utility)
        self.current = [None] * gpus
        self.batch_counts = [0] * gpus
        self.max_live = [0] * gpus
        self.done, self.abort_events, self.ranks = {}, {}, {}
        self.admitted, self.calls, self.aborted = [], [], set()
        self.slow_gate, self.fast_next_batch = asyncio.Event(), asyncio.Event()
        self.pending = 0

    async def utility(self, operation, method, timeout, args, kwargs, *, engine):
        rank = int(engine)
        self.calls.append((rank, method))
        if method == 'ps_game_prepare':
            assert self.current[rank] is None
            rows = kwargs['options']['specs']
            assert rows and all(row['rank'] == rank for row in rows)
            self.current[rank] = rows
            self.batch_counts[rank] += 1
            self.max_live[rank] = max(self.max_live[rank], len(rows))
            self.admitted.extend(rows)
            if rank == 0 and self.batch_counts[rank] == 2:
                self.fast_next_batch.set()
            result = []
            for row in rows:
                identity = row['episode_id']
                self.done[identity], self.abort_events[identity], self.ranks[identity] = False, asyncio.Event(), rank
                result.append(dict(request_id=identity, prompt_token_ids=[1, 2, 3, 4],
                    generation_seed=row['generation_seed'], data_parallel_rank=rank, max_tokens=99))
        elif method == 'ps_game_status':
            result = {row['episode_id']: dict(done=self.done[row['episode_id']]) for row in self.current[rank]}
        elif method in ('ps_stream_begin', 'ps_stream_reset'):
            result = dict(rank=rank)
        elif method == 'ps_game_finish':
            result = [dict(episode_id=row['episode_id'], generated_tokens=1) for row in self.current[rank]]
        elif method == 'ps_stream_finish':
            result = dict(requests={row['episode_id']: dict(loss_tokens=1) for row in self.current[rank]})
        elif method == 'ps_game_device_metrics':
            result = dict(rank=rank, kv_pool_allocated_bytes=1024)
        elif method == 'ps_train_reduce':
            result = dict(episodes=len(self.current[rank]), rank=rank)
        elif method == 'ps_game_release':
            self.current[rank] = None
            result = None
        elif method == 'ps_game_abort':
            result = None
        else:
            raise AssertionError(method)
        return [result]

    async def generate(self, prompt, params, identity, *, data_parallel_rank):
        rank = data_parallel_rank
        assert self.ranks[identity] == rank
        self.pending += 1
        try:
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[17])])
            if self.failure and rank == 0:
                raise RuntimeError('bad rollout')
            if rank == self.gpus - 1 and self.batch_counts[rank] == 1:
                await self.slow_gate.wait()
            self.done[identity] = True
            await self.abort_events[identity].wait()
        finally:
            self.pending -= 1

    async def abort(self, identities):
        for identity in identities:
            self.aborted.add(identity)
            self.abort_events[identity].set()


def runner(engine):
    return GameRolloutRunner(engine, poll_seconds=.001,
        params_factory=lambda request, **kwargs: SimpleNamespace(seed=request['generation_seed']))


@pytest.mark.parametrize('gpus', [4, 8])
def test_fast_ranks_continue_while_a_slow_rank_preserves_the_same_global_batch(tmp_path, gpus):
    async def run():
        engine = Engine(gpus)
        collector = RankCollector(engine, gpus, runner_factory=runner)
        config = RunConfig(model='/cache/768f209d9ea81521153ed38c47d515654e938aea', source='/native',
            output=str(tmp_path), source_sha256='a' * 64, gpus=gpus)
        specs = training_specs(config, 1, permutation(11))
        task = asyncio.create_task(collector.collect(specs, concurrency=4, factory='native.Factory',
            options_factory=lambda batch, rank: dict(specs=batch, temperature=.6, output=str(tmp_path / f'rank-{rank}')),
            policy_version=0))
        await asyncio.wait_for(engine.fast_next_batch.wait(), 3)
        assert engine.batch_counts[-1] == 1
        assert not any(engine.done[row['episode_id']] for row in engine.current[-1])
        engine.slow_gate.set()
        reports, reductions = await asyncio.wait_for(task, 3)
        assert sorted(engine.admitted, key=lambda row: row['episode_id']) == sorted(specs, key=lambda row: row['episode_id'])
        assert len({row['episode_id'] for report in reports for row in report['episodes']}) == 64
        assert sum(item[0]['episodes'] for item in reductions) == 64
        assert engine.max_live == [4] * gpus
        assert engine.pending == 0 and len(engine.aborted) == 64
        assert len(list(tmp_path.glob('rank-*/cohorts/*.json'))) == 16
        for rank in range(gpus):
            snapshot = json.loads((tmp_path / f'rank-{rank}/live-cohort.json').read_text())
            assert snapshot['backward'] is True and snapshot['policy_version'] == 0
            assert all(engine.ranks[identity] == rank for identity in snapshot['requests'])
        parallel = RunConfig(**dict(config.model_dump(), parallel_rank_collection=True))
        assert parallel.scientific_identity() == config.scientific_identity()
    asyncio.run(run())


def test_rank_scope_rejects_another_gpu_and_cancels_other_collectors_on_failure(tmp_path):
    async def run():
        engine = Engine(2, failure=True)
        view = RankEngine(engine, 0)
        with pytest.raises(ValueError, match='another rank'):
            view.generate({}, {}, 'bad', data_parallel_rank=1)
        with pytest.raises(ValueError, match='existing'):
            RankEngine(engine, 2)
        with pytest.raises(ValueError, match='allocated engine'):
            RankCollector(engine, 4)
        specs = [dict(rank=rank, episode_id=f'r{rank}-{index}', generation_seed=index + rank * 4)
                 for rank in range(2) for index in range(4)]
        collector = RankCollector(engine, 2, runner_factory=runner)
        with pytest.raises(RuntimeError, match='bad rollout'):
            await asyncio.wait_for(collector.collect(specs, concurrency=4, factory='native.Factory',
                options_factory=lambda batch, rank: dict(specs=batch, temperature=.6,
                    output=str(tmp_path / f'rank-{rank}')), policy_version=0), 3)
        assert engine.pending == 0
        assert engine.aborted == {row['episode_id'] for row in specs}
    asyncio.run(run())
