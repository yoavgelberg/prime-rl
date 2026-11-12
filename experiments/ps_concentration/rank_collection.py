"""Independent GPU rollout cohorts with one synchronous global policy update."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from prime_rl.orchestrator.ps_games import GameRolloutRunner, worker_rpc


class RankEngine:
    def __init__(self, engine, rank):
        core = engine.engine_core
        identities = core.core_engines if hasattr(core, 'core_engines') else [core.core_engine]
        if type(rank) is not int or not 0 <= rank < len(identities):
            raise ValueError('Select an existing data-parallel rank')
        self.engine, self.rank = engine, rank
        self.engine_core = SimpleNamespace(core_engines=[identities[rank]],
            _call_utility_async=core._call_utility_async)

    def generate(self, *args, data_parallel_rank, **kwargs):
        if data_parallel_rank != self.rank:
            raise ValueError('A rank-scoped collector cannot generate on another rank')
        return self.engine.generate(*args, data_parallel_rank=data_parallel_rank, **kwargs)

    async def abort(self, identities):
        await self.engine.abort(identities)


class RankCollector:
    def __init__(self, engine, gpus, *, runner_factory=GameRolloutRunner):
        core = engine.engine_core
        identities = core.core_engines if hasattr(core, 'core_engines') else [core.core_engine]
        if len(identities) != gpus:
            raise ValueError('Collector rank count differs from the allocated engine')
        self.engines = [RankEngine(engine, rank) for rank in range(gpus)]
        self.runners = [runner_factory(view) for view in self.engines]

    async def collect(self, specs, *, concurrency, factory, options_factory, policy_version):
        if concurrency < 1 or len({row['episode_id'] for row in specs}) != len(specs):
            raise ValueError('Use positive concurrency and unique trajectory identities')
        if any(not 0 <= row['rank'] < len(self.runners) for row in specs):
            raise ValueError('A trajectory names an unallocated rank')
        per_rank = [[row for row in specs if row['rank'] == rank] for rank in range(len(self.runners))]
        if any(not rows for rows in per_rank):
            raise ValueError('Every training rank must own trajectories in the global batch')

        async def collect_rank(rank):
            reports, reductions = [], []
            for number, start in enumerate(range(0, len(per_rank[rank]), concurrency)):
                batch = per_rank[rank][start:start + concurrency]
                options = options_factory(batch, rank)
                Path(options['output']).mkdir(parents=True, exist_ok=True)
                report = await self.runners[rank].collect(factory, options,
                    backward=True, policy_version=policy_version)
                reduction = await worker_rpc(self.engines[rank], 'ps_train_reduce', timeout=1800.)
                reports.append(report)
                reductions.append(reduction)
                folder = Path(options['output']) / 'cohorts'
                folder.mkdir(parents=True, exist_ok=True)
                receipt = folder / f'cohort-{number:04d}.json'
                temporary = receipt.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(dict(policy_version=policy_version, rank=rank,
                    report=report, reduction=reduction), allow_nan=False) + '\n')
                temporary.replace(receipt)
                await self.runners[rank].release()
            return reports, reductions

        tasks = [asyncio.create_task(collect_rank(rank)) for rank in range(len(self.runners))]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return ([report for reports, _ in results for report in reports],
                [reduction for _, reductions in results for reduction in reductions])
