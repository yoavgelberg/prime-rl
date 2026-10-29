"""Persistent-game rollout cohorts on the pinned native vLLM DP engine."""

import asyncio
import time


async def worker_rpc(engine, method, *, kwargs=None, timeout=600.):
    """Keep every DP reply: pinned vLLM's public RPC returns only rank zero."""
    core = engine.engine_core
    engines = core.core_engines if hasattr(core, 'core_engines') else [core.core_engine]
    replies = await asyncio.gather(*[
        core._call_utility_async('collective_rpc', method, timeout, (), kwargs, engine=identity)
        for identity in engines
    ])
    if any(not isinstance(reply, list) or len(reply) != 1 for reply in replies):
        raise ValueError('The game adapter requires one TP=PP=1 worker per data-parallel replica')
    return [reply[0] for reply in replies]


def sampling_params(request, *, temperature, greedy):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(temperature=0. if greedy else temperature, top_p=1., top_k=-1,
        seed=request['generation_seed'], max_tokens=request['max_tokens'], ignore_eos=True,
        logprobs=None, prompt_logprobs=None, detokenize=False, skip_special_tokens=False,
        output_kind=RequestOutputKind.DELTA)


class GameRolloutRunner:
    def __init__(self, engine, *, poll_seconds=.25, params_factory=sampling_params):
        self.engine, self.poll_seconds, self.params_factory = engine, poll_seconds, params_factory
        self.backward_started = False

    async def collect(self, factory, options, *, backward=False, policy_version=0):
        started = time.perf_counter()
        per_rank = await worker_rpc(self.engine, 'ps_game_prepare', kwargs=dict(factory=factory, options=options))
        requests = [request for rank_requests in per_rank for request in rank_requests]
        identities = [request['request_id'] for request in requests]
        if len(set(identities)) != len(identities):
            raise ValueError('Episode IDs must be unique across all GPU replicas')
        ranks = {request['request_id']: request['data_parallel_rank'] for request in requests}
        if backward:
            method = 'ps_stream_reset' if self.backward_started else 'ps_stream_begin'
            await worker_rpc(self.engine, method, kwargs={} if self.backward_started else dict(policy_version=policy_version))
            self.backward_started = True
        natural, aborted, engine_tokens = set(), set(), {identity: 0 for identity in identities}

        async def drive(request):
            identity = request['request_id']
            params = self.params_factory(request, temperature=options.get('temperature', .6),
                                         greedy=options.get('greedy', False))
            async for output in self.engine.generate(dict(prompt_token_ids=request['prompt_token_ids']), params,
                    identity, data_parallel_rank=request['data_parallel_rank']):
                engine_tokens[identity] += sum(len(row.token_ids) for row in output.outputs)
            natural.add(identity)

        tasks = {request['request_id']: asyncio.create_task(drive(request)) for request in requests}
        termination_signaled = set()
        try:
            while True:
                for task in tasks.values():
                    if task.done() and not task.cancelled() and task.exception() is not None:
                        raise task.exception()
                statuses = await worker_rpc(self.engine, 'ps_game_status', timeout=60.)
                combined = {identity: status for rank in statuses for identity, status in rank.items()}
                if combined.keys() != tasks.keys():
                    raise ValueError('Native-game status does not cover the admitted cohort exactly')
                exhausted = natural - aborted - termination_signaled
                exhausted = {identity for identity in exhausted if not combined[identity]['done']}
                if exhausted:
                    await worker_rpc(self.engine, 'ps_game_engine_finished',
                                     kwargs=dict(request_ids=sorted(exhausted)), timeout=60.)
                    termination_signaled.update(exhausted)
                finished = {identity for identity, status in combined.items() if status['done']}
                to_abort = finished - aborted
                if to_abort:
                    await self.engine.abort(sorted(to_abort))
                    aborted.update(to_abort)
                if len(finished) == len(tasks):
                    break
                await asyncio.sleep(self.poll_seconds)
            await asyncio.gather(*tasks.values())
            rows = await worker_rpc(self.engine, 'ps_game_finish')
            backward_rows = await worker_rpc(self.engine, 'ps_stream_finish') if backward else None
            if backward_rows is not None:
                for rank, report in enumerate(backward_rows):
                    expected = sum(row['generated_tokens'] for row in rows[rank])
                    observed = sum(row['loss_tokens'] for row in report['requests'].values())
                    if observed != expected:
                        raise ValueError(f'GPU {rank}: {observed} direct losses for {expected} sampled assistant tokens')
            return dict(episodes=[row for rank in rows for row in rank], backward=backward_rows,
                seconds=time.perf_counter() - started, engine_output_tokens=sum(engine_tokens.values()),
                engine_tokens_per_request=engine_tokens, policy_version=policy_version,
                physical_ranks=ranks, greedy=options.get('greedy', False))
        except BaseException:
            await self.engine.abort(identities)
            await worker_rpc(self.engine, 'ps_game_abort', kwargs=dict(reason='Rollout coordinator aborted'))
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise

    async def release(self):
        await worker_rpc(self.engine, 'ps_game_release')
