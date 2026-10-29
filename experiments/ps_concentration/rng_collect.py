"""Fresh on-policy native-game trajectories with actual sampled likelihoods."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
from queue import Queue
import sys
from threading import local
import time
import types

import torch

from .rng_batch import BatchDecoder, LaneClient
from .rng_metrics import score_prefix
from .rng_stream import TranscriptCapacity
from .rng_training_config import outcome_reward


class TrainingLane(LaneClient):
    def reset_training(self, spec, output):
        super().reset(spec['board_seed'], output)
        self.spec = spec
        self.rng.manual_seed(spec['generation_seed'])
        self.old_tempered_logps = []

    def save(self, row):
        if row.get('censored') and self.sampled and (not self.turns or self.turn_start >= self.turns[-1]['end']):
            self.turns.append(dict(start=self.turn_start, end=len(self.tokens), sampled_ids=self.sampled,
                                   old_logps=self.sample_logps, stop_condition='transcript_capacity',
                                   reasoning=self.tokenizer.decode(self.sampled, skip_special_tokens=False), content=''))
        record = dict(row, tokens=self.tokens, roles=self.roles, positions=list(range(len(self.tokens))),
                      loss_mask=[role == 'assistant' for role in self.roles], old_logps=self.old_logps,
                      old_tempered_logps=self.old_tempered_logps, turns=self.turns,
                      prefix_length=len(self.prefix), attention=self.args.attention, window=self.args.window,
                      thinking_retained=True, temperature=self.args.temperature,
                      top_p=self.args.top_p, top_k=self.args.top_k,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_kv_bytes=self.stream.cache_bytes,
                      allocated_kv_bytes=sum(k.numel() * k.element_size() * 2 for k in self.stream.keys))
        assert len(record['tokens']) == len(record['old_tempered_logps'])
        assert all((lp is not None) == keep for lp, keep in zip(record['old_tempered_logps'], record['loss_mask'], strict=True))
        path = self.output / 'tokens.json.gz'
        temporary = path.with_suffix('.gz.tmp')
        with gzip.open(temporary, 'wt', compresslevel=1) as handle:
            json.dump(record, handle)
        temporary.replace(path)
        return record


class TrainingDecoder(BatchDecoder):
    def __init__(self, clients, requests, args, greedy=False):
        super().__init__(clients, requests, args)
        self.greedy = greedy

    def _append(self, lane, tokens, role, sample_after=False):
        super()._append(lane, tokens, role, sample_after)
        lane.old_tempered_logps.extend([None] * (len(lane.tokens) - len(lane.old_tempered_logps)))
        if role == 'assistant':
            assert len(tokens) == 1
            lane.old_tempered_logps[-1] = lane.next_tempered_logp

    def _emit_many(self, indices, logits):
        active = []
        for index in indices:
            try:
                self.clients[index]._check_capacity(1)
            except TranscriptCapacity as error:
                self._fail(self.clients[index], error)
                continue
            active.append(index)
        if not active:
            return
        raw = logits[active].float()
        scaled = raw / self.args.temperature
        if self.greedy:
            tokens = raw.argmax(-1)
        else:
            values, choices = scaled.topk(self.args.top_k, dim=-1)
            probs = values.softmax(-1)
            probs.masked_fill_(probs.cumsum(-1) - probs >= self.args.top_p, 0.)
            probs /= probs.sum(-1, keepdim=True)
            selected = torch.stack([torch.multinomial(probs[row], 1, generator=self.clients[index].rng)[0]
                                    for row, index in enumerate(active)])
            tokens = choices.gather(1, selected[:, None])[:, 0]
        raw_logps = raw.gather(1, tokens[:, None])[:, 0] - raw.logsumexp(-1)
        tempered = scaled.gather(1, tokens[:, None])[:, 0] - scaled.logsumexp(-1)
        sampled = torch.stack((tokens.float(), raw_logps, tempered), dim=-1).tolist()
        for index, (token, raw_logp, logp) in zip(active, sampled, strict=True):
            self.clients[index].next_tempered_logp = logp
            self._emit(index, int(token), raw_logp)


class NativeCollector:
    def __init__(self, policy, tokenizer, config):
        self.policy, self.tokenizer, self.config = policy, tokenizer, config
        self.state = local()
        source = Path(config.source)
        sys.path.insert(0, str(source / '1_matching_pairs_new'))
        shim = types.ModuleType('model_presets')
        shim.make_client = lambda *a, **kw: self.state.client
        shim.parse_grid_size = lambda value: tuple(map(int, value.split('x')))
        sys.modules['model_presets'] = shim
        from modes import single_normal
        from common import prompts
        single_normal.make_client = shim.make_client
        prompts.OUTPUT_FORMAT_NO_COT = '''Output format:
After thinking, end your final response with exactly one action line:
Action: <coordinate, e.g. cD>
'''
        self.run_one_game = single_normal.run_one_game
        self.system_prompt = prompts.build_single_system(cot_enabled=False)
        self.source = source

    def collect(self, specs, output, *, greedy=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        requests = Queue()
        clients = [TrainingLane(self.policy, self.tokenizer, self.config, requests, i) for i in range(len(specs))]
        for client, spec in zip(clients, specs, strict=True):
            folder = output / spec['episode_id']
            if folder.exists():
                folder.rename(folder.with_name(f'{folder.name}-interrupted-{time.time_ns()}'))
            folder.mkdir()
            client.reset_training(spec, folder)

        def play(index):
            client, spec = clients[index], specs[index]
            self.state.client = client
            try:
                censored, reason = False, None
                try:
                    result = self.run_one_game(model='local', rows=4, cols=4, num_cards=None,
                        seed=spec['board_seed'], render='text', theme=None,
                        assets_dir=str(self.source / '1_matching_pairs_new/assets'), cell_size=64,
                        out_dir=client.output / 'native', max_responses=self.config.max_responses,
                        max_retries=2, cot_enabled=False)
                    assert result.get('error') is None, result
                except TranscriptCapacity as error:
                    censored, reason = True, str(error)
                native = json.loads((client.output / 'native/game.json').read_text())
                row = dict(score_prefix(native, self.config.max_responses), **spec,
                           censored=censored, resource_limit_reason=reason, greedy=greedy)
                if not censored:
                    assert row['score'] == result['score']
                row['training_return'] = None if censored and self.config.context_exhaustion_reward is None else outcome_reward(
                    row, self.config.max_responses, self.config.context_exhaustion_reward)
                row.update(generated_tokens=sum(role == 'assistant' for role in client.roles),
                           raw_transcript_tokens=len(client.tokens), seconds=time.perf_counter() - client.started)
                requests.put((index, None))
                record = client.save(row)
                (client.output / 'result.json').write_text(json.dumps(row, indent=2))
                return record
            finally:
                requests.put((index, None))

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = [pool.submit(play, index) for index in range(len(clients))]
            try:
                ticks = TrainingDecoder(clients, requests, self.config, greedy).run()
            except BaseException:
                for client in clients:
                    error = RuntimeError('Training collector aborted; coordinator retains original exception')
                    client.abort_error = error
                    if hasattr(client, 'pending'):
                        client.pending['error'] = error
                        client.pending['event'].set()
                raise
            records = [future.result() for future in futures]
        seconds = time.perf_counter() - started
        (output / f'batch-{specs[0]["episode_id"]}.json').write_text(json.dumps(dict(
            episodes=len(records), seconds=seconds, ticks=ticks,
            generated_tokens=sum(row['generated_tokens'] for row in records)), indent=2))
        for client in clients:
            client.stream = None
        torch.cuda.empty_cache()
        if any(row['training_return'] is None for row in records):
            raise RuntimeError('Saved all trajectories; resource-limit reward must be resolved before an optimizer update')
        return records
