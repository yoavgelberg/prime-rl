"""Pinned native Matching Pairs games inside persistent vLLM worker requests."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
from threading import local
import time
import types

from transformers import AutoTokenizer

from prime_rl.inference.vllm.ps_interactive import PersistentChat, TranscriptLimit

from .rng_metrics import score_prefix
from .rng_training_config import outcome_reward
from .overlap_config import BoardProtocol


class GameCohort:
    compress_native = False
    defer_native_logs = False

    def __init__(self, *, model, source, output, specs, rank, attention='ps', window=1024,
                 transcript_cap=262144, max_responses=96, temperature=.6, greedy=False,
                 context_exhaustion_reward=-1., collect_cuda_metrics=True, rows=4, cols=4, num_cards=8,
                 thinking_token_limit=None, action_token_limit=None, thinking_budget_transition=None,
                 compress_native=False, defer_native_logs=False):
        self.rank, self.attention, self.window = rank, attention, window
        self.temperature, self.greedy = temperature, greedy
        self.transcript_cap, self.max_responses = transcript_cap, max_responses
        self.thinking_token_limit = thinking_token_limit
        self.compress_native = compress_native
        self.defer_native_logs = defer_native_logs
        self.board = BoardProtocol(rows=rows, cols=cols, num_cards=num_cards, max_responses=max_responses)
        if self.board.cols != 4 and (not greedy or attention != 'ps'):
            raise ValueError('The larger board is reserved for greedy PS evaluation')
        self.context_exhaustion_reward = context_exhaustion_reward
        self.source, self.output = Path(source), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        lock = json.loads((self.source / 'SOURCE.json').read_text())
        if lock['commit'] != 'e3d195e857727dfa3d4a96561be202098d67da83':
            raise ValueError('The native game revision differs from the frozen benchmark')
        for row in lock['manifest']:
            if hashlib.sha256((self.source / row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Native source changed: {row["path"]}')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        generation = json.loads((Path(model) / 'generation_config.json').read_text())
        eos = generation['eos_token_id']
        eos = eos if isinstance(eos, list) else [eos]
        self.specs = {spec['episode_id']: spec for spec in specs if spec['rank'] == rank}
        self.chats, self.paths = {}, {}
        if collect_cuda_metrics:
            import torch
            self.device = torch.cuda.current_device()
        else:
            self.device = None
        state = local()
        sys.path.insert(0, str(self.source / '1_matching_pairs_new'))
        shim = types.ModuleType('model_presets')
        shim.make_client = lambda *args, **kwargs: state.client
        shim.parse_grid_size = lambda value: tuple(map(int, value.split('x')))
        sys.modules['model_presets'] = shim
        from modes import single_normal
        from common import prompts
        from common.turn_logger import TurnLogger
        from .native_logging import buffered_logger
        single_normal.TurnLogger = buffered_logger(TurnLogger, state) if defer_native_logs else TurnLogger
        single_normal.make_client = shim.make_client
        prompts.OUTPUT_FORMAT_NO_COT = '''Output format:
Keep your thoughts brief and concise. Once you have chosen your next move, finish your reasoning.
After thinking, your entire final answer must be exactly one action line:
Action: <coordinate, e.g. cD>
Do not put explanations, analysis, headings, or additional lines in the final answer.
Do not continue reasoning after </think> or start another thinking block.
'''
        if thinking_token_limit is not None:
            prompts.OUTPUT_FORMAT_NO_COT += (f'Your thinking budget is {thinking_token_limit} tokens per move. '
                'Use a few short sentences, finish reasoning within this budget, and give the required action. '
                'If uncertain, choose your best available action within the budget instead of continuing deliberation.\n')
        self.run_one_game, self.state = single_normal.run_one_game, state
        for identity, spec in self.specs.items():
            folder = self.output / identity
            folder.mkdir(exist_ok=False)
            self.paths[identity] = folder
            self.chats[identity] = PersistentChat(self.tokenizer, eos_token_ids=eos,
                transcript_cap=transcript_cap, thinking_token_limit=thinking_token_limit,
                action_token_limit=action_token_limit, thinking_budget_transition=thinking_budget_transition,
                on_turn=lambda turn, identity=identity: self._turn(identity, turn))
        self.pool = ThreadPoolExecutor(max_workers=max(len(self.chats), 1))
        self.futures = {identity: self.pool.submit(self._play, identity) for identity in self.chats}
        self.results = None

    def _turn(self, identity, turn):
        folder = self.paths[identity]
        with gzip.open(folder / 'turns.jsonl.gz', 'at', compresslevel=1) as handle:
            handle.write(json.dumps(turn) + '\n')
        client = self.chats[identity]
        row = dict(episode_id=identity, board_seed=self.specs[identity]['board_seed'],
            attention=self.attention, turns=len(client.turns), raw_transcript_tokens=len(client.tokens),
            generated_tokens=client.generated_tokens, latest_action=turn['content'],
            seconds=time.perf_counter() - client.started)
        temporary = folder / 'progress.json.tmp'
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(folder / 'progress.json')
        print(json.dumps(dict(event='game_turn', **row)), flush=True)

    def _play(self, identity):
        chat, folder, spec = self.chats[identity], self.paths[identity], self.specs[identity]
        self.state.client = chat
        try:
            censored, reason = False, None
            try:
                result = self.run_one_game(model='local-vllm', rows=self.board.rows, cols=self.board.cols,
                    num_cards=self.board.num_cards,
                    seed=spec['board_seed'], render='text', theme=None,
                    assets_dir=str(self.source / '1_matching_pairs_new/assets'), cell_size=64,
                    out_dir=folder / 'native', max_responses=self.max_responses, max_retries=2, cot_enabled=False)
                if result.get('error') is not None:
                    raise RuntimeError(f'Native game failed: {result["error"]}')
            except TranscriptLimit as error:
                censored, reason = True, str(error)
                chat.exhaust()
            finally:
                if self.defer_native_logs and getattr(self.state, 'logger', None) is not None:
                    self.state.logger.flush_pending()
            native = json.loads((folder / 'native/game.json').read_text())
            row = dict(score_prefix(native, self.max_responses), **spec)
            row.update(censored=censored, resource_limit_reason=reason, greedy=self.greedy)
            row['native_intermediate_full_json_flushes_deferred'] = self.defer_native_logs
            if native['config']['total_pairs'] != self.board.rows * self.board.cols // 2:
                raise ValueError('The native board differs from its evaluation protocol')
            row.update(self.board.identity())
            if not censored and row['score'] != result['score']:
                raise ValueError('Offline native score differs from the environment result')
            row['training_return'] = outcome_reward(row, self.max_responses, self.context_exhaustion_reward)
            row['entropy_tokens'] = chat.entropy_tokens
            row['thinking_token_limit'] = chat.thinking_token_limit
            row['action_token_limit'] = chat.action_token_limit
            row['budget_enforcement'] = chat.budget_enforcement
            row['thinking_budget_transition'] = chat.thinking_budget_transition
            row['thinking_tokens'] = sum(turn['thinking_tokens'] for turn in chat.turns)
            row['thinking_budget_forced_turns'] = sum(turn['thinking_budget_forced'] for turn in chat.turns)
            row['mean_policy_entropy_at_temperature'] = chat.entropy_sum / chat.entropy_tokens if chat.entropy_tokens else None
            row['negative_logp_sum'] = -sum(value for value in chat.old_tempered_logps if value is not None)
            row['board_identity_sha256'] = hashlib.sha256(native['config']['ground_truth_text'].encode()).hexdigest()
            row.update(generated_tokens=chat.generated_tokens, raw_transcript_tokens=len(chat.tokens),
                seconds=time.perf_counter() - chat.started, engine='vllm-persistent-game')
            chat.finish()
            record = dict(chat.record(), **row)
            native_path = folder / 'native/game.json'
            if self.compress_native:
                compressed = native_path.with_suffix('.json.gz')
                with gzip.open(compressed, 'wt', compresslevel=1) as handle:
                    json.dump(native, handle)
                native_path.unlink()
                native_path = compressed
            record.update(attention=self.attention, window=self.window,
                temperature=self.temperature, top_p=1., top_k=-1,
                native_game_json=str(native_path),
                logprob_definition='Unfiltered softmax at recorded temperature; greedy choices are evaluation only')
            if not all((value is not None) == keep for value, keep in zip(
                    record['old_tempered_logps'], record['loss_mask'], strict=True)):
                raise ValueError('Forced tokens acquired a direct loss or assistant likelihoods are missing')
            if self.device is not None:
                import torch
                record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device)
                record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(self.device)
                row['peak_allocated_bytes'] = record['peak_allocated_bytes']
                row['peak_reserved_bytes'] = record['peak_reserved_bytes']
            temporary = folder / 'tokens.json.gz.tmp'
            with gzip.open(temporary, 'wt', compresslevel=1) as handle:
                json.dump(record, handle)
            temporary.replace(folder / 'tokens.json.gz')
            row.update(token_record=str(folder / 'tokens.json.gz'),
                token_record_sha256=hashlib.sha256((folder / 'tokens.json.gz').read_bytes()).hexdigest())
            temporary = folder / 'result.json.tmp'
            temporary.write_text(json.dumps(row, indent=2) + '\n')
            temporary.replace(folder / 'result.json')
            return row
        except BaseException as error:
            chat.finish(error=error)
            raise

    def initial_requests(self):
        return [dict(request_id=identity, prompt_token_ids=chat.initial_prompt(),
            generation_seed=self.specs[identity]['generation_seed'], data_parallel_rank=self.rank,
            max_tokens=self.transcript_cap - len(chat.prefix)) for identity, chat in self.chats.items()]

    def status(self):
        return {identity: chat.status() for identity, chat in self.chats.items()}

    def finish(self):
        if not all(chat.status()['done'] for chat in self.chats.values()):
            raise RuntimeError('Cannot finish before native games are terminal')
        if self.results is None:
            self.results = [future.result() for future in self.futures.values()]
            self.pool.shutdown(wait=True)
        return self.results
