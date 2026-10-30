"""Scientific configuration, fixed seed plans and physical rollout cohorts."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from prime_rl.trainer.ps_backward.config import StreamConfig
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig
from .rng_training_config import training_rng


@dataclass(frozen=True)
class RunConfig:
    model: str
    source: str
    output: str
    source_sha256: str
    gpus: int = 8
    concurrency: int = 4
    evaluation_concurrency: int = 16
    parallel_rank_collection: bool = False
    seed: int = 11
    updates: int = 32
    window: int = 1024
    transcript_cap: int = 262144
    max_responses: int = 96
    temperature: float = .6
    validation_interval: int = 16
    validation_boards: int = 128
    kl_coefficient: float = .01
    allow_numerical_target_miss: bool = True
    wandb_project: str = 'ps-memory'
    wandb_entity: str = 'tech-ml'
    wandb_id: str = 'rng-4b-ps-grpo-s11-a1'
    revision: str = '768f209d9ea81521153ed38c47d515654e938aea'

    def __post_init__(self):
        if self.gpus not in (1, 2, 4, 8) or 8 % self.gpus:
            raise ValueError('The fixed eight-board batch requires 1, 2, 4 or 8 GPU replicas')
        if self.concurrency not in range(1, 5) or self.evaluation_concurrency not in range(1, 65):
            raise ValueError('Training concurrency must fit the audited host-memory ceiling of four')
        if type(self.parallel_rank_collection) is not bool:
            raise ValueError('Parallel rank collection is a physical-layout boolean setting')
        if type(self.window) is not int or self.window < 1:
            raise ValueError('Window size must be a positive token count')
        if Path(self.model).name != self.revision:
            raise ValueError('Training/reference/tokenizer must use the immutable starting snapshot')
        if len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256):
            raise ValueError('A captured source archive SHA256 is required')
        if self.seed not in (11, 22, 33) or not 1 <= self.updates <= 256:
            raise ValueError('Use the frozen training seeds and at most 256 total updates')
        if self.max_responses != 96 or self.transcript_cap != 262144:
            raise ValueError('The accepted native protocol uses 96 attempts and the native 262K transcript ceiling')
        if self.validation_boards != 128 or self.validation_interval != 16:
            raise ValueError('Checkpoint selection uses the frozen 128-board validation split every 16 updates')
        if self.temperature != .6 or self.kl_coefficient != .01:
            raise ValueError('The frozen experiment uses temperature .6 and reference KL coefficient .01')

    def model_dump(self):
        return asdict(self)

    def stream(self):
        return StreamConfig(window=self.window, chunk_tokens=1024, ring_tokens=1024,
            max_requests=self.concurrency, tail_block_tokens=16384, fp32_attention=True,
            joint_qkv=True, direct_weight_gradients=True, weight_gradient_backend='cublas',
            compact_kv=True, kv_cache=True, fine_grained_events=True, gpu_head_staging=True,
            stage_scheduling=True, copy_mode='sm', gpu_stages_in_flight=4, backward_batch_tokens=1024,
            score_temperature=self.temperature, loss_denominator=26624)

    def scientific_identity(self):
        return dict(task='RNGBench native textual Matching Pairs 4x4', algorithm='GRPO',
            model_revision=self.revision, native_revision='e3d195e857727dfa3d4a96561be202098d67da83',
            training_seed=self.seed, reference_checkpoint=self.model, window=self.window,
            transcript_cap=self.transcript_cap, max_responses=self.max_responses,
            temperature=self.temperature, top_p=1., top_k=-1, kl_coefficient=self.kl_coefficient,
            outcome=asdict(OutcomeConfig()), reward='pairs/8-attempts/96; resource exhaustion -1')


def permutation(seed):
    return np.random.default_rng(seed).permutation(1000000)


def training_specs(config, update, boards):
    if not 1 <= update <= 256:
        raise ValueError('Update numbers are one based')
    return [dict(rank=group // (8 // config.gpus), board_seed=int(boards[(update - 1) * 8 + group]),
        board_index=group, rollout_index=rollout,
        generation_seed=training_rng(config.seed, update - 1, group, rollout),
        episode_id=f'u{update:04d}-g{group}-r{rollout}') for group in range(8) for rollout in range(8)]


@dataclass(frozen=True)
class BoardProtocol:
    rows: int = 4
    cols: int = 4
    num_cards: int = 8
    max_responses: int = 96

    def __post_init__(self):
        if (self.rows, self.cols, self.num_cards, self.max_responses) not in ((4, 4, 8, 96), (4, 8, 8, 192)):
            raise ValueError('Use the frozen training board or the parameter-scaled larger-board evaluation')

    def identity(self):
        return dict(board_rows=self.rows, board_cols=self.cols, distinct_ranks=self.num_cards,
                    total_pairs=self.rows * self.cols // 2, max_responses=self.max_responses)


def board_protocol(split):
    return BoardProtocol(cols=8, max_responses=192) if split == 'larger' else BoardProtocol()


SPLITS = dict(screening=(1000000, 64), validation=(1100000, 128), test=(2000000, 512), larger=(3000000, 256),
              intervention=(4000000, 512))


def evaluation_specs(config, split, label, *, count=None):
    start, total = SPLITS[split]
    if count is not None:
        if not 1 <= count <= total:
            raise ValueError('An evaluation count must remain inside its reserved split')
        total = count
    return [dict(rank=index % config.gpus, board_seed=start + index, board_index=index, rollout_index=0,
        generation_seed=training_rng(0, 0, index, 0), episode_id=f'{label}-b{start + index}')
        for index in range(total)]


def cohorts(specs, gpus, concurrency):
    per_rank = [[row for row in specs if row['rank'] == rank] for rank in range(gpus)]
    for start in range(0, max(map(len, per_rank), default=0), concurrency):
        yield [row for rows in per_rank for row in rows[start:start + concurrency]]


def episode_metrics(rows):
    count = len(rows)
    if not count:
        raise ValueError('A metric report requires observed episodes')
    pair_counts = {row.get('total_pairs', 8) for row in rows}
    attempt_caps = {row.get('max_responses', 96) for row in rows}
    if len(pair_counts) != 1 or len(attempt_caps) != 1:
        raise ValueError('A metric report cannot mix board sizes or attempt limits')
    total_pairs, max_responses = pair_counts.pop(), attempt_caps.pop()
    attempts = sum(row['response_count'] for row in rows)
    tokens = sum(row['generated_tokens'] for row in rows)
    entropy_tokens = sum(row.get('entropy_tokens', 0) for row in rows)
    complete = [row for row in rows if row['done']]
    return dict(episodes=count, total_pairs=total_pairs, max_responses=max_responses,
        mean_return=float(np.mean([row['training_return'] for row in rows])),
        reward_std=float(np.std([row['training_return'] for row in rows])),
        mean_pairs=float(np.mean([row['score'] for row in rows])),
        matched_fraction=float(np.mean([row['score'] / total_pairs for row in rows])),
        completion_rate=len(complete) / count, mean_attempts=attempts / count,
        completed_only_mean_attempts=float(np.mean([row['response_count'] for row in complete])) if complete else None,
        completed_only_denominator=len(complete),
        resource_terminations=sum(bool(row.get('censored')) for row in rows),
        native_invalid_actions=sum(row['invalid_actions'] for row in rows),
        format_errors=sum(row['format_errors'] for row in rows),
        format_error_rate=sum(row['format_errors'] for row in rows) / max(attempts, 1),
        generated_tokens=tokens, raw_transcript_tokens=sum(row['raw_transcript_tokens'] for row in rows),
        entropy_tokens=entropy_tokens,
        mean_policy_entropy_at_temperature=sum((row.get('mean_policy_entropy_at_temperature') or 0.) * row.get('entropy_tokens', 0)
            for row in rows) / entropy_tokens if entropy_tokens else None,
        mean_tokens_per_game=tokens / count,
        mean_per_game_tokens_per_second=float(np.mean([row['generated_tokens'] / max(row['seconds'], 1e-9) for row in rows])))


def protocol_lock(config):
    source = Path(config.source)
    native = json.loads((source / 'SOURCE.json').read_text())
    if native['commit'] != 'e3d195e857727dfa3d4a96561be202098d67da83':
        raise ValueError('Native source revision changed')
    for row in native['manifest']:
        if hashlib.sha256((source / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Native source changed: {row["path"]}')
    model = Path(config.model)
    names = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    return dict(native=native, model_revision=config.revision,
        model_files={name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                     for name in names if (model / name).exists()})
