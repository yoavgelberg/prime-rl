"""Shared configuration for the native Matching Pairs GRPO experiment."""

import math
from dataclasses import dataclass
from pathlib import Path

MODEL_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
NATIVE_REVISION = "e3d195e857727dfa3d4a96561be202098d67da83"
THINKING_TRANSITION = "\n\nOkay, time to give the answer.</think>\n\n"


@dataclass(frozen=True)
class HeroConfig:
    model: str
    source: str
    output: str
    node_dir: str
    source_sha256: str
    deadline: float
    gpus: int = 8
    seed: int = 11
    updates: int = 128
    boards: int = 16
    group_size: int = 8
    denominator: int = 26624
    window: int = 1024
    transcript_cap: int = 262144
    max_responses: int = 96
    thinking_token_limit: int = 128
    action_token_limit: int = 32
    temperature: float = 0.6
    learning_rate: float = 2e-6
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_gradient_norm: float = 1.0
    warmup_updates: int = 8
    schedule_updates: int | None = None
    minimum_lr_fraction: float = 0.2
    continuation_start_update: int | None = None
    continuation_warmup_updates: int = 8
    continuation_initial_lr: float = 4e-7
    clip_ratio: float = 0.2
    kl_coefficient: float = 0.01
    entropy_coefficient: float = 0.001
    format_penalty: float = 0.02
    invalid_penalty: float = 0.02
    head_block_tokens: int = 128
    checkpoint_interval: int = 8
    checkpoint_io_lock: str | None = None
    export_interval: int = 16
    wandb_id: str = "rng-4b-ps-grpo-s11-hero-v2-a1"
    revision: str = MODEL_REVISION
    defer_native_logs: bool = False
    accepted_resume_sources: tuple[str, ...] = ()
    exact_kl_start_update: int | None = None
    accepted_kl_schedule_sources: tuple[str, ...] = ()

    def __post_init__(self):
        positive_counts = (
            "denominator",
            "transcript_cap",
            "warmup_updates",
            "checkpoint_interval",
            "export_interval",
            "head_block_tokens",
        )
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in positive_counts):
            raise ValueError("Update, token, warmup, and checkpoint counts must be positive integers")
        if type(self.updates) is not int or self.updates < 1 or self.updates * self.boards > 1_000_000:
            raise ValueError("Updates must be positive integers within the million-board pool")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        object.__setattr__(self, "accepted_resume_sources", tuple(self.accepted_resume_sources))
        object.__setattr__(self, "accepted_kl_schedule_sources", tuple(self.accepted_kl_schedule_sources))
        if self.exact_kl_start_update is not None and self.exact_kl_start_update < 1:
            raise ValueError("Exact KL needs a positive explicit activation update")
        if any(len(value) != 64 for value in self.accepted_kl_schedule_sources):
            raise ValueError("KL migration sources must be reviewed SHA256 digests")
        if self.gpus not in (4, 8) or self.boards != 16 or self.group_size != 8:
            raise ValueError("Hero uses16boards×8rollouts with4or8allocatedGPUs")
        if (
            Path(self.model).name != self.revision
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("Pin the original model revision and captured implementation")
        if (
            self.window not in (64, 128, 256, 512, 1024, 2048, 4096)
            or self.thinking_token_limit != 128
            or self.max_responses != 96
        ):
            raise ValueError("Keep the accepted memory task and thinking budget")
        if self.schedule_updates is not None and self.schedule_updates <= self.warmup_updates:
            raise ValueError("The fixed learning-rate horizon must exceed warmup")
        if self.continuation_start_update is not None and (
            self.continuation_start_update < 0
            or self.continuation_warmup_updates < 1
            or not 0 < self.continuation_initial_lr <= self.learning_rate
        ):
            raise ValueError("Continuation warmup needs a positive duration and an initial rate below its target")
        if self.boards % self.trainers or self.group_size < 2 or self.action_token_limit < 1:
            raise ValueError("Every trainer needs complete groups and a finite answer budget")
        if min(self.learning_rate, self.kl_coefficient, self.entropy_coefficient, self.denominator) <= 0:
            raise ValueError("Learning rate,KL,entropy and denominator must be positive")
        if type(self.defer_native_logs) is not bool or any(len(value) != 64 for value in self.accepted_resume_sources):
            raise ValueError("Logging mode must be boolean and reviewed resume sources must be SHA256 digests")

    @property
    def batch_size(self):
        return self.boards * self.group_size

    @property
    def actors(self):
        return self.gpus // 2

    @property
    def trainers(self):
        return self.gpus // 2

    @property
    def live_games(self):
        return self.batch_size // self.actors
