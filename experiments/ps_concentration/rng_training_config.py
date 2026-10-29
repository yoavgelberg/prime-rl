"""Frozen native-game training protocol, separate from POPGym settings."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass
class MatchingConfig:
    model: str
    source: str
    revision: str = '768f209d9ea81521153ed38c47d515654e938aea'
    task: str = 'RNGBench textual Matching Pairs 4x4'
    attention: str = 'ps'
    window: int = 1024
    transcript_cap: int = 262144
    max_responses: int = 96
    max_new_tokens: None = None
    seed: int = 11
    updates: int = 32
    boards: int = 8
    rollouts: int = 8
    temperature: float = .6
    top_p: float = .95
    top_k: int = 20
    learning_rate: float = 5e-6
    beta1: float = .9
    beta2: float = .95
    epsilon: float = 1e-8
    weight_decay: float = 0.
    warmup: int = 8
    kl_coefficient: float = .01
    denominator: int = 26624
    max_grad_norm: float = 1.
    context_exhaustion_reward: float | None = None
    activation_cpu_offload: bool = True
    validation_interval: int = 16
    validation_boards: int = 128
    training_likelihood: str = 'temperature-scaled full-softmax; same definition as existing trainer'
    kl_estimator: str = 'k3: exp(reference_logp-policy_logp)-(reference_logp-policy_logp)-1'
    reward_definition: str = 'pairs/8 - attempted_responses/96; resource termination configured separately'
    arithmetic: str = 'bf16_products_attention_fp32_residual_projection_accumulation_and_master_weights'

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')


def training_rng(training_seed, update, board_index, rollout_index):
    text = f'rng-matching-training-v1:{training_seed}:{update}:{board_index}:{rollout_index}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63 - 1)


def outcome_reward(record, max_responses=96, context_exhaustion_reward=None):
    if record.get('censored', False):
        if context_exhaustion_reward is None:
            raise ValueError('Context exhaustion must have an explicit recorded treatment before RL')
        return float(context_exhaustion_reward)
    assert 0 <= record['score'] <= 8
    assert 0 < record['response_count'] <= max_responses
    return record['score'] / 8 - record['response_count'] / max_responses
