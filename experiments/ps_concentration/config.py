from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_SHA256 = "23c010fe308ce960ac08af49666423ecd200449d8743c101f4d80ff223c43f99"
SPLITS = dict(screening=range(1_000_000, 1_000_064), validation=range(1_100_000, 1_100_128),
              test=range(2_000_000, 2_000_512), larger=range(3_000_000, 3_000_256),
              intervention=range(4_000_000, 4_000_512))


def system_prompt(cards=52, limit=104, prompted=False):
    prompt = (f"You are playing a card matching game. There are {cards} positions, numbered 0 through {cards-1}, "
              "and 13 ranks, numbered 0 through 12. There are "
              f"{'four' if cards == 52 else 'eight'} cards of each rank. Each action flips one position. "
              "Two successive valid flips form a pair. Matching ranks at distinct positions stay revealed and earn reward. "
              "An unsuccessful pair is shown in the observation and then turns face down before the next pair. "
              "Permanently matched cards cannot be used again. "
              f"Maximize total reward before the {limit}-flip limit. You may write scratch work before your action. "
              "End your response with exactly one line in this format:\nFLIP 17\n"
              "Replace 17 with your chosen position.")
    if prompted:
        prompt += "\nKeep an updated compact list of revealed unmatched card positions and ranks in your scratch work."
    return prompt


@dataclass
class Config:
    model: str = MODEL
    revision: str = ""
    attention: str = "ps"
    window: int = 1024
    max_new_tokens: int = 256
    transcript_cap: int = 65536
    cards: int = 52
    limit: int = 104
    boards: int = 8
    rollouts: int = 8
    seed: int = 11
    updates: int = 32
    learning_rate: float = 5e-6
    warmup: int = 8
    kl_coefficient: float = .01
    denominator: int = 104 * 256
    temperature: float = .8
    top_p: float = .95
    beta1: float = .9
    beta2: float = .95
    epsilon: float = 1e-8
    weight_decay: float = 0.
    max_grad_norm: float = 1.
    arithmetic: str = 'bf16_matmul_attention_fp32_residual_and_master_weights'
    allow_numerical_target_miss: bool = False
    decode_backend: str = 'eager'
    gpu_layout: str = 'colocated_data_parallel'
    rollout_batch_size: int = 8
    rollout_prefill_tokens: int = 0
    training_workers: int = 1
    source_sha256: str = ''

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def generation_seed(training_seed, update, board_index, rollout_index):
    text = f"concentration-v1:{training_seed}:{update}:{board_index}:{rollout_index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)
