"""Configuration for one fixed-policy cohort of streaming rollouts."""

from dataclasses import dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class StreamConfig:
    window: int = 2048
    chunk_tokens: int = 1024
    ring_tokens: int = 2048
    max_requests: int = 8
    tail_block_tokens: int = 16384
    fp32_attention: bool = True
    joint_qkv: bool = False
    direct_weight_gradients: bool = False
    weight_gradient_backend: str = "triton"
    compact_kv: bool = False
    kv_cache: bool = False
    fine_grained_events: bool = False
    gpu_head_staging: bool = False
    stage_scheduling: bool = False
    copy_mode: str = "sm"
    copy_programs: int = 0
    gpu_stages_in_flight: int = 4
    backward_batch_tokens: int = 1024

    def __post_init__(self):
        for name in (
            "window",
            "chunk_tokens",
            "ring_tokens",
            "max_requests",
            "tail_block_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ring_tokens < self.chunk_tokens or self.ring_tokens % self.chunk_tokens:
            raise ValueError("The GPU ring must hold an integral positive number of chunks")
        if self.tail_block_tokens < self.chunk_tokens:
            raise ValueError("Tail batches must hold at least one chunk")
        for name in (
            "fp32_attention",
            "joint_qkv",
            "direct_weight_gradients",
            "compact_kv",
            "kv_cache",
            "fine_grained_events",
            "gpu_head_staging",
            "stage_scheduling",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.copy_mode not in ("sm", "dma"):
            raise ValueError("copy_mode must be sm or dma")
        if self.weight_gradient_backend not in ("triton", "cublas"):
            raise ValueError("weight_gradient_backend must be triton or cublas")
        if type(self.copy_programs) is not int or self.copy_programs < 0:
            raise ValueError("copy_programs must be a nonnegative integer")
        if type(self.gpu_stages_in_flight) is not int or self.gpu_stages_in_flight < 1:
            raise ValueError("gpu_stages_in_flight must be a positive integer")
        if type(self.backward_batch_tokens) is not int or self.backward_batch_tokens < 1:
            raise ValueError("backward_batch_tokens must be a positive integer")

    def validate_hardware(self, device, *, query_heads, kv_heads, head_dim):
        import torch

        if torch.device(device).type != "cuda":
            raise ValueError("The vLLM streaming adapter requires CUDA")
        if version("flash-attn-4") != "4.0.0b36":
            raise ValueError("This adapter uses private FA4 APIs pinned to flash-attn-4==4.0.0b36")
        major, _ = torch.cuda.get_device_capability(device)
        if major not in (9, 10, 11):
            raise ValueError("FA4-PS requires Hopper or Blackwell")
        if self.fp32_attention and (major not in (10, 11) or query_heads <= kv_heads or head_dim == 256):
            raise ValueError("FP32 FA4 adjoints require Blackwell, GQA and head_dim != 256")
