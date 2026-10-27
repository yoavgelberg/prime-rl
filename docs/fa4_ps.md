# FA4 Prefix Sliding baseline

This adapter uses upstream FlashAttention-4 with the exact Prefix Sliding (PS) mask and sparse forward/backward tile lists. It does not replace FA4's attention kernel. For query position `q` and key position `k`, an edge exists exactly when

```text
0 <= k <= q and (k < prefix_length or k >= q - window + 1).
```

The prefix is always visible; the rest of the sequence has a causal sliding window. Original positions remain available for RoPE even when a training crop omits earlier completion tokens.

## Files

| File | Responsibility |
| --- | --- |
| `src/prime_rl/trainer/ps_backward/fa4.py` | Exact mask, sparse forward/backward metadata, tile alignment, upstream FA4 call |
| `src/prime_rl/trainer/ps_backward/paged.py` | Compact paged KV metadata and the same PS rule during generation |
| `src/prime_rl/inference/vllm/ps_fa4_backend.py` | Thin backend for the pinned PS-vLLM fork |
| `src/prime_rl/trainer/ps_backward/model.py` | Qwen2/Qwen3 weight adapter; ordinary differentiable forward |
| `src/prime_rl/trainer/ps_backward/baseline.py` | `backward_full` and `backward_truncated` |

`backward_full` differentiates every completion token through the entire PS graph. Layer checkpointing trades recomputation for activation memory; `retain_last_layers` keeps a chosen number of final layers. There is no detached KV boundary.

`backward_truncated` trains only the last W completion labels, using 3W preceding completion-query positions plus the full prompt. The completion crop therefore covers up to 4W query positions. It normalizes over selected labels and is approximate on long traces: earlier losses and their paths are absent, and deep dependencies extend beyond the crop.

```python
import torch

from prime_rl.trainer.ps_backward.baseline import backward_full, backward_truncated
from prime_rl.trainer.ps_backward.loss import LossConfig
from prime_rl.trainer.ps_backward.model import QwenPS
from prime_rl.trainer.ps_backward.types import Trace

# base is a Transformers Qwen2/Qwen3 model on CUDA. token_ids includes prompt + completion.
model = QwenPS(base, window=2048, checkpoint_layers=True, retain_last_layers=6)
trace = Trace(token_ids, prefix_length).to(model.device)
with torch.autocast('cuda', dtype=torch.bfloat16):
    result = backward_full(model, trace, LossConfig(head_block_tokens=1024))
# Apply your existing batch reduction, clipping and optimizer outside this function.
```

The dense attention implementation is a small independent test reference. It constructs a token-by-token mask and must not be selected for long traces.

## Environment and installation

The reviewed base is Prime-RL `5a9434a8fc49e7ac3500988c02ec4e0b43dce869`. The generation adapter depends on the PS-vLLM fork at `22abbc6ca5f77536aaeb91160667a4684e2bd694` (tested binary revision `06d490282f2bab6922137eb5230be9df5ebbe9c4`). Stock vLLM does not supply the per-request `prompt_lens` metadata or PS cache behavior used here. Apply this patch in the existing PS stack.

The GPU environment tested with this implementation is Python 3.12, Torch 2.9.1+cu128, Transformers 4.57.1, `flash-attn-4==4.0.0b36`, `nvidia-cutlass-dsl==4.8.1`, `quack-kernels==0.6.5`, and Triton 3.5.1. Install the patched Prime-RL package into that environment with `uv pip install --python /path/to/environment/bin/python --no-deps -e .`. The plugin registers `CUSTOM`; choose that backend explicitly. The PS-vLLM fork also reads `SWF` before imports, which must equal the configured window. Ordinary FA4 generation uses the normal vLLM worker, without recording hooks.

Upstream [FA4 documentation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md) targets Hopper and Blackwell. This adapter accepts compute-capability major versions 9, 10 and 11. It requires CUDA BF16/FP16, unquantized decoder attention and no decode context parallelism. It does not provide an AMD/CPU attention fast path. Kernel support and measured performance are separate claims; our long-trace timing evidence is on one B200.

## Tests

From the repository root, run the isolated test project so Prime-RL's parent test fixtures do not affect running inference services:

```bash
uv run --project tests/ps_backward --extra cpu --extra kernel-debug \
  python -m pytest --confcutdir=tests/ps_backward -c tests/ps_backward/pytest.ini \
  tests/ps_backward -m 'not gpu' -q
```

For GPU checks, use the pinned environment's Python with the same explicit `--confcutdir`, and omit `-m 'not gpu'`. The tests compare FA4 outputs and all Q/K/V gradients with an independent dense reference, check sparse metadata and tile padding, and distinguish full from truncated training.
