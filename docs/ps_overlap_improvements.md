# Experimental streaming backward optimizations

This branch adds selectable scheduling and backward-work reductions to the published FA4-PS streaming implementation. It preserves the full prefix/sliding dependency graph and the frozen-policy, delayed scalar-advantage objective. Numerical parity with a separately recomputed native forward remains unresolved as documented in [ps_streaming.md](ps_streaming.md). The complete 12-round, three-B200 campaign measures **3.13× less added training cost** than full FA4-PS, with a prespecified one-sided 95% bootstrap lower bound of **2.07×**; the measured complete cycle is **6.63% shorter**. This qualifies the performance gate within that one-host campaign. [All results and limitations](ps_overlap_b200_results.md).

The added-cost metric compares different native and recording-enabled forward paths. Its generation-delta clamp does not isolate backward overhead, so the 3.13× figure is not a speedup of backward computation in isolation. See the result report for this qualification.

```python
options = StreamConfig(
    window=2048, chunk_tokens=1024, ring_tokens=2048, max_requests=8,
    fp32_attention=True,
    joint_qkv=True, direct_weight_gradients=True, weight_gradient_backend="cublas",
    compact_kv=True, kv_cache=True,
    fine_grained_events=True, gpu_head_staging=True,
    stage_scheduling=True, gpu_stages_in_flight=4,
    backward_batch_tokens=4096,
    copy_mode="sm", copy_programs=8,
)
```

The real AsyncLLM example selects these settings with `--streaming --optimized`:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=32 \
uv run --active --no-sync python examples/ps_backward/generate.py \
  --model MODEL --revision REVISION --prompts prompts.jsonl \
  --length 131072 --requests 8 --streaming --optimized --output optimized
```

Use the pinned environment and lifecycle in [ps_streaming.md](ps_streaming.md). The example measures generation and drain; the result table additionally includes gradient reduction and discarded optimizer updates. Eight long traces need approximately 152 GiB peak GPU allocation and 491 GiB pinned-host storage in the measured combined configuration. Later memory-reduction and selective-field experiments are separate and are excluded from these results.

All switches default off, with the previous SM transfer mode retained. Hardware and engine restrictions are those of the original adapter. `copy_mode="dma"` selects asynchronous DMA transfers; neither route guarantees interference-free overlap. `copy_programs=0` retains an unrestricted mapped-copy grid, while a positive value limits concurrently launched programs per transfer kernel. The separate high-priority ring snapshots use the same limit.

The changes are:

- Joint QKV producer backward retains dQ until dK/dV are complete, then uses a packed input-gradient GEMM and one shared input-normalization VJP. No earlier layer can become ready before that point in the existing full-gradient schedule.
- Direct weight-gradient accumulation adds GEMM results into the current trajectory's FP32 gradient buffers. A graph input contains destination pointers, so shared CUDA graphs can select different trajectories without copying complete temporary weight-gradient matrices. The `cublas` backend uses device pointer arrays and a caller-owned workspace; `triton` uses the included tensor-core kernel and can write packed QKV gradients in one launch. CPU FP64 uses an independent addmm path. Changing the policy still requires the original reset/rewarm contract. The device pointer array interface is documented by [NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex).
- Compact history reads transfer only rotated K and raw V. A bounded per-layer frontier cache retains these values until their last local consumer; prefix values remain until EOF. Captured attention inputs receive K/V pieces directly, avoiding temporary concatenations and separate dtype-conversion buffers.
- Separate generation-ready, per-layer offload-ready, head-ready and reader-completion events allow GPU-resident work to begin before the full chunk finishes spilling. Host loads wait for their own offload event, and generation still waits before overwriting a protected ring location. Drain includes outstanding copies before storage reset.
- GPU head staging retains logits, normalized states, raw states and token IDs in a bounded two-chunk-per-request pool. Head backward releases this storage after producing its hidden-state adjoint; persistent input IDs are cloned before pool reuse.
- The stage scheduler skips unready requests, prioritizes ready heads and older records, and fairly rotates equivalent ready requests. It bounds unfinished submitted GPU stages using completion events. Queued chunks of the same trajectory can be grouped; each request has one active retirement iterator. Finished requests yield between tail layers, with only one tail prefetch owner to preserve the memory bound.

The retirement deadline and cross-chunk/cross-layer derivatives are unchanged mathematically. Fusion and accumulation change floating-point reduction order. The expected storage bound includes the original PS dependency band, the bounded head and KV pools, per-trajectory full-model gradients, and the configured GPU work credits. Larger buffers cannot compensate for backward being slower than the generation arrival rate indefinitely.

CPU validation covers FP64 complete gradients, independent interleaved trajectory accumulators, grouped retirement, graph-input packing, scheduler readiness and work credits. It also covers nested gradient scopes, restoring trajectory pointers and architecture selection; the current suite has 159 passing CPU cases. The Triton kernel has three passing CPU-interpreter indexing tests. The H200 suite passes 227 CPU/GPU cases, including transfers, ring/head ownership, both direct GEMM backends, FP32 attention and graph destination switching. All five real-vLLM variants pass same-recorded-value full-gradient audits at relative L2 0.00576–0.00668 on 3072/4097-token traces with FP32 attention gradients; the combined 12k/16k systems smoke releases all records. The B200 131072-token same-value audit passes at relative L2 0.00644481, and all eight published/improved token sequences match in every one of the 12 completed timing rounds. No online RL has run.

The FP32 FA4 accumulator converter has separate Hopper and Blackwell paths. Hopper reads WGMMA fragments using FA4's own MMA coordinate layout and stores their scaled FP32 values directly, including swapped and padded tiles. Blackwell retains its existing TMEM converter. On one H100, 17 GPU tests pass: seven WGMMA layout conversions match the upstream postprocessor after BF16 rounding, eight causal/noncausal cases match dense gradients within the BF16 tolerance and replay under CUDA graphs, and two recorded-attention VJP cases pass. The complete H200 real-engine FP32 qualification also passes. Synthetic TMEM tests remain Blackwell-only.
