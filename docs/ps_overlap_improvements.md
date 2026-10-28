# Experimental streaming backward optimizations

This branch adds selectable scheduling and backward-work reductions to the published FA4-PS streaming implementation. It preserves the full prefix/sliding dependency graph and the frozen-policy, delayed scalar-advantage objective. Numerical parity with a separately recomputed native forward remains unresolved as documented in [ps_streaming.md](ps_streaming.md). The new path is experimental: CPU tests, Hopper converter kernels and H200 engine audits with FP32 attention gradients pass; long-trace B200 performance remains pending.

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

All switches default off, with the previous SM transfer mode retained. Hardware and engine restrictions are those of the original adapter. `copy_mode="dma"` selects asynchronous DMA transfers; neither route guarantees interference-free overlap. `copy_programs=0` retains an unrestricted mapped-copy grid, while a positive value limits concurrently launched programs per transfer kernel. The separate high-priority ring snapshots use the same limit.

The changes are:

- Joint QKV producer backward retains dQ until dK/dV are complete, then uses a packed input-gradient GEMM and one shared input-normalization VJP. No earlier layer can become ready before that point in the existing full-gradient schedule.
- Direct weight-gradient accumulation adds GEMM results into the current trajectory's FP32 gradient buffers. A graph input contains destination pointers, so shared CUDA graphs can select different trajectories without copying complete temporary weight-gradient matrices. The `cublas` backend uses device pointer arrays and a caller-owned workspace; `triton` uses the included tensor-core kernel and can write packed QKV gradients in one launch. CPU FP64 uses an independent addmm path. Changing the policy still requires the original reset/rewarm contract. The device pointer array interface is documented by [NVIDIA](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex).
- Compact history reads transfer only rotated K and raw V. A bounded per-layer frontier cache retains these values until their last local consumer; prefix values remain until EOF. Captured attention inputs receive K/V pieces directly, avoiding temporary concatenations and separate dtype-conversion buffers.
- Separate generation-ready, per-layer offload-ready, head-ready and reader-completion events allow GPU-resident work to begin before the full chunk finishes spilling. Host loads wait for their own offload event, and generation still waits before overwriting a protected ring location. Drain includes outstanding copies before storage reset.
- GPU head staging retains logits, normalized states, raw states and token IDs in a bounded two-chunk-per-request pool. Head backward releases this storage after producing its hidden-state adjoint; persistent input IDs are cloned before pool reuse.
- The stage scheduler skips unready requests, prioritizes ready heads and older records, and fairly rotates equivalent ready requests. It bounds unfinished submitted GPU stages using completion events. Queued chunks of the same trajectory can be grouped; each request has one active retirement iterator. Finished requests yield between tail layers, with only one tail prefetch owner to preserve the memory bound.

The retirement deadline and cross-chunk/cross-layer derivatives are unchanged mathematically. Fusion and accumulation change floating-point reduction order. The expected storage bound includes the original PS dependency band, the bounded head and KV pools, per-trajectory full-model gradients, and the configured GPU work credits. Larger buffers cannot compensate for backward being slower than the generation arrival rate indefinitely.

CPU validation covers FP64 complete gradients, independent interleaved trajectory accumulators, grouped retirement, graph-input packing, scheduler readiness and work credits. It also covers nested gradient scopes, restoring trajectory pointers and architecture selection; the current suite has 159 passing CPU cases. The Triton kernel has three passing CPU-interpreter indexing tests. The H200 suite passes 227 CPU/GPU cases, including transfers, ring/head ownership, both direct GEMM backends, FP32 attention and graph destination switching. All five real-vLLM variants pass same-recorded-value full-gradient audits at relative L2 0.00576–0.00668 on 3072/4097-token traces with FP32 attention gradients; the combined 12k/16k systems smoke releases all records. Long-trace B200 accuracy and performance remain pending. No online RL has run.

The FP32 FA4 accumulator converter has separate Hopper and Blackwell paths. Hopper reads WGMMA fragments using FA4's own MMA coordinate layout and stores their scaled FP32 values directly, including swapped and padded tiles. Blackwell retains its existing TMEM converter. On one H100, 17 GPU tests pass: seven WGMMA layout conversions match the upstream postprocessor after BF16 rounding, eight causal/noncausal cases match dense gradients within the BF16 tolerance and replay under CUDA graphs, and two recorded-attention VJP cases pass. The complete H200 real-engine FP32 qualification also passes. Synthetic TMEM tests remain Blackwell-only.
