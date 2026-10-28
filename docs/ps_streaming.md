# Streaming backward during PS generation

Streaming backward reuses the forward values produced by vLLM, starts differentiating before rewards arrive, and delays each KV producer until every query that can use it has contributed its gradient. The attention calculation remains FA4. The approach aims to avoid another training forward pass and move backward work into generation, with a bounded activation lifetime. The completed timing retest below did not establish a reliable speedup.

The implementation is a reusable PS core plus a thin dense-Qwen3/vLLM adapter. It has no hard-coded model identifier, revision, hidden width, layer count, vocabulary or trace length. A different model architecture needs its own recorder and weight adapter. This is a fixed-policy cohort implementation; it does not implement a complete RL trainer, arbitrary request-slot reuse, tensor/pipeline parallelism, quantized weights or speculative decoding.

## Walkthrough

1. **Generate with FA4-PS.** `ps_fa4_backend.py` compacts the paged prefix/window KV view and calls FA4. The native baseline uses this same backend without activation recording. The mask and training baselines are described in [fa4_ps.md](fa4_ps.md).
2. **Record the actual forward values.** `ps_capture.py` adapts Qwen3's forward. `ps_capture_kernels.py` adds stores to the RMSNorm and SwiGLU launches already needed for generation. It records Q/K/V, attention outputs and log-sum-exp normalizers, normalized states and MLP inputs. Compact storage omits redundant residual/SwiGLU outputs; cheap pointwise operations reconstruct these exactly in the recorded dtype. Matrix multiplications and attention forward are not replayed.
3. **Keep a small GPU ring and bounded host pools.** `ps_host_tape.py` publishes complete chunks to pinned host storage. CUDA events prevent reuse until copies and readers finish. Recent records can be read directly from the GPU ring. `ps_mapped_copy.py` supplies the NVIDIA mapped-memory copy kernel. Pool capacities depend on depth, window, chunk and cohort size, rather than total trace length. The constant is substantial: the historical eight-request configuration used approximately 548 GB of pinned host memory.
4. **Start backward with known sampled-token labels.** `ps_live.py` runs a background worker on a separate CUDA stream. `recorded.py` defines VJPs whose forward returns the recorded value, so backward can compute parameter and input derivatives without repeating the generating GEMM. `recorded_pointwise.py` fuses RMSNorm, RoPE and SwiGLU backward. `recorded_graphs.py` captures repeated chunk stages in CUDA graphs.
5. **Retire dependencies only when complete.** `retirement.py` handles the graph independently of the GPU runtime. `recorded_stream.py` first consumes a query block: it differentiates the MLP/output/query path and accumulates dK/dV into every relevant producer block. For a nonprefix block ending at exclusive index `end`, its KV producer can finish only when query gradients through position `end + W - 2` are complete at that layer. The full gradient then becomes available to the next lower layer. Prefix KV stays live until the sequence ends because every future query can use it. This preserves cross-chunk and cross-layer derivatives.
6. **Drain and apply the eventual scalar advantages.** At sequence end, `recorded_tail.py` groups the remaining work into larger layer-wise batches and prefetches one layer ahead. Each trajectory then has a complete parameter-gradient vector. `delayed_reward.py` can combine reward groups; the worker's export method can apply supplied scalar advantages directly. There is no optimizer update hidden in either API.

The scheduler's activation bound, measured in token-layers, is

```text
prefix * depth + chunk * depth
    + ceil((W - 1) / chunk) * chunk * depth * (depth + 1) / 2.
```

Thus storage is bounded in trace length, with a depth-squared dependency band. Full-model gradient accumulators are also needed for each unfinished trajectory. Bounded memory does not imply a small footprint.

## What can be computed before the reward?

For each trajectory i, the implementation accumulates the mean negative-score gradient

```text
g_i = gradient_theta[-sum_t log pi_theta(a_it | prefix_it) / T_i].
```

Once scalar advantages arrive, the batch-mean loss gradient is `sum_i A_i * g_i / N`. This uses linearity of differentiation. It does not build a full logit Jacobian. It supports delayed sequence/group advantages at a frozen policy; it does not make later arbitrary token-specific weights, changed-policy PPO epochs or additional unknown loss terms free.

Floating-point reductions can differ from a monolithic backward. Exactness here means preserving every graph dependency, with explicit numerical audits. The accepted historical 131072-token audit compared against an unfused full backward using the **same recorded generation values**: relative L2 0.007389 and 0.007980, cosine above 0.99996. A separate comparison to a newly recomputed Hugging Face forward had approximately 0.17 relative L2 and remains unresolved. Do not interpret the same-value audit as numerical equivalence to that different forward implementation.

## Configuration and API

Use `StreamConfig` under vLLM's `additional_config['ps_streaming']` and select `prime_rl.inference.vllm.ps_streaming_worker.PSStreamingWorker` as `worker_cls`. Select the `CUSTOM` FA4 backend. All dimensions come from the active engine configuration. The backward model is constructed from that configuration and copies the **current inference weights**, including vLLM's fused QKV and gate/up matrices.

```python
from dataclasses import asdict
from vllm.config import AttentionConfig

from prime_rl.trainer.ps_backward.config import StreamConfig

options = StreamConfig(window=2048, chunk_tokens=1024, ring_tokens=2048,
                       max_requests=8, tail_block_tokens=16384,
                       fp32_attention=True)
# AsyncEngineArgs(..., attention_config=AttentionConfig(backend='CUSTOM'),
#     worker_cls='prime_rl.inference.vllm.ps_streaming_worker.PSStreamingWorker',
#     additional_config={'ps_streaming': asdict(options)})
```

The intended lifecycle through `engine.collective_rpc` is:

```text
ps_stream_begin(policy_version=version, warmup=True)
    generate representative warmup requests; finish each request
ps_stream_finish()
ps_stream_prepare_graphs()
ps_stream_reset()
    generate the real cohort; finish each request
ps_stream_finish()
ps_stream_export_gradients(advantages={exact_request_id: scalar, ...}, policy_version=version)
```

Call `ps_stream_request_finished(request_id)` when each native `AsyncLLM.generate` iterator completes. `ps_stream_finish` also finalizes remaining observed requests. The report gives exact internal request IDs for the advantage map. Gradient export returns CPU tensors and incurs extra transport time; it is optional and is outside historical timing. After a warmup, reset reuses pools and graphs at the same frozen policy. Changing policy requires a fresh recording/backward setup and rewarming; no in-place policy-refresh protocol is provided here.

The adapter requires TP=PP=1, BF16 dense Qwen3, dropout off, ordinary unfiltered unit-temperature sampling, and a fresh prefill with no prefix-cache hit. The prompt must fit `chunk_tokens`; increase both chunk and ring sizes for longer prompts. Use a fixed cohort no larger than `max_requests`; overflow is an error. Keep inference weights unchanged until draining. The explicit policy version and backward-parameter version checks do not monitor arbitrary external in-place edits to vLLM weights.

A runnable example uses the real AsyncLLM engine, JSONL rows containing `prompt_ids`, and a user-supplied model/revision:

```bash
uv run --active --no-sync python examples/ps_backward/generate.py \
  --model MODEL --revision REVISION --prompts prompts.jsonl \
  --length 131072 --requests 8 --output native
uv run --active --no-sync python examples/ps_backward/generate.py \
  --model MODEL --revision REVISION --prompts prompts.jsonl \
  --length 131072 --requests 8 --streaming --output streaming
```

Use the pinned environment described in [fa4_ps.md](fa4_ps.md). For overlap timing, set `CUDA_DEVICE_MAX_CONNECTIONS=32` and `CUDA_DEVICE_MAX_COPY_CONNECTIONS=32` **before** starting the process, and hold CPU threads and resource allocation fixed. The example deliberately forces completion lengths and performs no rewards or optimizer steps. It measures generation and backward drain; it does not reproduce the historical optimizer-inclusive table by itself. The optional test-only `audit_worker.AuditWorker` retains a full tape and must never be used for performance measurements.

## Hardware boundaries

| Component | Requirement / evidence |
| --- | --- |
| PS mask, retirement rule, delayed scalar-gradient algebra | Hardware-independent mathematics |
| FA4 attention integration | NVIDIA Hopper/Blackwell; CUDA BF16/FP16 |
| Fused pointwise/recording kernels | Triton/CUDA; validated on H200 and B200 in the research implementation |
| Mapped-memory transfer path | Linux/NVIDIA CUDA and pinned, mapped host memory |
| FP32 FA4 adjoint conversion | Hopper/Blackwell, GQA, head dimension other than 256, pinned FA4 4.0.0b36 internals |
| Long-trace timing | One B200; results varied widely, with a mean slowdown in the completed retest |

`fa4_adjoint.py` confines the private FA4 hook using a context variable. `fa4_fp32_postprocess.py` isolates accumulator conversion and includes the upstream license attribution. Its Blackwell path reads the TMEM layouts; `fa4_hopper_postprocess.py` derives Hopper's WGMMA coordinate map from the pinned FA4 tile geometry and directly stores FP32 values. The attention tensor-core computation remains upstream FA4. `fp32_attention=False` explicitly selects native FA4 gradient outputs. The Hopper converter passes 17 allocated-H100 kernel/gradient/graph tests and the complete H200 real-engine FP32 qualification. There is no silent precision fallback.

## Completed retest of the published implementation

Twelve alternating-order pairs ran the published implementation on one B200 with Qwen3-1.7B, window 2048 and eight 96k/128k completions in vLLM AsyncLLM. These are fixed-policy systems measurements, not online RL. Generation plus update is constructed from measured components; startup, warmup, rewards, weight broadcast and trace serialization are excluded. Streaming cost charges generation slowdown when positive, the full drain RPC, gradient reduction and a discarded optimizer update. The full FA4 control uses the faster median of two warmed full-update variants.

| Pair | Full FA4 added cost | Streaming added cost | Full FA4 cycle | Streaming cycle |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34.97 s | 139.87 s | 852.03 s | 956.93 s |
| 2 | 35.07 s | 147.41 s | 843.26 s | 955.60 s |
| 3 | 35.05 s | 144.11 s | 851.85 s | 960.92 s |
| 4 | 35.49 s | 4.65 s | 851.03 s | 818.82 s |
| 5* | 35.14 s | 102.56 s | 838.30 s | 905.72 s |
| 6 | 35.11 s | 155.56 s | 831.02 s | 951.47 s |
| 7 | 35.14 s | 4.59 s | 851.34 s | 815.63 s |
| 8 | 35.12 s | 17.32 s | 833.20 s | 815.40 s |
| 9 | 35.04 s | 24.73 s | 828.16 s | 817.85 s |
| 10 | 35.01 s | 4.66 s | 851.41 s | 817.29 s |
| 11 | 35.09 s | 5.67 s | 847.64 s | 818.21 s |
| 12 | 35.21 s | 6.23 s | 848.49 s | 819.52 s |
| **Mean** | **35.12 s** | **63.11 s** | **843.98 s** | **871.11 s** |

The mean streaming cycle was **3.2% slower**, despite winning seven of twelve individual pairs. Pair 5 had a roughly 12-second overlapping GPU diagnostic with unknown effect; excluding it still gives a 2.8% mean regression. Smaller diagnostic overlaps occurred in some earlier pairs. The slow and fast streaming runs produced identical token sequences across all eight requests, but native and streaming generated different sequences despite matching prompts, requested lengths and seeds. The cause of the timing variability is unproven. Six later streaming-only cohorts completed with 809.85–814.34 s generation and 4.53–4.62 s drain; this demonstrates sustained fast operation in that period, not reliability across conditions.

The published fused streaming recorder rounds BF16 intermediates differently from native vLLM. An experimental rounded variant matched native on one B200 operator test but still diverged in sampled tokens on two H200 full-model requests. Same-recorded-value gradient checks do not establish exact native-forward or full-gradient parity. A short non-aligned completion also triggered an unwarmed attention-graph error in the published branch. These issues remain unresolved.

## Measured comparison with truncation

These are historical means from three paired runs of the captured accepted implementation, before this readability/API refactor: Qwen3-1.7B, one B200, W=2048, eight simultaneous alternating 96k/128k completions, 917504 completion tokens. Source capture SHA256: `62c63232449cb1c8545ff7b06f13071788a635ad18f8717862097eaf8ca9f7db`.

| Method | Work after generation | Added training cost including generation slowdown | Generation + update |
| --- | ---: | ---: | ---: |
| Full FA4-PS | 31.65 s | 31.65 s | 837.52 s |
| Streaming full graph | 4.38 s | 9.14 s | 815.02 s |
| Truncated final window | 1.81 s | 1.81 s | 807.68 s estimated |

Native generation averaged 805.87 s and streaming generation 810.64 s. The added cost for streaming is therefore `(810.64 - 805.87) + 4.38 = 9.14 s`. The 3.46x improvement is `31.65 / 9.14`; complete-cycle wall time improves by approximately 2.7%. This is overlap/forward-reuse performance, not a claim that the FA4 attention kernel itself became 3.46x faster.

Compared with truncation, streaming has 5.05x the added training cost and 2.42x the post-generation work, while differentiating every completion token. The estimated complete cycle is 7.33 s, or 0.91%, longer. The truncated cycle adds independently measured native generation and truncated updates on same-shaped candidate traces; it was not a separately timed RL iteration. All update timings include the benchmark's reduction, clipping and discarded AdamW updates. Startup, warmup, reward evaluation and policy-weight broadcast are excluded. Peak allocated GPU memory was approximately 155 GB for streaming versus 169 GB for the fastest full baseline, with approximately 548 GB pinned host storage for streaming.

This establishes the captured fixed-cohort systems result, not maximum-throughput or multi-GPU RL performance. Refactor validation is reported separately in the accompanying handoff; historical timings are not relabeled as new measurements. No online RL run has been launched.

Refactor test results, actual long-trace numerical audits and source identities are in [ps_streaming_validation.md](ps_streaming_validation.md).
