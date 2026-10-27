# Refactor validation — 2026-10-26

The cleaned implementation passed the checks below, but the subsequent twelve-pair B200 retest did **not** reproduce a reliable speedup: streaming averaged 63.11 s added training cost and 871.11 s generation plus update, versus 35.12 s and 843.98 s for full FA4-PS. One streaming pair overlapped a GPU diagnostic for approximately 12 seconds; its effect is unknown. Results, workload and limitations are detailed in [the walkthrough](ps_streaming.md#completed-retest-of-the-published-implementation). The earlier 3.46x added-cost result belongs to the historical captured source, not to the completed retest.

## Checks completed

- 115 CPU/GPU tests passed on one B200: exact FP64 full-model streaming comparisons, FA4/dense attention and Q/K/V gradients, Blackwell FP32 adjoints, fused pointwise/recording kernels, repeated ring wraparound, delayed-reward algebra and the baseline objectives.
- The independently applied FA4-only patch passed 28 CPU tests; 9 GPU cases were excluded from that separate CPU check.
- The final meta-model initialization test passed, including tied weights and reconstruction of deterministic RoPE buffers. Recorded backward uses the actual inference RoPE cache.
- Real vLLM AsyncLLM generation, warmup, CUDA graph capture, request completion and full same-generation-value audits passed. No optimizer update or online RL was run.
- All measured streaming calls in these audits had zero eager fallback stages and zero new FA4 compilations. Every loss token was covered; all owned activation records were reclaimed.

| Completion tokens | Relative gradient L2 error | Gradient cosine |
| ---: | ---: | ---: |
| 12,288 | 0.00617678 | 0.999980924 |
| 16,384 | 0.00676097 | 0.999977218 |
| 131,072 | 0.00778005 | 0.999970253 |
| 6,144 | 0.00571620 | 0.999983662 |

The numerical gate was relative L2 < 0.01 and cosine >= 0.9999. The reference differentiates the full graph using the same recorded forward values, with unfused pointwise backward. These audit runs intentionally retain an additional full tape in test support; they do not measure bounded-memory performance. The separate recomputed-HF-forward comparison remains unresolved, as documented in [the walkthrough](ps_streaming.md).

## Additional numerical and integration limits

The streaming recorder and native vLLM use different BF16 rounding for RMSNorm and SwiGLU. Native and streaming sampled different tokens in the long retest even though prompts, seeds and requested lengths matched; their contribution to this divergence is not isolated. A candidate that matches native on one B200 operator test still differed from native tokens in H200 full-model tests. A separately recomputed HF-style forward/backward with the same weights differed from streaming by 0.316/0.255 relative L2 on 3072/6144-token B200 traces, normalized by the streaming gradient. These are not proof of exact native-forward or end-to-end gradient parity. A short non-aligned completion also triggered an unwarmed attention-graph error; a local fallback was not published or long-trace qualified.

## Source identities and scope

Prime-RL base: `5a9434a8fc49e7ac3500988c02ec4e0b43dce869`.

The 115-test source archive SHA256 is `3adf8704ba43f5bbe026ab847c0b41edb7f9c782c4bc4d67c1e492f09cf26da6`. The short/131072-token engine audit source is `f911674eab9876e68e9b8ea74668ed54e0b3725a6e2b7f844035b2a4976db055`; its runtime matches that unit-test source. The final runtime archive SHA256 is `6d53b4d15f8c009c36f5d1c991201d937695b21afb8c33d968a5ca21374b1eb9`; it additionally initializes HF RoPE buffers after meta allocation and passed the 6144-token engine audit. All runtime files in the delivered diff match that final successful smoke source. Only documentation and the test-only audit memory-report labeling changed afterward.

All runs used the pinned stack described in [the baseline documentation](fa4_ps.md), Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, and a single B200 per process. The short two-request case generated 12k and 16k completions; the long and final-source checks each used one request. Maximum allocation reservations were one hour for the main checks and 15 minutes for the final-source smoke.

Two launcher failures are preserved in the evidence: a shell quoting error before tests, and conflicting vLLM backend settings after all 115 tests passed. Both were fixed before the successful engine runs. Total charged allocation for this cleanup and validation was 0.548056 GPU-hours, including failures and startup.

The handoff contains `01-fa4-ps-baseline.patch`, `02-streaming-backward.patch`, and their combined diff, all against the stated base. Strict `git apply --check --whitespace=error` and byte-for-byte reconstruction verify the patch pair. The accompanying manifest records file and patch hashes.
