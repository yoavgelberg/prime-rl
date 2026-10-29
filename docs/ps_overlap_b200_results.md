# Complete B200 comparison

All 12 prespecified rounds completed on three B200 GPUs sharing one host. Each method uses one GPU. Qwen3-1.7B, window 2048, eight real vLLM AsyncLLM traces per round: alternating 98,304 and 131,072 completion tokens, 917,504 completion losses per cohort. All rows and outliers are included.

| Method | Generation | Added training cost | Work after generation | Complete measured cycle |
|---|---:|---:|---:|---:|
| Full FA4-PS update | 801.59 | 34.89 | 34.89 | 836.48 |
| Published streaming | 841.17 | 47.89 | 8.31 | 849.48 |
| Improved streaming | 774.72 | 11.14 | 6.32 | 781.04 |
| Approximate final-window update | 801.59 | 1.79 | 1.79 | 803.39 |

Times are means in seconds. Added streaming cost is max(0, streaming generation − matched native generation) + max(worker drain, drain RPC) + gradient reduction + clipping/discarded AdamW update. Faster generation receives no credit in this metric. Complete cycles use each method's actual generation time. Full FA4 includes forward recomputation, complete backward and the update; each round selects the faster measured full-gradient precision control. Startup, warmup, reward computation, weight broadcast and trace serialization are excluded. Final-window training includes fewer losses and a different objective.

The ratio of mean added costs is **3.132× versus full FA4-PS** and **4.300× versus published streaming**. The measured complete cycle is **6.63% shorter** than native generation plus full FA4. Raw generation differences contribute to the cycle comparison and are reported below.

The prespecified one-sided 95% bootstrap lower bound on added-cost speedup versus full FA4 is **2.0665×**. The greater-than-2× performance gate is **PASSED**. This supporting estimate resamples three GPU shards and circular blocks of three rounds within each shard (10,000 draws, seed 20261027). Three GPUs on one host do not establish reliability across independent hosts.

| Round | Full FA4 added | Published streaming added | Improved streaming added | Final-window added | Full / improved |
|---|---:|---:|---:|---:|---:|
| 0 | 35.14 | 56.28 | 7.01 | 1.79 | 5.02× |
| 1 | 35.14 | 23.41 | 5.32 | 1.79 | 6.60× |
| 2 | 35.14 | 44.14 | 5.37 | 1.79 | 6.55× |
| 3 | 35.02 | 38.70 | 5.19 | 1.80 | 6.74× |
| 4 | 34.79 | 113.91 | 20.65 | 1.79 | 1.69× |
| 5 | 34.76 | 27.26 | 20.24 | 1.79 | 1.72× |
| 6 | 34.76 | 9.30 | 5.32 | 1.79 | 6.54× |
| 7 | 34.78 | 21.70 | 7.17 | 1.79 | 4.85× |
| 8 | 34.81 | 51.50 | 41.33 | 1.79 | 0.84× |
| 9 | 34.74 | 96.03 | 5.35 | 1.79 | 6.49× |
| 10 | 34.84 | 43.98 | 5.36 | 1.79 | 6.50× |
| 11 | 34.69 | 48.44 | 5.33 | 1.79 | 6.51× |

| Round | Native generation | Published generation | Improved generation | Improved generation change | Improved work after generation |
|---|---:|---:|---:|---:|---:|
| 0 | 817.03 | 865.94 | 805.88 | -11.15 | 7.01 |
| 1 | 798.43 | 817.11 | 746.36 | -52.06 | 5.32 |
| 2 | 798.48 | 837.11 | 749.34 | -49.14 | 5.37 |
| 3 | 793.97 | 828.01 | 743.47 | -50.50 | 5.19 |
| 4 | 805.63 | 890.13 | 816.88 | +11.25 | 9.39 |
| 5 | 789.15 | 811.68 | 801.81 | +12.66 | 7.58 |
| 6 | 809.61 | 814.23 | 772.14 | -37.47 | 5.32 |
| 7 | 807.72 | 824.79 | 763.58 | -44.14 | 7.17 |
| 8 | 812.68 | 855.75 | 846.59 | +33.91 | 7.42 |
| 9 | 790.38 | 873.91 | 750.52 | -39.86 | 5.35 |
| 10 | 793.78 | 833.23 | 749.55 | -44.24 | 5.36 |
| 11 | 802.25 | 842.15 | 750.49 | -51.76 | 5.33 |

| Round | Full FA4 cycle | Published cycle | Improved cycle | Final-window cycle | Published work after generation |
|---|---:|---:|---:|---:|---:|
| 0 | 852.17 | 873.31 | 812.89 | 818.83 | 7.37 |
| 1 | 833.57 | 821.83 | 751.69 | 800.22 | 4.72 |
| 2 | 833.62 | 842.62 | 754.70 | 800.28 | 5.52 |
| 3 | 829.00 | 832.68 | 748.66 | 795.77 | 4.67 |
| 4 | 840.42 | 919.54 | 826.28 | 807.42 | 29.41 |
| 5 | 823.91 | 816.42 | 809.39 | 790.95 | 4.73 |
| 6 | 844.37 | 818.92 | 777.46 | 811.40 | 4.69 |
| 7 | 842.50 | 829.42 | 770.75 | 809.51 | 4.63 |
| 8 | 847.48 | 864.18 | 854.01 | 814.47 | 8.43 |
| 9 | 825.12 | 886.42 | 755.87 | 792.17 | 12.50 |
| 10 | 828.62 | 837.76 | 754.90 | 795.57 | 4.53 |
| 11 | 836.95 | 850.69 | 755.82 | 804.04 | 8.53 |

The frozen candidate runtime is `9e8edea1bc35c2dd5d9662274763b4e9bac07ecb`; capture SHA256 `cc91c4b228fd21a4e3f8a5ce95e17d24200cbe9bc20aee963ba27fac4e251114` (645 verified files). Jobs 151842/151843/151844 ran rounds 0–3/4–7/8–11, using rotated native/published/improved orders. The later gradient-scope, memory and coalesced-field patches are excluded from this comparison.

The implementation retains FA4 attention and optimizes the surrounding streaming backward: joint QKV projection/input-normalization VJPs, direct FP32 gradient accumulation, compact KV reuse, per-layer readiness, GPU head staging, bounded transfers and GPU work credits. It preserves complete PS dependencies.

All eight complete published/improved sampled sequences match in every round. Numerical qualification passed the 131,072-token same-recorded-value gradient audit at relative L2 0.00644481. Independent native-forward numerical parity remains unresolved; these audits do not establish it. The workload uses a unit-advantage mean of per-trajectory score gradients. Optimizer updates are discarded systems measurements; no online RL has run. [Full-precision JSON rows](data/ps_overlap_b200_a5.json) and the [matching analysis](data/ps_overlap_b200_a5_analysis.json) are included. Use [the configuration and example](ps_overlap_improvements.md) to select this implementation.
