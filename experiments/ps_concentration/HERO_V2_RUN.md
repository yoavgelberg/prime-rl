# Matching Pairs hero v2

The active experiment studies adaptation to bounded attention in native RNG-Bench textual Matching Pairs. It uses Qwen3-4B-Thinking-2507, a 4×4 board with eight pairs, and a limit of 96 attempted responses. The earlier POPGym/Instruct experiment and the stopped streaming-gradient run remain historical experiments; their settings and results must not be mixed with this run.

The starting model/tokenizer revision is `768f209d9ea81521153ed38c47d515654e938aea`. The native game revision is `e3d195e857727dfa3d4a96561be202098d67da83`. Native rules, observations, retries, and invalid-action consequences are preserved. Generated final answers use the native `Action: cD` format. The actual native score is pairs found. The reward used by training is defined below.

The system prompt contains the native game instructions and explicit answer-format instructions, asks for brief thoughts, and states the thinking budget. It supplies no notebook format, memory demonstration, external memory, or note-writing reward. Native thinking is part of the sampled assistant text. The controller samples at most 128 thinking tokens, inserts `\n\nOkay, time to give the answer.</think>\n\n` when the budget is exhausted, then permits at most 32 sampled final-answer tokens. Evaluation uses the identical controller. The inserted transition is 11 tokens with the pinned tokenizer. It is context with zero direct policy loss. It does not count against the 128 sampled thinking tokens.

## Attention and exact backward

Only the initial system message and its fixed delimiters are pinned. Its observed prefix length is 439 tokens. The first board observation belongs to the rolling suffix. For a query at original position `i` outside the prefix, allowed keys satisfy `j < P` or `max(P,i-W+1) <= j <= i`. Original position IDs increase monotonically and pretrained RoPE settings are unchanged.

Rollout retains streaming KV. It never repeatedly prefills a cropped prefix-plus-suffix transcript. Learners use each full recorded episode with the same PS mask, including losses before the first eviction. They retain backward dependencies through attention computations; no chunk-boundary detachment is used. Independent episodes occupy independent model requests and cannot attend across episode boundaries.

The current learner uses [NativeTeacher](native_teacher.py), effective BF16 forward values, FP32 masters/Adam state, FA4 PS attention, and activation checkpointing. The actor forward/backward computes policy loss, sampled reference KL, and full-vocabulary entropy together. One frozen-reference forward supplies reference log probabilities. The current v2 implementation does not reuse gradients computed during generation. That is an approved implementation choice, not a restriction imposed by KL.

Strict native-vLLM/teacher BF16 log-probability parity has not been established. The run retains actual native rollout log probabilities and logs teacher/rollout discrepancies and clipping. Do not describe this as passing the original strict numerical-parity gate or infer a measured speedup over another backend without comparable measurements. Exact attention/backward dependencies and floating-point agreement with native decoding are different properties.

## Reward and loss

For episode `k` on board `g`, let `m` be pairs found, `n` attempted responses, `f` format errors, and `v` native invalid actions. The two error counters are disjoint.

\[
R_{gk}=\frac{m_{gk}}8-\frac{n_{gk}}{96}-0.02f_{gk}-0.02v_{gk}.
\]

The metric historically called `avg_native_reward` is the unshaped efficiency reward `m/8-n/96`. It is not the game's native score. New continuation logs also expose `avg_unshaped_efficiency_reward`. `avg_reward` is the shaped reward above.

Groups contain eight independent sampled trajectories from the same reset board. GRPO uses sample standard deviation, without a leave-one-out baseline:

\[
\bar R_g=\frac18\sum_{k=1}^8 R_{gk},\quad
s_g=\sqrt{\frac17\sum_{k=1}^8(R_{gk}-\bar R_g)^2},\quad
A_{gk}=\frac{R_{gk}-\bar R_g}{s_g+10^{-4}}.
\]

Zero-variance groups have zero task advantage. KL and entropy remain active in those groups. Every sampled scratch/action/EOS token receives its trajectory advantage. Environment text, role delimiters, and forced tokens have zero direct loss but remain differentiable context.

For a sampled token, let `ell` be its current temperature-adjusted log probability, `ell_old` its stored native rollout log probability, and `ell_ref` the frozen reference log probability. Both policy/reference probabilities use temperature 0.6. Define

\[
q=\exp(\operatorname{clip}(\ell-\ell_{old},-20,20)),\qquad
d=\operatorname{clip}(\ell_{ref}-\ell,-20,20),\qquad
K_3=\exp(d)-1-d.
\]

With full-vocabulary entropy `H`, sampled-token mask `M`, global batch `B=128`, and the fixed inherited denominator `D=104*256=26624`, the minimized loss is

\[
\mathcal L=\frac1{BD}\sum_{g,k,t}M_{gkt}
\left[-\min\left(q_{gkt}A_{gk},\operatorname{clip}(q_{gkt},0.8,1.2)A_{gk}\right)
+0.01K_{3,gkt}-0.001H_{gkt}\right].
\]

The denominator is not the trajectory's actual token count. One fresh on-policy rollout batch produces one optimizer step, with one epoch and no stale-batch reuse. Exact transformer backward does not differentiate through sampled discrete tokens or the game engine. See [hero_objective.py](hero_objective.py) for the implementation and logged per-generated-token statistics.

## Optimizer and physical layout

AdamW uses learning rate `2e-6`, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay zero, and gradient norm clipping at 1. Warmup lasts eight updates. The parent then follows a cosine schedule to 20% of peak learning rate at update 128. Both continuations preserve that schedule horizon and clamp the terminal rate at `4e-7`; they do not restart warmup or rescale the cosine to 192.

The parent uses eight B200s: four independent vLLM actors and four replicated learners. There are 16 distinct boards × eight rollouts = 128 trajectories per update, with 32 live games per actor GPU. Each four-B200 continuation uses two actors and two learners, retaining global batch 128 and group size eight, with 64 live games per actor GPU. The two continuations use disjoint sets of four GPUs on an eight-GPU node, or run sequentially on a four-GPU fallback. Gradients are summed with the same global denominator. Throughput, memory, and allocated GPU-hours must come from saved measurements.

Completed board groups dispatch to available learners while other games continue generating. Policy weights remain frozen throughout that batch's collection and gradient accumulation. One gradient sum and optimizer step ends the batch. Actors that finish their whole assigned share wait at that policy barrier; the implementation does not collect the next policy's games before the update.

Use regular `yejin` only. Run/environment identities, complete dependency/revision locks, board permutations, rollout plans, generation RNGs, token/role/loss/position/log-probability records, per-update metrics, W&B URLs, full optimizer snapshots, and verified BF16 inference exports are saved by [hero_v2.py](hero_v2.py). Durable storage must cover both full continuation states and one temporary checkpoint write; checkpoint writes are serialized across branches. Policy exports use exact lossless archives with a shared full reference. No user-preserved checkpoint is silently discarded.

The parent writes full snapshots asynchronously every eight updates; continuation snapshots and inference candidates are scheduled every 16 updates. A stopping boundary also commits the exact model/optimizer/RNG state. Resume requires the latest committed update's full snapshot and never silently falls back to an older optimizer state. Complete matching episodes can be reused only after policy, seeds, controller, and token hashes agree. Native intermediate game writes may be buffered, but finalized compressed records retain the complete turn history.

## Approved continuation and evaluation protocol

Finish the seed-11 W1024 parent at 128 updates. Run a short greedy evaluation on the same first 32 validation boards at W1024/512/256/128/64, plus original/full and parent/full references. W256 for subsequent training is predeclared; the short sweep does not select it. Fork the complete parent model, Adam, step, and RNG state into matched W1024 and W256 continuations. Give each 64 additional updates through 192, using identical fresh board order and generation seed derivation. The reference remains the original starting checkpoint under each arm's mask.

Training samples a fixed permutation of board seeds 0–999999 without replacement. The seed is 11. Continuations use board indices 2048–3071 of that permutation, paired across arms. Validation uses seeds 1100000–1100127, final standard-board tests 2000000–2000511, larger boards 3000000–3000255, and intervention collection 4000000–4000511. Every new episode starts with an empty suffix and no previous board's text.

Both branches must finish 192 updates and shut down before final checkpoint selection/evaluation. Select among 128/144/160/176/192 by complete 128-board greedy validation efficiency return under that branch's training window, with earlier ties. Parent 128 remains an explicitly labeled baseline. If selected, it is not a newly trained continuation checkpoint. Final tests never choose a checkpoint or training window.

Evaluate the original model, parent 128, and both selected arms under PS256, PS1024, and full attention on all 512 final boards. The main continuation comparison is W256-trained versus W1024-trained, both evaluated under W256. The W1024 control was itself trained with prefix sliding; it is not a Full-RL control. The W256 arm is transfer from a W1024-trained parent, not training solely at W256 from the original model.

Evaluate both selected arms across W64/128/256/512/1024/2048/4096 on the first 128 final boards. Evaluate parameter-scaled 4×8 boards under PS256 on the 256 larger-board seeds; these have eight ranks with four copies each, 16 pairs, and 192 attempted responses. This is a parameter-scaled RNG-Bench task, not POPGym ConcentrationHard. The common transcript cap is 262144 tokens.

Report native pairs, efficiency return, completions, response counts with completed-only denominators, format/native invalids, generated and raw tokens, KV allocation scope, time, and total allocation cost. Score known-match utilization from complete native histories, binning the age of latest direct environment reveal; generated notes do not refresh that age. Bootstrap paired return differences 10000 times over board seeds. Both arms share one training seed and a parent; neither the 512 boards nor the two arms constitute independent training replications.

Collect separate greedy intervention games under W256 for both selected arms. Retained explicit old-target associations must satisfy the protocol's reveal-age, eviction, scratch-only, and equal-token-length requirements. Compare identical cold rebuilds with unedited/swap/sham scratch work, and compare unedited rebuilds with ordinary streaming continuation. Count every qualified and untestable state. Fewer than 32 qualified states is insufficient for a textual-memory conclusion. Never relabel sampled training traces as greedy to enter this collection.

Reports generate training-return, final-window, and reveal-age figures; main, size, and intervention tables; and 20 fixed-seed-order greedy transcripts per selected arm with eviction annotations. Good PS performance alone supports adaptation to bounded KV, not a claim that readable notes explain it.

## Execution entry points

Use immutable source captures and the resolved runtime environment. `CUDA_VISIBLE_DEVICES` must be the devices assigned by the regular-yejin allocation. The lease wrappers prepare dependencies, validate captures, and prevent duplicate campaign ownership; do not independently start a second coordinator for an active campaign.

```bash
python -m experiments.ps_concentration.hero_v2 --config RESOLVED_TRAIN_CONFIG.json
python -m experiments.ps_concentration.hero_pair --plan FROZEN_PAIR_PLAN.json \
  --gpus 8 --node-base UNIQUE_NODE_DIRECTORY --deadline ALLOCATION_END_EPOCH
python -m experiments.ps_concentration.hero_pair_final --plan FROZEN_PAIR_PLAN.json \
  --output POSTTRAIN_DIRECTORY --gpus 8
python -m experiments.ps_concentration.hero_pair_report --final-plan FINAL_PLAN.json \
  --memory MEMORY_DIRECTORY --output RESULTS_DIRECTORY
```

[hero_pair_plan.py](hero_pair_plan.py) freezes the approved matrix. [hero_continue.py](hero_continue.py) verifies state-preserving forks. [hero_pair.py](hero_pair.py) performs the short sweep and matched training. [hero_pair_final.py](hero_pair_final.py) validates, selects, evaluates, and intervenes only after both branches complete. [hero_pair_report.py](hero_pair_report.py) derives results and figures from saved records. These stages do not automatically declare goal completion or release the allocation.

Preserve complete node-local actor reports and verify inference exports with these CPU-only commands:

```bash
python -m experiments.ps_concentration.hero_telemetry --run-dir RUN_DIRECTORY
python -m experiments.ps_concentration.hero_exports --checkpoint RUN_DIRECTORY/checkpoints/latest.pt \
  --policy NODE_POLICY_DIRECTORY --output RUN_DIRECTORY/export-catalog/policy-VERSION.json
```

Telemetry checks exact episode coverage, policy versions, and token totals, then saves hashed durable reports. Its KV-pool allocation is shared capacity, not per-game logical retained KV. Export verification compares every effective BF16 parameter against the matching FP32 masters, handles tied weights, and verifies tokenizer/template/model/RoPE metadata. Run it before the corresponding latest snapshot is replaced. Exact policy archives preserve model bytes and metadata across allocations; hydration verifies their hashes before use.

The parent run's deployed artifacts are under `/hai/scratch/gelberg/ps/concentration/rng-hero-v2-a1/ps-seed11`; its W&B run is [rng-4b-ps-grpo-s11-hero-v2-a1](https://wandb.ai/tech-ml/ps-memory/runs/rng-4b-ps-grpo-s11-hero-v2-a1). Paired-plan artifacts are under `rng-hero-paired-a1` in the same project root. Source captures, launch receipts, and dependency locks record the actual implementation used by each stage. Training curves describe sampled training episodes; separate greedy results are required for held-out performance claims.
