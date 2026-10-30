The current pilot uses native RNG-Bench textual Matching Pairs 4×4 and Qwen3-4B-Thinking-2507. It runs PS-only GRPO through PrimeRL's persistent vLLM game runner and recorded backward during generation. The earlier POPGym CLI and standalone `rng_train.py` are separate historical implementations.

Launch a captured configuration with:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config full-config.json --attention full --split screening
python -m experiments.ps_concentration.rng_overlap train --config ps-config.json --competence FULL_SCREENING_SUMMARY.json
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
```

Configurations use `overlap_config.RunConfig`. Full-context evaluation requires `evaluation_concurrency <= 4`; training targets four recording requests per GPU, with eight inference-only PS validation requests. Physical GPU count can be four or eight without changing the 64-trajectory batch, board groups, generation seeds, reward, or denominator. Same-layout optimizer snapshots restore per-rank RNG; changing the physical layout is recorded and does not promise bitwise identity.

The training command requires 64 complete greedy full-context screening episodes with at least 80% matched cards. It saves the complete optimizer and FP32 masters after every update, then evaluates 128 validation boards at update 0, every 16 updates, and the final update. Selection uses highest validation efficiency return, preferring the earlier checkpoint on a tie. Interrupted future rollouts are quarantined; completed evaluation episodes are reused only after identity and token-log checksum checks.

Reward is `pairs/8 - attempts/96`, with an explicit -1 resource-exhaustion reward. Native thinking has no per-turn token limit; the transcript ceiling remains 262144 tokens and native attempts are capped at 96. GRPO uses eight rollouts of each of eight boards, sample-standard-deviation normalization plus 1e-4, fixed token denominator 26624, one fresh optimizer step, and frozen-reference k3 KL .01. All sampled assistant tokens contribute; inserted environment/template tokens have zero direct loss. Sampling is temperature .6 with unfiltered softmax; earlier sampled benchmark results used top-p .95/top-k20 and are reported separately.

The first actual learning batch supplies graph capture and likelihood auditing. Strict likelihood targets remain mean .01/p99 .05; an explicitly logged exploratory miss up to mean .02/p99 .15 is allowed, and larger errors stop before the optimizer step. No additional GPU infrastructure or performance smoke is launched. Four requests per GPU is a host-memory planning ceiling, not a measured 4B throughput claim.

W&B uses the repository's standard monitor, entity `tech-ml`, project `ps-memory`, and a stable run ID across resumes. Canonical JSON metrics, complete native game files, compressed token records, source/dependency locks, exact seed plans, and checkpoint metadata remain on scratch. Report compute time separately from Slurm allocation cost, including idle and failed time. The initial pilot has 32 total updates; this command does not automatically expand the main training matrix.
