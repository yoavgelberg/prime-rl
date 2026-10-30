The current pilot uses native RNG-Bench textual Matching Pairs 4×4 and Qwen3-4B-Thinking-2507. It runs PS-only GRPO through PrimeRL's persistent vLLM game runner and recorded backward during generation. The earlier POPGym CLI and standalone `rng_train.py` are separate historical implementations.

Launch a captured configuration with:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config full-config.json --attention full --split screening
python -m experiments.ps_concentration.rng_overlap train --config ps-config.json --competence FULL_SCREENING_SUMMARY.json
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
```

Configurations use `overlap_config.RunConfig`. Full-context evaluation requires `evaluation_concurrency <= 4`; training targets four recording requests per GPU. The launcher uses 16 inference-only PS validation games per GPU on eight GPUs, or 32 on four GPUs, admitting all 128 validation boards together. The earlier 4B inference benchmark successfully used 60 PS games on one B200; this informs inference concurrency but does not establish native-vLLM performance or backward memory fit. Pure evaluation configurations can use up to 64 games per GPU. Physical GPU count can be four or eight without changing the 64-trajectory training batch, board groups, generation seeds, reward, or denominator. Same-layout optimizer snapshots restore per-rank RNG; changing the physical layout is recorded and does not promise bitwise identity.

The launcher evaluates the original checkpoint under full attention on the same 128 validation boards as PS update0. Training requires at least 80% matched cards; the CLI also accepts a complete 64-board full screening reference for compatibility. It saves the complete optimizer and FP32 masters after every update, then evaluates 128 validation boards at update 0, every 16 updates, and the final update. Selection uses highest validation efficiency return, preferring the earlier checkpoint on a tie. Interrupted future rollouts are quarantined; completed evaluation episodes are reused only after identity and token-log checksum checks. The pilot emits exact GO/HOLD values for full competence, initial forgetting gap, PS return improvement, and format errors; it does not automatically expand the matrix.

Reward is `pairs/8 - attempts/96`, with an explicit -1 resource-exhaustion reward. Native thinking has no per-turn token limit; the transcript ceiling remains 262144 tokens and native attempts are capped at 96. GRPO uses eight rollouts of each of eight boards, sample-standard-deviation normalization plus 1e-4, fixed token denominator 26624, one fresh optimizer step, and frozen-reference k3 KL .01. All sampled assistant tokens contribute; inserted environment/template tokens have zero direct loss. Sampling is temperature .6 with unfiltered softmax; earlier sampled benchmark results used top-p .95/top-k20 and are reported separately.

The first actual learning batch supplies graph capture and likelihood auditing. Strict likelihood targets remain mean .01/p99 .05; an explicitly logged exploratory miss up to mean .02/p99 .15 is allowed, and larger errors stop before the optimizer step. No additional GPU infrastructure or performance smoke is launched. Four requests per GPU is a host-memory planning ceiling, not a measured 4B throughput claim.

Native likelihood recomputation uses `native_teacher.NativeTeacher`: BF16 effective embeddings, projections, norm weights, RoPE cache, and stored residuals, with FP32 sums before fused normalization. FP32 masters retain straight-through derivatives through effective BF16 rounding. An independent CPU implementation with separate native residual streams matches exactly across multiple window evictions, including non-BF16 master values. GPU forward parity and long-trajectory memory remain first-learning-batch checks; the CPU result does not establish GPU parity.

W&B uses the repository's standard monitor, entity `tech-ml`, project `ps-memory`, and a stable run ID across resumes. Canonical JSON metrics, complete native game files, compressed token records, source/dependency locks, exact seed plans, and checkpoint metadata remain on scratch. Report compute time separately from Slurm allocation cost, including idle and failed time. The initial pilot has 32 total updates; this command does not automatically expand the main training matrix.

Evaluation saves per-cohort receipts across resumes, including elapsed time and each device's physical KV-pool allocation. Interrupted cohorts without a complete receipt make the summed inference cost a lower bound; allocation accounting remains separate. Physical KV-pool bytes include block padding and available capacity, whereas the logical attention mask retains at most P+W keys.

Final held-out evaluation and the prespecified first128-board window sweep use separate output configurations per condition:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-ps-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-w512-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test --window 512 --count 128
python -m experiments.ps_concentration.rng_report --manifest result-manifest.json --output REPORT_DIR --tokenizer ORIGINAL_SNAPSHOT --examples-condition ps-seed11-w1024
```

The report manifest has `conditions`, `comparisons`, and optional `training_runs` lists. A condition contains `name`, `policy` (`base` or `PS RL`), `training_seed` (null for base), `split`, `attention`, `window`, `greedy:true`, `expected_boards`, and `episodes` (the exact episode JSONL path). A comparison names `treatment`, `sliding_control`, and `full_control` conditions. Training runs identify `training_seed` and `run_dir`. Each condition reports pairs/8, efficiency return, completed-only attempt count/denominator, attempts with incomplete games counted at96, formats, invalid moves, censored runs, tokens and per-game throughput. Reports reject incomplete splits unless explicitly labeled partial. Paired intervals resample board seeds 10,000 times within each training seed; independent training seeds determine training variability. A frozen-base control is explicitly labeled and never called Full RL. The plotter produces validation return versus training episodes, test return versus window, and known-match utilization versus direct-reveal age from saved records. Twenty example traces use fixed board-seed order, retaining all thinking and eviction annotations.

Text-memory interventions run only after a trained checkpoint is selected:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config intervention-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split intervention
python -m experiments.ps_concentration.rng_intervention_states --episodes INTERVENTION_EPISODES.jsonl --tokenizer ORIGINAL_SNAPSHOT --output STATES_DIR
python -m experiments.ps_concentration.rng_intervene --checkpoint SELECTED_HF_CHECKPOINT --states STATES_DIR/states.jsonl.gz --reference ORIGINAL_SNAPSHOT --source PINNED_NATIVE_SOURCE --output INTERVENTION_DIR
```

Collection uses only board seeds4000000–4000511 and stops at128 qualifying states. The cursor is immediately before the first coordinate token of a pending second-pick action. A direct reveal must be older than2W and absent from the retained suffix; sampled scratch must explicitly associate its unmatched partner with the target rank. Conservative coordinate/rank patterns can be supplemented with manual exact token-span annotations via `--annotations`. Notes do not reset reveal age. Untestable qualified states remain in the reported denominator, with the reason recorded.

Every branch independently rebuilds native-effective BF16 model KV from only the prefix and suffix, using original position IDs and exact FA4 PS attention. Edits change sampled scratch only; false-association and unrelated-sham edits preserve both token count and number of changed tokens. The runner reconstructs and clones the native game state, records one greedy next flip and32 stochastic continuations per branch, and audits full-history recomputed likelihood against the actual native streaming choice. Probabilities are Monte Carlo estimates at the training sampling temperature .6. All qualified states and the subset whose unedited greedy action selects the partner are reported separately. A textual-memory conclusion requires at least32 fully executed states, complete qualified-state coverage, strict likelihood parity, and a false-association effect beyond sham edits. CPU checks establish the cache/replay/edit invariants, not a GPU or learned-memory result.
