# Native Matching Pairs PS-GRPO

Current allocation priority: run PS-GRPO immediately with `--defer-evaluation`. This explicit user override skips the full-context competence screening, PS update0 evaluation and periodic validation. Evaluation runs only afterward if allocation time remains. It preserves the training update, group and global batch sizes, seeds, optimizer, rewards and checkpoint saves. While deferred, there is no new same-run baseline, validation-selected checkpoint or pilot GO/HOLD result. `execution-options.json`, snapshot metadata and `pilot-complete.json` record that state.

The current pilot uses native RNG-Bench textual Matching Pairs 4×4 and Qwen3-4B-Thinking-2507. It runs PS-only GRPO through PrimeRL's persistent vLLM game runner and recorded backward during generation. The earlier POPGym CLI and standalone `rng_train.py` are separate historical implementations.

Launch training from a captured configuration in the pinned environment. Run evaluation separately after training:

```bash
python -m experiments.ps_concentration.rng_overlap train --config ps-config.json --defer-evaluation
python -m experiments.ps_concentration.rng_overlap evaluate --config full-config.json --attention full --split screening
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
```

Deferred runs retain the full optimizer snapshot without exporting validation checkpoints. After a completed update, export it on an allocated CPU or GPU node before evaluation:

```bash
python -m experiments.ps_concentration.export_checkpoint --checkpoint RUN_DIR/ps-seed11/checkpoints/latest.pt
```

A deferred run overwrites `latest.pt`; retain its committed 16/32-update snapshots before subsequent saves if they will be evaluated later. The current allocation uses a metadata-only hardlink watcher, verifying version, inode and file size without model evaluation or training interruption. Export from a retained snapshot after its commit.

This creates `policy-UPDATE` beside the snapshot, preserving the effective BF16 parameters, tokenizer, native generation stop tokens and training identity. It leaves the optimizer snapshot untouched, writes the export atomically, and records that export performed no validation selection. It refuses snapshots with no completed optimizer update and never replaces an existing export. Use `--reference` only for a relocated copy of the same recorded starting revision.

Configurations use `overlap_config.RunConfig`. Training provisions vLLM maximum requests, KV storage and forward CUDA graph sizes for `concurrency=4`, independently of evaluation concurrency. Provisioning the four-GPU training engine for 32 evaluation requests reserved an unnecessary 5.875 GiB of KV per GPU and contributed to a backward-tail OOM. The corrected training KV pool is 2 GiB/GPU, with forward graph sizes 1, 2 and 4. Full-context evaluation requires `evaluation_concurrency <= 4`; PS evaluation can use up to 64 games/GPU. The current launcher configures 16 inference-only PS games/GPU on eight GPUs or 32 on four GPUs. These evaluation settings do not reserve capacity in the training engine. Four recording games/GPU is the current measured layout, not a promise that larger batches fit.

Four or eight GPUs preserve the same 64-trajectory batch, eight board groups, generation RNG derivation, reward, optimizer and fixed denominator. Four GPUs use four 16-game cohorts; eight use two 32-game cohorts. Same-layout optimizer snapshots restore per-rank RNG; a layout change is recorded without claiming bitwise equality. `parallel_rank_collection=false` uses aligned cohorts because the pinned vLLM per-forward DP padding collective couples replicas. Independent rank collection remains an optional implementation and should not be enabled without workload qualification.

Without `--defer-evaluation`, the evaluation-first workflow evaluates the original checkpoint under full attention on the same 128 validation boards as PS update0. Training requires at least 80% matched cards; the CLI also accepts a complete 64-board full screening reference for compatibility. It saves the complete optimizer and FP32 masters after every update, then evaluates 128 validation boards at update 0, every 16 updates, and the final update. Selection uses highest validation efficiency return, preferring the earlier checkpoint on a tie. Interrupted future rollouts are quarantined; completed evaluation episodes are reused only after identity and token-log checksum checks. The pilot emits exact GO/HOLD values for full competence, initial forgetting gap, PS return improvement, and format errors; it does not automatically expand the matrix.

Reward is `pairs/8 - attempts/96`, with an explicit −1 resource-exhaustion reward. Native attempts are capped at 96 and transcript capacity at 262144 tokens. The current controller permits 128 sampled thinking-body tokens, then inserts `\n\nOkay, time to give the answer.</think>\n\n` if thinking has not ended naturally. Its 11 tokens have role `thinking_budget`, zero direct loss and no sampled likelihood. They count toward rolling attention and original positions but not the 128 sampled-thinking budget. Final answer text remains sampled until the native action-line or EOS stop; there is no forced action prefix or final-answer cap. Training and evaluation must use this same controller and system prompt.

GRPO uses eight rollouts on each of eight boards, sample standard deviation (`ddof=1`) plus 1e−4, fixed token denominator 26624, and one optimizer step per fresh 64-trajectory batch. Sampling is temperature .6 with unfiltered softmax, top-p 1 and no top-k cutoff. Every sampled thinking/action token contributes, including losses before the first eviction; inserted environment, template and budget tokens have zero direct loss. Native streaming backward supplies the task gradients during generation. AdamW uses learning rate 5e−6, betas (.9,.95), epsilon 1e−8, weight decay 0, eight-update linear warmup and gradient clipping 1.0. Seed 11 is the 32-update pilot; 256 is a total main-run cap, including any continued pilot updates.

**The active run sets `kl_coefficient=0`.** It loads no frozen reference and performs no second policy forward/backward pass. Each trajectory's captured native negative log-likelihood is checked against its actual sampled rollout likelihood, with maximum allowed mean sum error 1e−6 per sampled token. Disabled reference KL is reported as DISABLED, not as measured zero distribution KL or numerical parity GO. An independent BF16 NativeTeacher implementation disagreed with native rollout likelihoods on some tokens; that discrepancy remains unresolved, and its auxiliary pass is excluded from the native-only training path. Intervention likelihood validity is assessed separately before any causal textual-memory conclusion. Earlier sampled inference benchmarks used different sampling/controller settings and are historical controls only.

W&B uses the repository's standard monitor, entity `tech-ml`, project `ps-memory`, and a stable run ID across resumes. Mean episodic reward is `train/mean_return`; `train/avg_reward` is an identical alias in the current code. Metrics are logged after a complete, committed 64-trajectory update; a partial collection never becomes a completed-update point. Canonical JSON metrics, complete native game files, compressed token records, source/dependency locks, exact seed plans, and checkpoint metadata remain on scratch. Report compute time separately from Slurm allocation cost, including idle and failed time. The initial pilot has 32 total updates; this command does not automatically expand the main training matrix.

Evaluation saves per-cohort receipts across resumes, including elapsed time and each device's physical KV-pool allocation. Interrupted cohorts without a complete receipt make the summed inference cost a lower bound; allocation accounting remains separate. Physical KV-pool bytes include block padding and available capacity, whereas the logical attention mask retains at most P+W keys.

Final held-out evaluation and the prespecified first128-board window sweep use separate output configurations per condition:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-ps-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-w512-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test --window 512 --count 128
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-larger-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split larger
python -m experiments.ps_concentration.rng_report --manifest result-manifest.json --output REPORT_DIR --tokenizer ORIGINAL_SNAPSHOT --examples-condition ps-seed11-w1024
```

The report manifest has `conditions`, `comparisons`, and optional `training_runs` lists. A condition contains `name`, `policy` (`base` or `PS RL`), `training_seed` (null for base), `split`, `attention`, `window`, `greedy:true`, `expected_boards`, and `episodes` (the exact episode JSONL path). A comparison names `treatment`, `sliding_control`, and `full_control` conditions. Training runs identify `training_seed` and `run_dir`. Each condition reports pairs/8, efficiency return, completed-only attempt count/denominator, attempts with incomplete games counted at96, formats, invalid moves, censored runs, tokens and per-game throughput. Reports reject incomplete splits unless explicitly labeled partial. Paired intervals resample board seeds 10,000 times within each training seed; independent training seeds determine training variability. A frozen-base control is explicitly labeled and never called Full RL. The plotter produces validation return versus training episodes, test return versus window, and known-match utilization versus direct-reveal age from saved records. Twenty example traces use fixed board-seed order, retaining all thinking and eviction annotations.

The return-versus-training-episodes figure also supports deferred evaluation: it plots sampled training returns from complete 64-trajectory updates in `update-metrics`, and greedy validation returns in a separate panel when available. Both are labeled explicitly and exported to `training-return-curve.csv`. Only versions at or below the committed `state.json` policy version are included; collecting half a batch never becomes a training point. A sampled training curve does not establish held-out improvement or select a checkpoint.

Optional condition `summary` paths attach cohort inference costs and measured physical KV-pool bytes, with board-count/attention identity checks. The manifest can include `interventions:[{name,training_seed,summary}]` to produce the intervention table and `allocation_accounting` plus an explicit `accounting_scope` to attach the full allocation-cost ledger. Reports distinguish accepted optimizer-update compute, inference cohort time, and whole allocation time. Requeued allocations and failed/setup/idle/cleanup time belong in allocation cost; pending reservations consume zero GPU-hours.

Capture costs on HAIC using explicit experiment-owned allocation IDs:

```bash
python -m experiments.ps_concentration.allocation_accounting --jobs JOB_ID JOB_ID --output ACCOUNTING_DIR --reservations RESERVATIONS.json
```

The accounting module uses only the standard library. It captures UTC `sacct --duplicates` rows, retains every requeue interval, extends parent intervals for batch/extern cleanup, and never sums steps as separate allocations. Missing restart history is reported as incomplete. B200/H200 hours remain separate; nominal GPU-hour sums do not normalize hardware speed.

Text-memory interventions run only after a trained checkpoint is selected:

```bash
python -m experiments.ps_concentration.rng_overlap evaluate --config intervention-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split intervention
python -m experiments.ps_concentration.rng_intervention_states --episodes INTERVENTION_EPISODES.jsonl --tokenizer ORIGINAL_SNAPSHOT --output STATES_DIR
python -m experiments.ps_concentration.rng_intervene --checkpoint SELECTED_HF_CHECKPOINT --states STATES_DIR/states.jsonl.gz --reference ORIGINAL_SNAPSHOT --source PINNED_NATIVE_SOURCE --output INTERVENTION_DIR
```

Collection uses only board seeds4000000–4000511 and stops at128 qualifying states. The cursor is immediately before the first coordinate token of a pending second-pick action. A direct reveal must be older than2W and absent from the retained suffix; sampled scratch must explicitly associate its unmatched partner with the target rank. Conservative coordinate/rank patterns can be supplemented with manual exact token-span annotations via `--annotations`. Notes do not reset reveal age. Untestable qualified states remain in the reported denominator, with the reason recorded.

Every branch independently rebuilds native-effective BF16 model KV from only the prefix and suffix, using original position IDs and exact FA4 PS attention. Edits change sampled scratch only; false-association and unrelated-sham edits preserve both token count and number of changed tokens. The runner reconstructs and clones the native game state, records one greedy next flip and32 stochastic continuations per branch, and audits full-history recomputed likelihood against the actual native streaming choice. Probabilities are Monte Carlo estimates at temperature .6, conditional on the fixed cursor after the existing greedy thinking text and Action prefix; they do not marginalize over freshly generated thinking. All qualified states and the subset whose unedited greedy action selects the partner are reported separately. Intervention intervals resample boards with every collected state in each sampled board; a one-board collection has no between-board interval. A textual-memory conclusion requires at least32 fully executed states, complete qualified-state coverage, strict likelihood parity, and a false-association effect beyond sham edits. CPU checks establish the cache/replay/edit invariants, not a GPU or learned-memory result.

Reports verify the exact reserved board seed set and group training-seed statistics by board-set hash as well as policy, attention, window and split. The512-board final test and first128-board window sweep therefore stay separate even atW1024. `main-comparison.csv`, `window-sweep.csv` and `size-generalization.csv` identify each scope; window curves connect only identical board sets and label their board counts. `training-seed-summary.csv` reports per-seed means and sample standard deviations for task, cost and reveal-age utilization metrics, with the number of defined training runs for every metric. Empty utilization bins and undefined completed-only attempt counts remain undefined. One seed has no training-variability estimate.

The `larger` split is evaluation-only: 256 held-out seeds3000000–3000255 on a4×8 board with32 cards, eight distinct ranks and four cards of every rank, with192 attempted responses. It retains the selected PS window, the native262144-token transcript ceiling, the same128-token thinking budget and11-token forced transition, and the identical system prompt; board dimensions change only the native observations. This is a parameter-scaled RNG-Bench task, and additional partners can make some decisions easier. The effective board parameters are saved in `evaluation-protocol.json` and episode records. Use a `larger` report condition with `expected_boards:256`; `size-generalization.csv` reports pairs/16 separately from the4×4 main table. Resource censors remain explicit rather than becoming unobserved192-attempt failures. Training remains4×4 with96 attempts and fixed gradient denominator26624.

Manual intervention annotations can map an exact sampled-token `surface` (for example `(1,1)`) to a native `position` (for example `aA`) and `rank`. Swaps reuse the corresponding retained surface alias or an explicitly supplied `replacement_surfaces` mapping. Both swap and sham still require equal token lengths and changed-token counts; recorded observations and past action lines remain immutable. Annotation changes only the offline intervention, never the prompt or training reward.

The completed historical sampled baseline has its own CPU-only scorer:

```bash
python -m experiments.ps_concentration.baseline_memory_report --run-dir HISTORICAL_PAIRED_ROOT --tokenizer ORIGINAL_SNAPSHOT --output BASELINE_DIAGNOSTICS --workers 8
```

It verifies all64 paired boards and fixed system-prefix identities, reads only completed native responses, and reports pooled known-match utilization with a paired board bootstrap. The policies visit different decision states; these diagnostics are not a shared-state causal comparison. Historical sampling (temperature .6/top-p .95/top-k20) and custom StreamingQwen arithmetic remain explicit, and are kept separate from the native-vLLM greedy qualification and trained-policy results.


The first two updates saved each rank's materialized task gradient under `pending-task-gradient/` before their original auxiliary KL pass. These are recovery artifacts, not optimizer checkpoints. Never apply a pending batch whose policy version is stale. A historical recovery used the command below after its original driver stopped, under the exclusive campaign lock and on the original replica count:

```bash
python -m experiments.ps_concentration.resume_pending --config RUN_CONFIG.json \
  --pending RUN_DIR/pending-task-gradient/update-0002.json \
  --disable-reference-kl --continue-training
```

The explicit flag permits only the recorded reference-KL identity transition, preserves the model/optimizer and native task gradients, commits once, and continues training in the same engine. Current ordinary resumes restore `checkpoints/latest.pt` automatically; do not replay already-committed pending gradients. If a later uncommitted batch has no saved gradients, its partial logs are quarantined and the same frozen board/RNG plan is resampled from the last committed policy. The optimizer snapshot is authoritative.

The pinned PyTorch ZeRO checkpoint gather previously constructed `torch.ByteTensor(bytearray)` from multi-gigabyte shards. The scoped buffer-based equivalent `torch.frombuffer(BytesIO.getbuffer(),dtype=uint8)` transfers the same serialized payload without element-by-element byte conversion. Tensor contents, rank order, NCCL collectives and checkpoint format are unchanged. The override is restored after the gather. CPU round-trip tests and actual multi-gigabyte checkpoint saves validate this transport.

The pinned PyTorch ZeRO `load_state_dict` forces scalar optimizer state onto CPU, including fused AdamW step counters. Snapshot restore moves only fused/capturable step tensors back to each parameter's device after loading, preserving their values and all optimizer settings. Nonfused, noncapturable CPU step state stays on CPU. This fixes resume of an existing fused optimizer; the initial fresh step did not encounter it.
