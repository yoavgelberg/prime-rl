# Native Matching Pairs PS-GRPO

Current allocation priority: run PS-GRPO immediately with `--defer-evaluation`. This explicit user override skips the full-context competence screening, PS update0 evaluation and periodic validation. Evaluation runs only afterward if allocation time remains. It preserves the training update, group and global batch sizes, seeds, optimizer, rewards and checkpoint saves. While deferred, there is no new same-run baseline, validation-selected checkpoint or pilot GO/HOLD result. `execution-options.json`, snapshot metadata and `pilot-complete.json` record that state.

The current run uses native RNG-Bench textual Matching Pairs 4×4 and Qwen3-4B-Thinking-2507. It runs PS-only GRPO through PrimeRL's persistent vLLM game runner and recorded backward during generation. The user extended seed 11 to **128 total updates / 8192 trajectories**, including its initial pilot updates. Set `updates=128` in `RunConfig`; the generic default remains 32. Continuations restore the model, optimizer and RNG state and keep the original update numbers and board permutation. The earlier POPGym CLI and standalone `rng_train.py` are separate historical implementations.

Launch training from a captured configuration in the pinned environment. Run evaluation separately after training:

```bash
python -m experiments.ps_concentration.rng_overlap train --config ps-config.json --defer-evaluation
python -m experiments.ps_concentration.rng_overlap evaluate --config full-config.json --attention full --split screening
python -m experiments.ps_concentration.rng_overlap evaluate --config eval-config.json --checkpoint SELECTED_HF_CHECKPOINT --attention ps --split test
```

The deferred trainer retains the full optimizer snapshot. Export a completed snapshot on an allocated CPU or GPU node before evaluation:

```bash
python -m experiments.ps_concentration.export_checkpoint --checkpoint RUN_DIR/ps-seed11/checkpoints/latest.pt
```

A deferred run overwrites `latest.pt`; retain each checkpoint needed for later validation before subsequent saves. This campaign retains full optimizer snapshots at updates 16/32 and uses a separate CPU watcher to export effective BF16 policies every 16 updates through 128. The watcher hardlinks a committed snapshot, verifies its version, inode and file size, then uses the exporter and checks all output file hashes. It removes its temporary link after successful export. The full optimizer snapshots and current `latest.pt` remain available. This adds no model evaluation or GPU work. Export paths can be set explicitly:

```bash
python -m experiments.ps_concentration.export_checkpoint --checkpoint RETAINED_SNAPSHOT.pt --output RUN_DIR/ps-seed11/policies/policy-0016
```

This creates `policy-UPDATE` beside the snapshot, preserving the effective BF16 parameters, tokenizer, native generation stop tokens and training identity. It leaves the optimizer snapshot untouched, writes the export atomically, and records that export performed no validation selection. It refuses snapshots with no completed optimizer update and never replaces an existing export. Use `--reference` only for a relocated copy of the same recorded starting revision.

Configurations use `overlap_config.RunConfig`. Training provisions vLLM maximum requests, KV storage and forward CUDA graph sizes for `concurrency=4`, independently of evaluation concurrency. Provisioning the four-GPU training engine for 32 evaluation requests reserved an unnecessary 5.875 GiB of KV per GPU and contributed to a backward-tail OOM. The corrected training KV pool is 2 GiB/GPU, with forward graph sizes 1, 2 and 4. Full-context evaluation requires `evaluation_concurrency <= 4`; PS evaluation can use up to 64 games/GPU. The current launcher configures 16 inference-only PS games/GPU on eight GPUs or 32 on four GPUs. These evaluation settings do not reserve capacity in the training engine. Four recording games/GPU is the current measured layout, not a promise that larger batches fit.

Tail backward operations process at most 8192 tokens together. The earlier 16384-token grouping exhausted a B200 during update 8 when FA4 needed another 260 MiB. Smaller groups preserve the attention mask, all token losses and complete backward dependencies; they change gradient accumulation order, so bitwise gradient equality is not claimed. Training still admits four recording games per GPU.

After each completed cohort has been reduced into immutable CPU board-group buffers, the trainer keeps at most eight recently used attention CUDA graphs and discards older shape-specific attention graphs. Fixed dense-stage graphs remain reusable. This bounds GPU memory retained by varying attention shapes across updates. The previous cache grew from roughly55 to140stages perGPU during updates8–13; update14 subsequently ran out of memory during an FP32 tail conversion. Eviction occurs only after streaming backward is drained; it changes reuse and capture cost, not attention, losses or derivatives. The8192-token tail grouping and four live training games perGPU remain unchanged. Per-cohort `graph_cache_retention` records the retained/evicted counts.

Four or eight GPUs preserve the same 64-trajectory batch, eight board groups, generation RNG derivation, reward, optimizer and fixed denominator. Four GPUs use four 16-game cohorts; eight use two 32-game cohorts. Same-layout optimizer snapshots restore per-rank RNG; a layout change is recorded without claiming bitwise equality. `parallel_rank_collection=false` uses aligned cohorts because the pinned vLLM per-forward DP padding collective couples replicas. Independent rank collection remains an optional implementation and should not be enabled without workload qualification.

Without `--defer-evaluation`, the evaluation-first workflow evaluates the original checkpoint under full attention on the same 128 validation boards as PS update0. Training requires at least 80% matched cards; the CLI also accepts a complete 64-board full screening reference for compatibility. It saves the complete optimizer and FP32 masters after every update, then evaluates 128 validation boards at update 0, every 16 updates, and the final update. Selection uses highest validation efficiency return, preferring the earlier checkpoint on a tie. Interrupted future rollouts are quarantined; completed evaluation episodes are reused only after identity and token-log checksum checks. The pilot emits exact GO/HOLD values for full competence, initial forgetting gap, PS return improvement, and format errors; it does not automatically expand the matrix.

Reward is `pairs/8 - attempts/96`, with an explicit −1 resource-exhaustion reward. Native attempts are capped at 96 and transcript capacity at 262144 tokens. The current controller permits 128 sampled thinking-body tokens, then inserts `\n\nOkay, time to give the answer.</think>\n\n` if thinking has not ended naturally. Its 11 tokens have role `thinking_budget`, zero direct loss and no sampled likelihood. They count toward rolling attention and original positions but not the 128 sampled-thinking budget. Final answer text remains sampled until the native action-line or EOS stop; there is no forced action prefix or final-answer cap. Training and evaluation must use this same controller and system prompt.

GRPO uses eight rollouts on each of eight boards, sample standard deviation (`ddof=1`) plus 1e−4, fixed token denominator 26624, and one optimizer step per fresh 64-trajectory batch. Sampling is temperature .6 with unfiltered softmax, top-p 1 and no top-k cutoff. Every sampled thinking/action token contributes, including losses before the first eviction; inserted environment, template and budget tokens have zero direct loss. Native streaming backward supplies the task gradients during generation. AdamW uses learning rate 5e−6, betas (.9,.95), epsilon 1e−8, weight decay 0, eight-update linear warmup and gradient clipping 1.0. The active seed-11 budget is 128 total updates; extending the budget does not reset warmup or add another pilot.

**The active run sets `kl_coefficient=0`.** It loads no frozen reference and performs no second policy forward/backward pass. Each trajectory's captured native negative log-likelihood is checked against its actual sampled rollout likelihood, with maximum allowed mean sum error 1e−6 per sampled token. Disabled reference KL is reported as DISABLED, not as measured zero distribution KL or numerical parity GO. An independent BF16 NativeTeacher implementation disagreed with native rollout likelihoods on some tokens; that discrepancy remains unresolved, and its auxiliary pass is excluded from the native-only training path. Intervention likelihood validity is assessed separately before any causal textual-memory conclusion. Earlier sampled inference benchmarks used different sampling/controller settings and are historical controls only.

W&B uses the repository's standard monitor, entity `tech-ml`, project `ps-memory`, and a stable run ID across resumes. Mean episodic reward is `train/mean_return`; `train/avg_reward` is an identical alias. `train/mean_attempts` averages action responses across all 64 games, including invalid attempts and unfinished games at the 96-attempt cap. One action response flips one card. Metrics are logged after a complete, committed update; a partial collection never becomes a completed-update point. Canonical JSON metrics, complete native game files, compressed token records, source/dependency locks, exact seed plans, and checkpoint metadata remain on scratch. Report compute time separately from Slurm allocation cost, including idle and failed time. Matrix expansion remains explicit; the accepted 128-update budget applies to the current seed-11 run.

For the 4×4 board with exactly two copies of each of eight ranks, optimal play with perfect recall takes an expected `1116484/45045 ≈ 24.78597` individual flips under uniform shuffling. This is twice the expected 12.39298 two-flip moves in Table 2 of [The Solitaire Memory Game](https://ktfoerster.github.io/paper/2013-memory.pdf); its Lemma 4.1 proves the known-pair-first exploration strategy optimal. The corresponding expected efficiency reward is about 0.741813. This is an offline reference, with no memory demonstration or strategy added to model inputs. The result does not establish optimality for the rank-repeated larger board or the original POPGym task.

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

After the deferred training barrier and all prespecified greedy validation conditions finish, select a checkpoint with:

```bash
python -m experiments.ps_concentration.rng_select --plan POSTTRAINING_PLAN.json --run-dir TRAINING_RUN_DIR --gpus 4
```

Use `--gpus 8` for that plan's eight-GPU layout. This CPU-only command checks the complete128-board validation episode records against their summaries, planned configurations, controller identities and checkpoint metadata. It requires every planned PS candidate, including update0, and the matching original full-context control. It selects the highest PS validation efficiency return, retaining the earlier update on ties, and writes `selected-checkpoint.json` with input hashes and competence/forgetting GO or HOLD values. It cannot run before the training barrier or select from final-test data. Run the evaluations separately; this command performs no model inference or training.

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

Collection uses only board seeds4000000–4000511 and stops at128 qualifying states. The primary cursor is before any sampled thinking in a pending second-pick response, after its native observation and assistant template. A direct reveal must be older than2W and absent from the retained suffix; earlier sampled scratch must explicitly associate its unmatched partner with the current target rank. Current-turn thinking cannot qualify as preserved memory. Qualification does not discard completed source turns with malformed actions. An unfinished source response leaves qualifying-state coverage unknown and must be resolved before a primary causal collection. Conservative coordinate/rank patterns can be supplemented with exact manual token-span annotations via `--annotations`. Notes do not reset reveal age. Untestable qualified states remain in the reported denominator, with the reason recorded.

Each unedited, swap and sham branch independently rebuilds native-effective BF16 KV from only the same pinned prefix and retained suffix, using original position IDs and exact FA4 PS attention. Each branch then generates new thinking and the next action through the same `PersistentChat` controller as native rollout, including its128-token sampled-thinking budget and forced transition. The source record must match the exact checkpoint, board, PS window and thinking controller. Edits change earlier sampled scratch only; false-association and unrelated-sham edits preserve token count and number of changed tokens. The runner reconstructs and independently clones the native game, records one greedy next flip and32 stochastic continuations per branch, and audits all sampled likelihoods in the recorded source turn against native streaming likelihoods. This separate intervention audit adds no model pass to training.

Probabilities are Monte Carlo estimates at temperature.6, including newly sampled thinking and excluding the separate greedy continuation. All qualified states and the subset whose unedited greedy action selects the partner are reported separately. Comparison with the ordinary streaming action starts before thinking too, so it can reveal cached hidden-state dependence during reasoning. Intervention intervals resample boards retaining every collected state in a sampled board; one board has no between-board interval. A textual-memory conclusion requires at least32 fully executed states, complete qualified-state coverage, strict likelihood parity and a false-association effect beyond sham edits. CPU checks cover the cache/edit/controller invariants; GPU likelihood validity and learned-memory effects remain unmeasured.

The optional `rng_intervention_states --boundary first_coordinate` retains the earlier action-cursor diagnostic. It conditions on already generated current-turn thinking and cannot establish that earlier text carried memory across turns. Such results can report local causal text use at the action cursor, while `causal_textual_memory_established` remains false. Do not combine the two boundaries in a primary intervention result.

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
