# Original POPGym Concentration experiment

This is the earlier 52-card POPGym/Instruct protocol. It is distinct from the completed RNG-Bench/Thinking GRPO experiment described in [README.md](README.md).

This experiment tests outcome-based adaptation to a fixed system prefix and bounded rolling attention. The neutral prompt permits scratch work but supplies no note format, examples, memory tools, or note-writing reward. Native observations, rewards, action-0 format fallback and transient mismatch observations are preserved.

Run from the repository root with `PYTHONPATH=src:.` and the captured environment active. The commands are also runnable through `uv run --active --no-sync` in that environment.

```bash
python -m experiments.ps_concentration.audit
python -m experiments.ps_concentration.baselines --split screening
python -m experiments.ps_concentration.train --attention full --seed 11 --updates 32
python -m experiments.ps_concentration.train --attention ps --seed 11 --updates 32
python -m experiments.ps_concentration.evaluate --checkpoint CHECKPOINT --split test
python -m experiments.ps_concentration.intervene --checkpoint CHECKPOINT
python -m experiments.ps_concentration.report --run-dir RUN_DIR
```

The default public model name is only a convenience. Actual runs must pass the exact snapshot path recorded in `locks/revisions.json`. The model and tokenizer revision are `cdbee75f17c01a7cc42f958dc650907174af0554`. No thinking tokens or RoPE changes are used. The system prefix is 152 tokens for the neutral 52-card prompt. The conservative UTF-8 byte bound raises the common 52-card transcript cap from 65,536 to 73,203; the larger-board bound fits 262,144. See the saved tokenizer audit for the arithmetic and any later stricter bound.

`locks/` contains the complete observed CPU and GPU package versions. The runtime was staged from a byte-verified portable environment and two pure-Python wheels (`popgym==1.0.7`, `gymnasium==1.0.0`). Captures and manifests live under the workspace's `artifacts/ps-concentration/`; remote source captures and run receipts live under `/hai/scratch/gelberg/ps/concentration/`. Local CPU calibration and GPU results are separate receipts. The native environment source SHA256 must match the specification before comparisons.

Rollout uses a persistent KV ring with original increasing positions. It never recomputes only the retained context between game turns. The controller's complete history is used only for logs and offline diagnostics. Every call begins with fresh episode caches. Episodes are never packed together; independent batch lanes have independent KV. Only system-template prefix tokens are pinned. Role delimiters and environment tokens have zero direct loss.

Training reuses the repository's Qwen adapter, exact FA4 prefix/window mask and checkpointed full-episode graph. It includes every sampled scratch/action token, including early scratch before the first eviction; no cache detachment is used. Sequences are padded only after their final sampled token to a 1,024-token allocation boundary, with no loss on padding and no effect on preceding causal queries. This reduces shape-dependent compilation. Linear tensor-core operations use BF16 operands, with FP32 projection outputs, residuals, master weights, gradient accumulation and Adam state. The arithmetic is shared by both arms and reference policies and must pass the saved streaming/training parity audit; Policy checkpoints retain the exact effective forward values: BF16 projection operands and FP32 embeddings, normalization weights and biases. Full unrounded FP32 masters and Adam state remain in the resume checkpoint. A save/reload equivalence check verifies the effective policy.

On eight 180-GB B200 GPUs, launch eight synchronous processes that each use their GPU for rollout and then training:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m experiments.ps_concentration.train --attention ps --seed 11 --updates 32 \
  --model EXACT_SNAPSHOT --window FROZEN_W --output RUN_DIR
```

Each of eight processes receives one board and eight trajectories, then accumulates eight complete-episode gradients. Four-GPU pilots receive two boards per process. FP32 master gradients are summed across workers before clipping and one AdamW update. Colocating rollout and training lets all allocated GPUs work in both synchronous phases. This is replicated data parallelism with activation checkpointing: the larger B200 VRAM permits avoiding FSDP communication. The global trajectory count and objective are unchanged. Use `--separate-actors` with four processes to select the original four-trainer/four-actor layout. A single-GPU smoke uses the same 64-trajectory global update. Throughput and memory must be measured, not inferred from this layout.

The loss is `sum(-A*logp + 0.01*k3)/(26624*64)`, with leave-one-out episodic-return advantages and no standard-deviation normalization. Here `k3=exp(logp_ref-logp_policy)-(logp_ref-logp_policy)-1`, the same algebra as the established trainer's nonnegative sampled mismatch estimator, applied to the frozen same-attention reference. Log probabilities use temperature-adjusted logits before top-p truncation, following the existing rollout/trainer convention; sampling uses temperature 0.8 and top-p 0.95. This convention is recorded explicitly rather than claiming an exact likelihood for the truncated sampling distribution. Report both normalized KL and KL per generated token.

Checkpoint selection uses validation return under training attention, with earlier ties retained. `latest_optimizer.pt` includes FP32 masters, Adam state, update index, best checkpoint and RNG state. Resume with `--resume RUN_DIR/latest_optimizer.pt`; `--updates 256` means 256 total updates. Board permutations and generation RNG derivation are fixed independently of attention. Main runs use seeds 11, 22 and 33.

The user authorized skipping Full RL if the neutral original full-context model already meets the 80% competence threshold. In that case the comparator is the untrained original under PS, and the report must state that rule/format learning is not matched. The user also authorized exploratory continuation for marginal performance-gate misses; record the original GO/HOLD values and any exploratory designation. Major correctness problems are not performance gate exceptions.

Interventions recognize explicit position-rank scratch associations conservatively and log the exact spans. They edit only assistant scratch token IDs, preserve length and pinned prefix, compare three identical cold rebuild procedures, and separately clone the ordinary streaming cache. All qualified states, including states for which a sham cannot be constructed, are counted. There is no textual-memory conclusion with fewer than 32 tested qualified states. Irregular note formats require manual review; the automatic recognizer is not a universal memory detector.

Reports are generated from saved records and include CSV/JSON tables, PNG figures and an HTML artifact. Bootstrap intervals resample boards within each training seed; variation across training seeds is reported separately. Provisional numerical probes and incomplete screening runs are not final test results.

CUDA graphs replay persistent-cache decoder operations; both PS and full-cache boundary checks reproduce eager logits and cache values exactly. Full decode uses the installed FA4 kernel with a dynamic causal-length mask, avoiding masked-SDPA fallback. Enable with `PS_CONCENTRATION_GRAPHS=1`.

Initial fresh long-trajectory parity measured mean absolute log-probability error .00610 and p99 .06437: the mean passes and p99 misses the .05 target. The system smoke measured .00675/.06527. These target misses remain explicit HOLD receipts. The user permits exploration for marginal gates, so `--allow-numerical-target-miss` permits pilots only within documented investigation bounds (mean <= .02, p99 <= .15); it does not relabel the original target as passed. No major correctness, nonfinite-gradient, immutable-prefix or cache-boundary failure is waived.

HAIC scratch had 604 GB free at setup, so selected and latest policy binaries are retained with metadata for every scheduled checkpoint; unselected older binaries are retired only after a complete replacement and optimizer receipt exist. Full training token logs and resume state are preserved. Retired checkpoint directories record this explicitly. Bulk artifacts stay on HAIC or local `/tmp`; the workspace home had less than 1 GB free.

The H200 fallback uses six full-attention rollout lanes per GPU (eight for PS); B200 uses eight. Both retain global batch64 and the same episode/optimizer rules. H200 pilots can hand off at a complete optimizer update to the eight-B200 main allocation. An allocation lease keeps GPUs across worker failures, and a shared lock prevents two controllers from updating the same run concurrently.

An optional queued-token path (`PS_CONCENTRATION_PREFILL_TOKENS=32`) ingests known transcript tokens in blocks through the existing exact FA4 mask while preserving the streaming cache. It is disabled by default. `python -m experiments.ps_concentration.prefill_check --model EXACT_SNAPSHOT --output RECEIPT` checks PS eviction, immutable prefix, full-cache growth and cached timings; fresh rollout/training parity is still required before enabling it. It never skips sampled losses or rebuilds from a cropped history.
