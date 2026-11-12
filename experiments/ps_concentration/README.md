# Matching Pairs memory RL

This experiment tests whether outcome RL teaches a language model with bounded attention to preserve useful board state in its generated thinking. The completed experiment uses **native RNG-Bench textual Matching Pairs**, Qwen3-4B-Thinking-2507, and full-parameter PS-GRPO. The directory retains its historical `ps_concentration` name. The original 52-card POPGym/Instruct implementation is documented separately in [POPGYM.md](POPGYM.md).

## Game and model

A 4×4 board has 16 initially hidden cards, eight distinct values, and two copies of each value. Each assistant response selects one card using `Action: cD`. Two successive valid flips form a pair; matching cards are removed, and mismatches are shown before becoming hidden again. A game ends when all eight pairs are found or after 96 attempted responses. Native observations and invalid-action handling are retained. One response is one card attempt, not a complete two-card round.

The pinned model and tokenizer revision is `768f209d9ea81521153ed38c47d515654e938aea` of `Qwen/Qwen3-4B-Thinking-2507`. The pinned [RNG-Bench](https://github.com/InternLM/RNGBench) revision is `e3d195e857727dfa3d4a96561be202098d67da83`. The system prompt specifies the game, answer format, and thinking budget. It supplies no memory strategy, notebook format, memory examples, or reward for writing notes.

Only the initial system message and fixed template delimiters are pinned (439 tokens with this tokenizer). At absolute query position `i`, attention can use causal prefix keys `j < P` and rolling keys `max(P, i-W+1) <= j <= i`, with `W=1024`. Position IDs keep increasing. Actors retain streaming KV between turns. They never rebuild a cropped transcript. Learners process the full episode with the same attention mask and all backward dependencies.

## Training recipe

| Setting | Parent run |
| --- | --- |
| Batch | 16 boards × 8 independent trajectories = 128 games/update |
| Training seed | 11; fixed permutation of seeds 0–999999 |
| Sampling | Temperature 0.6, top-p 1, no top-k cutoff |
| Per response | 128 sampled thinking tokens; up to 32 sampled answer tokens |
| Forced transition at thinking cap | `\n\nOkay, time to give the answer.</think>\n\n` |
| Optimizer | AdamW, LR 2e-6, betas (0.9, 0.95), epsilon 1e-8, weight decay 0 |
| Schedule | Eight-update warmup; cosine to 20% of peak at update 128 |
| Gradient clipping | Global norm 1 |
| GRPO | Within-board mean and sample standard deviation; epsilon 1e-4 |
| Policy ratio clipping | [0.8, 1.2], using stored native actor log probabilities |
| Regularization | Reference KL coefficient 0.01; entropy coefficient 0.001 |
| Loss normalization | Fixed denominator 26,624 per trajectory, averaged over 128 |
| Update rule | One fresh batch, one optimizer step; no rollout reuse across updates |

The reward is `pairs/8 - responses/96 - 0.02*format_errors - 0.02*invalid_actions`. Format and native invalid-action counters are disjoint. The historical metric `avg_native_reward` means `pairs/8 - responses/96`; the native game's actual score is pairs found. Every sampled assistant token, including early thinking and sampled EOS, receives the episode advantage. Inserted transitions, environment observations, and role delimiters have zero direct loss while remaining differentiable context.

The exact loss and original protocol are in [HERO_V2_RUN.md](HERO_V2_RUN.md). The full W1024 lineage comprises updates 1–128 (parent), 129–192 (continuation), 193–320 (extension), and 321–1600 (long continuation). The extension warms LR from 4e-7 to 1e-6 over eight updates and then keeps it constant. Exact categorical `KL(policy || reference)` replaces sampled K3 at update 215. Temperature rises to 0.8 after update 320. The reference remains the original starting checkpoint. These are changes within one evolving seed, not independent training runs. [Extension protocol](HERO_EXTENSION_RUN.md), [post-320 protocol](HERO_POST320_RUN.md).

## Runtime and setup

The final training path uses independent native vLLM actors and joint FA4-PS learner passes for policy loss, KL, and entropy, plus a frozen-reference forward. It does **not** reuse generation-time streaming gradients. The earlier streaming-gradient implementation remains available under [OVERLAP_RUN.md](OVERLAP_RUN.md).

Eight B200s use four actors and four replicated learners, with 32 concurrent games per actor. Four B200s use two actors and two learners, with 64 games per actor. Both layouts retain batch 128, group size eight, the same loss denominator, and gradient accumulation before a single synchronized step. Finished board groups train while other groups are still generating; weights stay fixed until the batch barrier.

Use Python 3.12 and the project's PS-enabled vLLM and FA4 runtime. [locks/hero-extension-runtime-packages.lock.txt](locks/hero-extension-runtime-packages.lock.txt) records observed package versions, and [locks/hero-post320-runtime-revisions.json](locks/hero-post320-runtime-revisions.json) records model/game/runtime identities. These captures document custom providers as well as installed packages; installing stock wheels from the version list alone does not reconstruct the deployed vLLM overlay. Use the existing qualified runtime for GPU execution. CPU checks can use ordinary CPU PyTorch.

The commands below assume the runtime is active and this repository is on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_ENTITY=YOUR_ENTITY WANDB_PROJECT=ps-memory
```

Download the pinned Hugging Face snapshot into a directory whose final component is its revision hash. Prepare the native source from a fresh checkout; only its game files are included in the source manifest:

```bash
git clone https://github.com/InternLM/RNGBench.git /path/to/RNGBench
git -C /path/to/RNGBench checkout e3d195e857727dfa3d4a96561be202098d67da83
uv run --active --no-sync python - /path/to/RNGBench <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert revision == "e3d195e857727dfa3d4a96561be202098d67da83"
subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "1_matching_pairs_new"], check=True)
paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "1_matching_pairs_new"], text=True).splitlines()
assert paths
manifest = [{"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in paths]
(root / "SOURCE.json").write_text(json.dumps({"commit": revision, "manifest": manifest}, indent=2) + "\n")
PY
```

Capture the **experiment implementation** independently with `git archive --format=tar.gz HEAD > /path/to/implementation.tar.gz`, and compute its SHA256 with `sha256sum`. The `--source` argument below identifies the native game; `--source-sha256` identifies that implementation archive. Run the captured clean implementation for corresponding provenance.

## Launch and resume

Prepare the original 128-update W1024 parent recipe without GPUs:

```bash
uv run --active --no-sync python -m experiments.ps_concentration.launch \
  --model /path/to/768f209d9ea81521153ed38c47d515654e938aea \
  --source /path/to/RNGBench --source-sha256 IMPLEMENTATION_ARCHIVE_SHA256 \
  --output /path/to/runs/ps-memory --node-dir /node/local/ps-memory/control \
  --deadline ALLOCATION_END_UNIX_SECONDS --gpus 8 --wandb-id UNIQUE_RUN_ID \
  --prepare-only
```

This writes `launch-config.json`, verifies the native file hashes, and preserves an existing configuration if it matches. Remove `--prepare-only` to launch inside an assigned regular `yejin` allocation. Use `--gpus 4` for the same ML recipe on four B200s. `CUDA_VISIBLE_DEVICES` must expose exactly the assigned devices. A new process group is created for each worker, and termination uses the same cleanup path as the direct training CLI.

The underlying CLI remains available for recorded continuation configurations:

```bash
uv run --active --no-sync python -m experiments.ps_concentration.hero_v2 --config /path/to/launch-config.json
```

Resume with the saved configuration, a fresh node-local control directory, and the new allocation deadline/GPU count. Keep the same run directory and W&B ID. The runner restores `checkpoints/latest.pt` (model, Adam, and all-rank RNG) and verifies the saved identity. Source changes require explicit reviewed migration digests; it never silently resumes from an earlier optimizer state. Continuation/fork helpers preserve the same checks. Checkpoint frequency is configured with `checkpoint_interval` and `export_interval`; the last W1024 continuation used 64-update boundaries and saves at a requested final stop as well.

Runtime output includes complete compressed token/role/loss/position/log-probability records, board permutations and generation seeds, update metrics, W&B metrics, and checkpoint metadata. Keep outputs in `outputs/`, `runs/`, or another external directory; they are ignored by Git. Credentials belong in the runtime environment, never in configuration or source control.

## Code map and validation

| Module | Purpose |
| --- | --- |
| `hero_config.py`, `launch.py` | Shared recipe and portable parent configuration/launch |
| `hero_v2.py` | Synchronous actor/learner coordination, checkpointing, W&B |
| `hero_objective.py`, `native_teacher.py` | Masked GRPO, KL, entropy, full-episode backward |
| `rng_vllm.py`, `hero_inference.py` | Native game adapter and actor weight updates |
| `hero_continue.py`, `hero_extension.py` | State-preserving continuations |
| `hero_memory.py`, `rng_memory_metrics.py` | Opportunity-normalized recall and note diagnostics |
| `rng_intervene.py`, `rng_intervention_states.py` | Controlled edits and matched cold-cache branches |
| `hero_budget_final.py`, `hero_post320_report.py` | Frozen final evaluations and saved-metric reports |

Run CPU regression tests without allocating GPUs:

```bash
export PS_CONCENTRATION_NATIVE_SOURCE=/path/to/RNGBench
uv run --active --no-sync python -m pytest experiments/ps_concentration -q
```

The native-source variable enables the nine logging checks against the pinned game implementation. Archive checks also require the `zstd` executable.

GPU numerical and performance audits require the qualified runtime and separate resources. Exact PS dependency structure does not establish floating-point identity between vLLM and the teacher: strict actor/learner log-probability parity remains unresolved. CPU regression tests do not resolve that limitation or establish GPU performance.
