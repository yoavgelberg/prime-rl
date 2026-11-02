# Fixed-window continuation and fresh evaluation

This experiment uses native textual RNG-Bench Matching Pairs 4×4, with eight pairs and a 96-response limit. One response flips one coordinate. It is separate from the original POPGym Concentration implementation.

Both policies fork the same complete W=1024 update-192 checkpoint, preserving model weights, Adam state, and RNG state. They continue for 128 updates under W=1024 or W=512, ending at update 320. This is an exploratory extension after inspecting earlier tests. It uses one shared training seed, 11, rather than independent seed replications.

The model is Qwen3-4B-Thinking-2507, revision `768f209d9ea81521153ed38c47d515654e938aea`. The native game revision is `e3d195e857727dfa3d4a96561be202098d67da83`. Native thinking is limited to 128 sampled tokens; the controller then inserts `\n\nOkay, time to give the answer.</think>\n\n` and permits 32 sampled answer tokens. Forced text, environment messages, and template delimiters have zero direct loss. All sampled assistant tokens contribute loss. There are no memory demonstrations, prescribed notebooks, or note-writing rewards.

Each fresh update has 16 boards and eight independent trajectories per board: global batch 128, group size 8. Both arms use the same ordered board seeds and generation RNG derivation. Sampling uses temperature 0.6, top-p 1, and no top-k cutoff. Reward is pairs/8 minus responses/96, with another 0.02 subtracted per format error or native invalid action. Native score is the number of pairs; efficiency reward and error-penalized training reward are reported separately.

GRPO advantages subtract the group mean and divide by sample standard deviation plus 1e-4. Zero-variance groups have zero task advantage. The token objective combines the PPO-clipped task surrogate (ratio clip 0.2), reference KL coefficient 0.01, and entropy coefficient 0.001. It sums over sampled tokens, divides by the fixed denominator 26,624, and averages over 128 trajectories. It does not normalize by each game's generated length.

Both arms use sampled K3 KL through update 214 and exact categorical KL from update 215. Exact KL sums `p * (log p - log q)` over the vocabulary at each sampled assistant position, using temperature-adjusted actor and frozen-reference distributions under that arm's attention mask. It reuses the actor/reference transformer passes and reference head products; it retains reference log probabilities for the checkpointed actor head. Exact KL does not resolve the measured native-rollout/learner log-probability discrepancy. That limitation and the clipping diagnostics remain in the saved records.

Learning rate warms linearly from 4e-7 to 1e-6 over updates 193–200, then stays at 1e-6. AdamW uses betas 0.9/0.95, epsilon 1e-8, no weight decay, gradient clipping at 1, and one optimizer step per fresh batch. Forward weights are BF16 effective values; master weights, gradient accumulation, and Adam state are FP32. The learner uses the joint FA4 actor forward/backward plus a frozen reference forward. This version does not reuse generation-time gradients.

Eight GPUs initially split into two four-GPU arms. Each arm has two native vLLM actors with 64 concurrent games each and two learners. After one arm finishes, an operational handoff can give the remaining arm all eight GPUs while preserving global batch, group size, optimizer, and update budget. The saved per-rank RNG mapping and reduction-order change are recorded. Regular `yejin` is required. Full snapshots use HAIC rolling saves every five updates; explicit handoff/final snapshots preserve transitions.

## Fresh evaluation

`hero_extension_final` freezes a final-320 comparison without checkpoint selection. Standard test boards use seeds 5,000,000–5,000,511; parameter-scaled 4×8 boards use 6,000,000–6,000,255. The primary comparison is W512-trained versus W1024-trained, both evaluated under PS512. The common parent192 under PS512 is a secondary baseline. Full-attention evaluations, original-Qwen evaluations, and a window sweep provide references and diagnostics.

The sweep uses W in {256,512,1024,2048,4096} on the same first 128 fresh test boards. The W512/W1024 points derive from the matching complete 512-board conditions, avoiding duplicate inference. Larger-board evaluation retains PS512 and uses the native parameter-scaled board with a 192-response cap. All evaluation is greedy and uses the same thinking/answer controller.

PS evaluation permits 64 concurrent games per GPU, matching the existing native actor capacity. At W4096 its KV reservation bound is 42.75 GiB per GPU, excluding weights and workspaces. Full attention retains the separately audited 16-game concurrency and 53,568-token KV bound. Physical four/eight-GPU layouts preserve policy conditions and board identities.

```bash
python -m experiments.ps_concentration.hero_extension_final prepare \
  --plan TRAIN_PLAN --output FINAL_DIR --gpus 8 \
  --source-root CAPTURED_SOURCE --source-sha256 ARCHIVE_SHA

# Fails before GPU evaluation unless both final320 states are complete.
python -m experiments.ps_concentration.hero_extension_final finish \
  --plan FINAL_DIR/plan.json --gpus 8

# Reporting is CPU-only. Partial reports explicitly list missing results.
python -m experiments.ps_concentration.hero_extension_report \
  --final-plan FINAL_DIR/plan.json --output REPORT_DIR --allow-partial
```

Run from the repository root with the captured environment and `PYTHONPATH=src:.`. The `run` subcommand performs evaluation only; `memory` resumes the intervention stage separately. `finish` performs evaluation, interventions, and the complete report. Runtime checks verify source archives, policy identities, both training barriers, and idle assigned GPUs before inference. The supervisor can queue this stage after training; it must not divert training GPUs early.

Intervention collection uses the separate 4,000,000–4,000,511 seeds. Both final policies collect under PS512; the W1024 policy additionally collects under PS1024. Up to 128 states qualify before the next thinking turn. The existing intervention runner compares unedited, false-association, and sham branches using equal-length scratch-only edits and identical cold rebuilds at original positions. It reports streaming versus cold continuation separately, including numerical parity, every qualified/untestable state, and Monte Carlo next-flip probabilities from 32 samples per branch. Fewer than 32 tested qualified states is insufficient for a readable-memory conclusion.

Reports contain main comparison, size-generalization, window-sweep, intervention, and cache-control CSV tables; JSON metrics and provenance; PNG/PDF figures for training return, return versus W, and known-match utilization versus direct-reveal age; and the first 20 fixed-seed-order greedy transcripts per trained arm under PS512. Figures derive from saved records. Paired intervals use 10,000 bootstrap resamples over boards; training-seed standard deviation remains undefined for this single shared lineage. Good PS performance alone establishes adaptation to bounded KV, not a causal readable notebook.
