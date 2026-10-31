# Matching Pairs hero v2

This run trains Qwen3-4B-Thinking-2507 from its original checkpoint on native RNG-Bench textual Matching Pairs4×4, the task authorized after earlier model/environment screening. It is separate from the original POPGym ConcentrationHard experiment. The earlier PS-GRPO checkpoint35 and its optimizer/RNG/logs remain preserved.

Inside an existing regular yejin allocation with the assigned CUDA_VISIBLE_DEVICES:

```bash
python -m experiments.ps_concentration.hero_v2 --config CONFIG.json
```

CONFIG.json uses HeroConfig in hero_v2.py. The deployed launcher,captured sources,dependency locks and source manifests are retained at /hai/scratch/gelberg/ps/concentration. Output: rng-hero-v2-a1/ps-seed11. W&B: tech-ml/ps-memory/rng-4b-ps-grpo-s11-hero-v2-a1. Model revision:768f209d9ea81521153ed38c47d515654e938aea. Native environment revision:e3d195e857727dfa3d4a96561be202098d67da83.

Eight GPUs use four independent single-GPU vLLM actors,each admitting32games,and four replicated FA4-PS learners. Four GPUs use two actors admitting64games each and two learners. Both layouts preserve128trajectories/update,16boards×8rollouts,seed derivation and optimizer settings. Completed groups dispatch to learner capacity while generation continues. Weights stay frozen throughout collection/accumulation; one gradient sum and optimizer step ends each batch. Actors that finish their entire share wait for that final policy barrier.

Reward is pairs/8 minus attempted responses/96 minus0.02per format error minus0.02per native invalid action. The two error counters are disjoint. Advantages use the board-group mean and sample standard deviation plus1e-4; identical-return groups get zero task advantage. The loss combines tokenwise clipped GRPO (ratio clip0.2),sampled k3 KL coefficient0.01 and full-distribution entropy bonus0.001. All three terms sum sampled tokens and divide by128×26624; actual trajectory length is not the denominator. Stored native rollout likelihoods define the importance ratio. Log ratios are bounded to±20 for stability.

Each trajectory has a frozen-reference forward and a joint actor forward/backward with layer checkpointing and a checkpointed output head. This path does not reuse generation-time gradients. Every sampled thinking/action token receives loss; forced transitions,observations and template tokens have zero direct loss while remaining differentiable context. Reference and policy preserve original RoPE frequencies and the exact PS dependency graph with W1024,system-onlyprefix and original increasing positions. Independent native-versus-teacher BF16 likelihood parity remains unresolved; saved metrics report mismatch and clipping instead of claiming strict parity.

Generation uses temperature0.6,top-p1,no top-k,96responses,128sampled initial thinking tokens,then the existing11-token forced transition. A32sampled-token final-answer ceiling bounds post-transition output. The verified439-token system prefix explicitly requires one final Action: coordinate line,lowercase row+uppercase column,and no explanation or extra final-answer lines. No memory-note format or memory demonstrations are added. Streaming KV remains bounded by prefix+window.

AdamW uses FP32 masters,learning rate2e-6,betas0.9/0.95,epsilon1e-8,no weight decay,gradient norm clip1,8-update warmup and cosine decay to20percent of peak over128updates. The frozen reference retains FP32 buffers. Full model/optimizer/RNG snapshots save asynchronously every8updates and at the allocation boundary/final update. BF16 candidates save every16updates on node-local storage; persist selected/final exports before releasing the allocation.

Outputs include resolved configuration,dependency/revision locks,board permutation and generation seeds,complete compressed native traces/token records,per-update metrics and checkpoint metadata. Result files publish atomically. Recovery reuses only complete episodes with identical policy/seed/controller and verified token hashes; incomplete episodes restart while their logs remain preserved. Selected-checkpoint evaluation and memory interventions are separate work; sampled training gains alone do not establish causal textual memory.

CPU verification (29checks for deployed recovery source):

```bash
python -m pytest experiments/ps_concentration/test_hero_v2.py experiments/ps_concentration/test_rng_vllm.py experiments/ps_concentration/test_streaming_rl.py experiments/ps_concentration/test_objective.py -q
```

No separate GPU smoke or baseline evaluation preceded the real run. Speed,peak memory and KL overhead must be measured from run artifacts; five minutes/update remains a target.

The optional defer_native_logs setting buffers intermediate native game.json writes until finalization,while retaining every in-memory turn and existing incremental token logs. Interruption forces a flush before scoring. Tests compare every final byte and response count against the pinned native TurnLogger,including forced partial flushes and concurrent episodes. A source-only checkpoint migration requires accepted_resume_sources to explicitly list the reviewed previous source digest; every other saved identity field must match. These settings change logging and implementation provenance,not the game,prompt,reward,loss or optimizer schedule.

Preserve node-local actor timing and device reports before an allocation ends:

```bash
python -m experiments.ps_concentration.hero_telemetry --run-dir RUN
```

This CPU command copies complete actor reports into RUN/actor-telemetry,checks their collection policy,episode identities and token totals against every committed batch,and saves a hashed manifest. Existing copies must match their original bytes. Shared KV-pool allocation is reported separately from per-game logical retained KV. Run the command on the training node while its reports remain available; saved copies remain usable after node cleanup. It launches no model and changes no training state.

Prepare deferred validation with a captured evaluation implementation:

```bash
python -m experiments.ps_concentration.hero_validation_plan --run-dir RUN --output VALIDATION_DIR --policy-dir POLICIES --source-root SOURCE --source-sha256 SHA256
```

The plan launches no model. It fixes original full/PS controls and PS candidates every16updates through128,all on the reserved128validation boards,with four- and eight-GPU command layouts. Export paths may be prepared before exports exist; preserve and verify those files before execution. Run evaluations only after training and its workers finish. Selection uses the highest validation efficiency reward pairs/8 minus responses/96,with earlier updates winning ties; training error penalties remain separately reported. The original checkpoint remains eligible.

The existing rng_select and rng_final_plan CLIs accept this hero plan. They validate the nested hero training identity and the current128-thinking/32-answer controller,without presenting the shared inference RunConfig as the training algorithm. Final report training-episode counts derive from the saved global batch128. Its efficiency curve uses the unshaped efficiency reward; hero_report additionally plots the reward including error penalties. Native RNG score is pairs found. Greedy controls using this final controller remain necessary even when earlier unrestricted-thinking baselines were strong.

Verify each scheduled BF16 candidate while its matching FP32 snapshot is still available:

```bash
python -m experiments.ps_concentration.hero_exports --checkpoint RUN/checkpoints/latest.pt --policy NODE_POLICY_DIR --output RUN/export-catalog/policy-0016.json
```

This CPU command compares every effective BF16 parameter,including tied-weight handling,against the saved FP32 masters and records file hashes. It checks tokenizer vocabulary,native chat template,special tokens,architecture and unchanged RoPE/context settings. It supplies the original reference generation_config.json when that standalone export lacks it. The experiment.json marker must exist and both snapshot and candidate must have the same policy version. Receipts belong outside the export directory. This verifies an inference artifact; it runs no model,restores no optimizer and selects no checkpoint. Node-local weights still need preservation before releasing the allocation.
