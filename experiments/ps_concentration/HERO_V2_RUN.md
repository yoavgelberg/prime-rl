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
