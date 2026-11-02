# W1024 continuation after update 320

The continuation restores the trained W1024 model, Adam state and RNG state from update 320. Its initial budget is 128 additional updates, through absolute update 448. Sampling temperature is 0.8, top-p is 1 and top-k is disabled. The new eight-update warmup reaches learning rate 1e-6 at update 328. Global batch 128, group size 8, exact categorical KL coefficient 0.01, entropy coefficient 0.001 and the 128-thinking/32-answer token controller remain fixed.

Use the captured fork receipt, source archive, dependency locks and runtime configuration to launch the trainer:

```bash
uv run python -m experiments.ps_concentration.hero_v2 --config VERIFIED_RUNTIME_CONFIG.json
```

The runtime configuration selects four or eight assigned GPUs while preserving the global batch and group size. Four actor GPUs run 32 games each in the eight-GPU layout. Forced controller tokens have zero direct loss; all sampled assistant tokens contribute to the normalized objective. Rollouts retain streaming prefix-sliding KV with original position IDs.

Generate figures and CSV from complete committed updates:

```bash
uv run python -m experiments.ps_concentration.hero_report --run-dir RUN_DIR --output REPORT_DIR
```

For a continuation, the report reads `fork-receipt.json` and requires every update after its recorded parent. It keeps additional-run episodes separate from cumulative policy-lineage episodes and excludes uncommitted updates. Figures are written as PNG and PDF with input hashes and configuration snapshots.

This is a single-lineage exploratory phase after inspecting earlier results. Sampled training curves and retained scratch associations do not establish causal use of generated text. The separate-seed cold-rebuild, unedited, swap and sham interventions remain necessary for that claim. Native actor/learner numerical parity remains unresolved. The prior fixed-320 checkpoints and deferred evaluations are preserved separately.
