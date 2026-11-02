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

Fresh post-continuation evaluations use `post320_test` (seeds 7,000,000–7,000,511) and `post320_larger` (8,000,000–8,000,255). The latter uses the parameter-scaled 4×8 board, 16 pairs and 192-response cap. The 128-thinking/32-answer token controller and PS window remain fixed. The existing `extension_test` and `extension_larger` splits belong to the earlier fixed-320 experiment and must not be presented as fresh tests of this continuation.

Create a separate inference configuration from the verified run configuration, keeping its model, native source and controller:

```python
import json
from dataclasses import replace
from pathlib import Path
from experiments.ps_concentration.hero_protocol import inference_config
from experiments.ps_concentration.hero_v2 import HeroConfig

training = HeroConfig(**json.loads(Path('RUN_DIR/resolved_config.json').read_text()))
evaluation = replace(inference_config(training, output='EVALUATION_DIR', gpus=8),
                     evaluation_concurrency=64)
Path('EVALUATION_CONFIG.json').write_text(json.dumps(evaluation.model_dump(), indent=2) + '\n')
```

After training, evaluate a verified inference export, rather than a full Adam-state snapshot:

```bash
uv run python -m experiments.ps_concentration.rng_overlap evaluate --config EVALUATION_CONFIG.json --checkpoint VERIFIED_INFERENCE_EXPORT --attention ps --split post320_test
uv run python -m experiments.ps_concentration.rng_overlap evaluate --config EVALUATION_CONFIG.json --checkpoint VERIFIED_INFERENCE_EXPORT --attention ps --split post320_larger
```

Use separate output directories for each checkpoint and attention/window condition. All evaluations are greedy. Window diagnostics use `--window W --count 128` on the first 128 `post320_test` boards and cannot select a checkpoint or window. CPU tests validate split separation, four/eight-GPU board identity, larger-board worker options and report rejection of reused earlier test seeds. These commands do not launch an evaluation during training.

Prepare a fixed evaluation matrix from the continuation plan using a separately captured evaluation source:

```bash
uv run python -m experiments.ps_concentration.hero_post320_final prepare --plan RUN_DIR/plan.json --output FINAL_DIR --gpus 8 --source-root CAPTURED_EVALUATION_SOURCE --source-sha256 ARCHIVE_SHA256
```

The plan fixes update448, parent320 and the original checkpoint before inspecting the fresh tests. It includes512standardboards perpolicy, first128boards for extra windows256/512/2048/4096,256largerboards perpolicy, and a separate512-board intervention collection for the continued policy. W1024sweep points derive from the first128matching standard records. Three full-attention conditions are prepared separately. No test outcome selects a checkpoint or window.

After verified completion of all448training updates and retirement of training workers:

```bash
uv run python -m experiments.ps_concentration.hero_post320_final run --plan FINAL_DIR/plan.json --gpus 8
uv run python -m experiments.ps_concentration.hero_post320_final memory --plan FINAL_DIR/plan.json --gpus 8
uv run python -m experiments.ps_concentration.hero_post320_final run --plan FINAL_DIR/plan.json --gpus 8 --phase full
```

The default run executes PS conditions. Full attention is an explicit later phase. The memory command preserves the reserved4Msplit, prethinking qualification,128-state maximum,32-state conclusion minimum and common cold rebuild for unedited/swap/sham branches. It counts every qualified state and leaves insufficient evidence visible. Both commands reject incomplete training before any GPU work and resume only fixed, verified policy archives and condition configurations. Four-GPU fallback changes physical execution while preserving the same boards, controller and checkpoint identities. Use available private HAIC storage for final outputs when shared scratch headroom is insufficient.

Generate the integrated continuation report from the frozen final plan:

```bash
uv run python -m experiments.ps_concentration.hero_post320_report --final-plan FINAL_DIR/plan.json --output REPORT_DIR
```

Use `--allow-partial` for a dated progress snapshot while conditions remain incomplete. The strict report requires all448training updates, all22evaluation conditions and the completed memory analysis; zero qualifying intervention states remain a completed negative result. Tables separate512-board standard comparisons,128-board window diagnostics and256-board size generalization. The W1024sweep points use only the first128records of the standard tests. The paired448-minus320 interval uses10,000board resamples and does not imply independent training-seed replication. PNG/PDF figures, input hashes, twenty fixed-order continued-policy PS1024transcripts and intervention/cache controls derive from saved records. Reporting runs separately from the captured GPU evaluator and changes no training or evaluation source.
