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
