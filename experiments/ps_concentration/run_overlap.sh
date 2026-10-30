#!/usr/bin/env bash
set -euo pipefail
: "${PS_SOURCE_ROOT:?Set the immutable captured source directory}"
: "${PS_SOURCE_SHA256:?Set the captured source archive digest}"
: "${SLURM_JOB_ID:?Use an allocated regular yejin job}"
allocation_root=${1:?Supply the existing allocation lease directory}
base=/hai/scratch/gelberg/ps/concentration
campaign="$base/rng-overlap-grpo-a1"
mkdir -p "$campaign"
exec 8>"$campaign/training.lock"
if ! flock -n 8; then
    printf 'Another allocation owns this pilot; no duplicate work was started.\n'
    exit 0
fi
[[ "$SLURM_JOB_PARTITION" == yejin ]]
export ALLOCATION_ROOT="$allocation_root"
export JOB_OUTPUT="$allocation_root"
export PS_CONCENTRATION_HANDOFF_MARGIN=1800
job_info=$(TZ=UTC scontrol show job --oneliner "$SLURM_JOB_ID")
end_stamp=$(printf '%s' "$job_info" | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')
export PS_CONCENTRATION_END_EPOCH="$(date -u -d "$end_stamp" +%s)"

if [[ -z "${PS_DEPENDENCY_ROOT:-}" ]]; then
    for candidate in /tmp/ps-gelberg-dependencies-153918 /tmp/ps-gelberg-dependencies-154032 /tmp/ps-gelberg-dependencies-154069; do
        if [[ -x "$candidate/env/bin/python" ]]; then export PS_DEPENDENCY_ROOT="$candidate"; break; fi
    done
fi
if [[ -z "${PS_DEPENDENCY_ROOT:-}" ]]; then
    export PS_DEPENDENCY_ARCHIVE=/hai/scratch/gelberg/ps/runs/overlap-local-dependencies-cpu-a1/attempt-0
    flock -u 8
    exec 8>&-
    exec bash /hai/scratch/gelberg/ps/sources/overlap-query-fields-b200-a3/harness/with_local_dependencies.sh bash "$0" "$@"
fi
export PS_PYTHON_BIN="$PS_DEPENDENCY_ROOT/env/bin/python"
export PYTHONPATH="$PS_SOURCE_ROOT/src:$PS_SOURCE_ROOT:$base/framework-site-a1:$base/report-site-v1:$base/screen-a2/site:$PS_DEPENDENCY_ROOT"
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1
export HF_HOME="$base/hf"
export WANDB_ENTITY=tech-ml WANDB_MODE=online
export TRITON_CACHE_DIR="$campaign/triton"
export PS_CONCENTRATION_GRAPHS=1
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm must set assigned GPUs}"
export PS_RUN_GPUS="${#devices[@]}"
[[ "$PS_RUN_GPUS" == 4 || "$PS_RUN_GPUS" == 8 ]]
cd "$PS_SOURCE_ROOT"
"$PS_PYTHON_BIN" - "$campaign" <<'PY'
import json, os, sys
from pathlib import Path
from experiments.ps_concentration.overlap_config import RunConfig
from experiments.ps_concentration.rng_overlap import write_json
root = Path(sys.argv[1])
base = root.parent
model = base / 'hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea'
source = base / 'rngbench-source-e3d195e857727dfa3d4a96561be202098d67da83'
for name, concurrency in [('full-reference', 4), ('ps-seed11', 128 // int(os.environ['PS_RUN_GPUS']))]:
    config = RunConfig(model=str(model), source=str(source), output=str(root / name),
        source_sha256=os.environ['PS_SOURCE_SHA256'], gpus=int(os.environ['PS_RUN_GPUS']),
        evaluation_concurrency=concurrency)
    write_json(root / f'{name}-config-{os.environ["SLURM_JOB_ID"]}.json', config.model_dump())
write_json(root / f'launch-{os.environ["SLURM_JOB_ID"]}.json', dict(
    source_root=os.environ['PS_SOURCE_ROOT'], source_sha256=os.environ['PS_SOURCE_SHA256'],
    dependency_root=os.environ['PS_DEPENDENCY_ROOT'], gpus=int(os.environ['PS_RUN_GPUS']),
    partition=os.environ['SLURM_JOB_PARTITION'], no_extra_gpu_smoke=True,
    maximum_allocation_gpu_hours=int(os.environ['PS_RUN_GPUS']) * 24))
PY
full_config="$campaign/full-reference-config-$SLURM_JOB_ID.json"
ps_config="$campaign/ps-seed11-config-$SLURM_JOB_ID.json"
competence="$campaign/full-reference/evaluation/validation/summary.json"
if [[ ! -f "$competence" ]]; then
    "$PS_PYTHON_BIN" -u -m experiments.ps_concentration.rng_overlap evaluate --config "$full_config" \
        --attention full --split validation > "$allocation_root/full-reference-$SLURM_JOB_ID.log" 2>&1
fi
"$PS_PYTHON_BIN" -u -m experiments.ps_concentration.rng_overlap train --config "$ps_config" \
    --attention ps --competence "$competence" > "$allocation_root/ps-grpo-$SLURM_JOB_ID.log" 2>&1
