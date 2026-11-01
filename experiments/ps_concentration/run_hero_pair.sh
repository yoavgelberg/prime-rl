#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use the authorized regular yejin allocation}"
[[ "$SLURM_JOB_PARTITION" == yejin ]]
allocation_root=${1:?Supply the existing lease directory}
source_root=${2:?Supply the immutable qualified source directory}
source_sha256=${3:?Supply its archive SHA256}
plan=${4:?Supply the frozen paired plan}
base=/hai/scratch/gelberg/ps/concentration
campaign="$base/rng-hero-v2-a1"
if [[ -z "${PS_DEPENDENCY_ROOT:-}" ]]; then
    for candidate in /tmp/ps-gelberg-dependencies-153918 /tmp/ps-gelberg-dependencies-154032 /tmp/ps-gelberg-dependencies-154069; do
        if [[ -x "$candidate/env/bin/python" ]]; then export PS_DEPENDENCY_ROOT="$candidate"; break; fi
    done
fi
if [[ -z "${PS_DEPENDENCY_ROOT:-}" ]]; then
    export PS_DEPENDENCY_ARCHIVE=/hai/scratch/gelberg/ps/runs/overlap-local-dependencies-cpu-a1/attempt-0
    exec bash /hai/scratch/gelberg/ps/sources/overlap-query-fields-b200-a3/harness/with_local_dependencies.sh bash "$0" "$@"
fi
parent_version=$("$PS_DEPENDENCY_ROOT/env/bin/python" -c 'import json;from pathlib import Path;p=Path("/hai/scratch/gelberg/ps/concentration/rng-hero-v2-a1/ps-seed11/state.json");print(json.loads(p.read_text())["policy_version"] if p.exists() else 0)')
if (( parent_version < 128 )); then
    bash "$base/run_hero_v2_a3.sh" "$allocation_root"
    parent_version=$("$PS_DEPENDENCY_ROOT/env/bin/python" -c 'import json;print(json.load(open("/hai/scratch/gelberg/ps/concentration/rng-hero-v2-a1/ps-seed11/state.json"))["policy_version"])')
    if (( parent_version < 128 )); then exit 0; fi
fi
export PYTHONPATH="$source_root/src:$source_root:$base/framework-site-a1:$base/report-site-v1:$base/screen-a2/site:$PS_DEPENDENCY_ROOT"
export OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1
export HF_HOME="$base/hf" WANDB_ENTITY=tech-ml WANDB_MODE=online
export TRITON_CACHE_DIR="$campaign/triton"
export HERO_PAIR_SOURCE_ROOT="$source_root" HERO_PAIR_SOURCE_SHA256="$source_sha256" HERO_PAIR_PLAN="$plan"
"$PS_DEPENDENCY_ROOT/env/bin/python" - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['HERO_PAIR_SOURCE_ROOT'])
expected=os.environ['HERO_PAIR_SOURCE_SHA256']
assert hashlib.sha256(root.with_suffix('.tar.gz').read_bytes()).hexdigest()==expected
plan=json.loads(Path(os.environ['HERO_PAIR_PLAN']).read_text())
assert plan['source_root']==str(root) and plan['source_sha256']==expected
PY
job_info=$(TZ=UTC scontrol show job --oneliner "$SLURM_JOB_ID")
end_stamp=$(printf '%s' "$job_info" | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')
deadline=$(date -u -d "$end_stamp" +%s)
gpus=$("$PS_DEPENDENCY_ROOT/env/bin/python" -c 'import os;print(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))')
node_base="/dev/shm/ps-hero-pair/$SLURM_JOB_ID-$(date -u +%s)"
cd "$source_root"
exec "$PS_DEPENDENCY_ROOT/env/bin/python" -u -m experiments.ps_concentration.hero_pair \
    --plan "$plan" --gpus "$gpus" --node-base "$node_base" --deadline "$deadline" \
    > "$allocation_root/hero-paired-$SLURM_JOB_ID-$(date -u +%s).log" 2>&1
