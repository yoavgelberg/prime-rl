#!/bin/bash
set -euo pipefail
run_root=/hai/scratch/gelberg/ps/concentration/overnight-a1
mkdir -p "$run_root"
printf '%s\n' "$SLURM_JOB_ID" > "$run_root/job_id"
date -u +%FT%TZ > "$run_root/allocation_start"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv > "$run_root/gpus.csv"
# The allocation is reserved ahead of implementation, at the user's request.
# Execute only an explicitly staged command after its source capture is ready.
while [[ ! -f "$run_root/ready" ]]; do sleep 10; done
exec bash "$run_root/launch.sh"
