#!/usr/bin/env bash
set -uo pipefail
root=$1
mkdir -p "$root/commands"
printf '%s\n' "${SLURM_JOB_ID:?}" > "$root/job_id"
date -u +%FT%TZ > "$root/allocation_start-${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv > "$root/gpus-${SLURM_RESTART_COUNT:-0}.csv"
trap 'touch "$root/shutdown.requested"' USR1
attempt=0
while true; do
 if [[ -f "$root/finish" || -f "$root/completed" ]]; then exit 0; fi
 if [[ ! -f "$root/next" ]]; then sleep 2; continue; fi
 task=$(cat "$root/next")
 mv "$root/next" "$root/commands/consumed-$attempt"
 date -u +%FT%TZ > "$root/attempt-$attempt-started"
 sha256sum "$task" > "$root/attempt-$attempt-command.sha256"
 printf 'Starting stage %s: %s\n' "$attempt" "$task"
 bash "$task" > "$root/attempt-$attempt.log" 2>&1 &
 child=$!
 wait "$child"
 code=$?
 if kill -0 "$child" 2>/dev/null; then wait "$child"; code=$?; fi
 printf '%s\n' "$code" > "$root/attempt-$attempt-exit-code"
 date -u +%FT%TZ > "$root/attempt-$attempt-ended"
 printf 'Stage %s exited %s\n' "$attempt" "$code"
 if [[ -f "$root/handoff.completed" || -f /hai/scratch/gelberg/ps/concentration/campaign-a1/completed ]]; then exit 0; fi
 attempt=$((attempt+1))
done
