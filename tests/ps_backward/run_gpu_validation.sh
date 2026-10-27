#!/usr/bin/env bash
set -euo pipefail
: "${PS_PYTHON:?Set PS_PYTHON to the pinned vLLM/FA4 environment Python}"
: "${PS_PROMPTS:?Set PS_PROMPTS to JSONL prompt token IDs}"
: "${JOB_OUTPUT:?Set JOB_OUTPUT to an empty output directory}"
export PYTHONPATH="$PWD/src:$PWD/tests/ps_backward"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_DEVICE_MAX_COPY_CONNECTIONS=32
if [[ ${PS_SKIP_KERNEL_TESTS:-0} != 1 ]]; then
  "$PS_PYTHON" -m pytest --confcutdir=tests/ps_backward -c tests/ps_backward/pytest.ini tests/ps_backward -q | tee "$JOB_OUTPUT/tests.txt"
fi
common=(--model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
        --prompts "$PS_PROMPTS" --warm-length 8192)
"$PS_PYTHON" examples/ps_backward/generate.py "${common[@]}" --length 16384 --requests 2 \
  --output "$JOB_OUTPUT/native-short"
"$PS_PYTHON" examples/ps_backward/generate.py "${common[@]}" --length 16384 --requests 2 --streaming \
  --worker audit_worker.AuditWorker --audit --output "$JOB_OUTPUT/stream-short-audit"
"$PS_PYTHON" examples/ps_backward/generate.py "${common[@]}" --length 174763 --requests 1 --streaming \
  --worker audit_worker.AuditWorker --audit --output "$JOB_OUTPUT/stream-128k-audit"
