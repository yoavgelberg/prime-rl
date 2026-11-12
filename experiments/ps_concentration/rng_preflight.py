"""Run one source-locked maximum-length audit before the native RL pilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def accepted(result, source_sha):
    return (result['source_sha256'] == source_sha and result['raw_tokens'] == 262144
        and result['learned_parameter_updates'] == 0
        and result['zero_gradient_optimizer_weights_unchanged']
        and result['early_loss_tokens'] > 0 and result['optimizer_states_resident']
        and result['mean_abs'] <= .02 and result['p99'] <= .15)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source = os.environ['PS_CONCENTRATION_SOURCE_SHA']
    assert len(source) == 64
    with (root / 'audit.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = root / 'accepted.json'
        if receipt.exists() and json.loads(receipt.read_text())['source_sha256'] == source:
            return
        failed = root / 'failed.json'
        if failed.exists() and json.loads(failed.read_text())['source_sha256'] == source:
            raise RuntimeError('This exact source already failed preflight; inspect it rather than repeating the failed audit')
        result_path = root / f'result-{source[:12]}.json'
        log_path = root / f'job-{os.environ.get("SLURM_JOB_ID", "local")}-{source[:12]}.log'
        command = [sys.executable, '-u', '-m', 'experiments.ps_concentration.rng_memory_check',
                   '--model', args.model, '--record', args.record, '--output', str(result_path),
                   '--offload', '--zero-step-smoke']
        with log_path.open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        valid = process.returncode == 0 and result_path.exists()
        result = json.loads(result_path.read_text()) if valid else None
        valid = valid and accepted(result, source)
        metadata = dict(source_sha256=source, job=os.environ.get('SLURM_JOB_ID'),
            result=str(result_path), log=str(log_path), exit_code=process.returncode, unix_time=time.time(),
            strict_parity_status=result.get('status') if result else None,
            exploratory_numerical_limits=dict(mean_abs=.02, p99=.15))
        atomic_json(receipt if valid else failed, metadata)
        if not valid:
            raise RuntimeError(f'Maximum-length preflight failed; see {log_path}')
        print(json.dumps(dict(event='maximum_length_preflight_accepted', **metadata)), flush=True)


if __name__ == '__main__':
    main()
