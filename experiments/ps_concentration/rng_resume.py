"""Preserve interrupted streaming games before starting a fresh allocation."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import time


TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','PREEMPTED','OUT_OF_MEMORY',
          'NODE_FAIL','BOOT_FAIL','REVOKED','DEADLINE'}
RESULT_FIELDS=('seed','score','total_pairs','rounds_played','response_count','done','error','censored',
               'turn_cap_failure','terminal_outcome_reward','generated_tokens','raw_transcript_tokens',
               'seconds','attention','window','generation_seed','max_responses','batch_size','peak_memory_scope')


def predecessor_state(job_id):
    active=subprocess.check_output(['squeue','-h','-j',str(job_id),'-o','%T'],text=True).strip()
    if active:
        raise RuntimeError(f'Predecessor {job_id} is still present in squeue: {active}')
    output=subprocess.check_output(['sacct','-nP','-j',str(job_id),'--format=JobIDRaw,State'],text=True)
    states=[line.split('|')[1].split()[0] for line in output.splitlines()
            if line.split('|')[0]==str(job_id)]
    if not states or states[-1] not in TERMINAL:
        raise RuntimeError(f'Predecessor {job_id} has no verified terminal state: {states}')
    return states[-1]


def recover_receipt(directory,seed):
    path=directory/f'seed-{seed}-tokens.json.gz'
    if not path.exists():
        return None
    try:
        with gzip.open(path,'rt') as handle:
            record=json.load(handle)
    except (EOFError,OSError,json.JSONDecodeError):
        return None
    if record.get('seed')!=seed or record.get('error'):
        return None
    terminal=(record.get('done') and record.get('score')==record.get('total_pairs')) or record.get('censored')
    terminal=terminal or (record.get('turn_cap_failure') and record.get('response_count')==record.get('max_responses'))
    if not terminal:
        return None
    tokens=record.get('tokens',[])
    masks=record.get('loss_mask',[])
    logps=record.get('old_logps',[])
    if not tokens or len(tokens)!=len(masks) or len(tokens)!=len(logps):
        return None
    if record.get('generated_tokens')!=sum(masks) or any(p is None for p,m in zip(logps,masks) if m):
        return None
    return {key:record[key] for key in RESULT_FIELDS if key in record}


def prepare(run_dir,state,job_id):
    if state not in TERMINAL:
        raise RuntimeError(f'Refusing to archive an unverified predecessor: {state}')
    root=Path(run_dir)
    recovery=root/f'interrupted-before-{job_id}-{time.time_ns()}'
    result=dict(predecessor_state=state,current_job=str(job_id),recovered=[],archived=[])
    for attention in ('full','ps'):
        directory=root/attention
        if not directory.exists():
            continue
        seeds=sorted({int(match[1]) for path in directory.glob('seed-*')
                      if (match:=re.match(r'^seed-(\d+)(?:$|-)',path.name))})
        for seed in seeds:
            receipt=directory/f'seed-{seed}-result.json'
            valid=False
            if receipt.exists():
                try:
                    row=json.loads(receipt.read_text())
                    valid=row.get('seed')==seed and row.get('error') is None and (
                        row.get('done') or row.get('censored') or row.get('turn_cap_failure'))
                except json.JSONDecodeError:
                    pass
            if valid:
                continue
            row=recover_receipt(directory,seed)
            if row is not None:
                if receipt.exists():
                    destination=recovery/attention
                    destination.mkdir(parents=True,exist_ok=True)
                    receipt.rename(destination/receipt.name)
                temporary=receipt.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(row,indent=2));temporary.replace(receipt)
                result['recovered'].append(dict(attention=attention,seed=seed))
                continue
            paths=[directory/f'seed-{seed}']+list(directory.glob(f'seed-{seed}-*'))
            paths.append(directory/f'progress-{seed}-batch.json')
            moved=[]
            for path in paths:
                if not path.exists():
                    continue
                destination=recovery/attention
                destination.mkdir(parents=True,exist_ok=True)
                path.rename(destination/path.name)
                moved.append(path.name)
            if moved:
                result['archived'].append(dict(attention=attention,seed=seed,files=moved,
                    restart='Fresh reset and original RNG; no native partial replay or retained-suffix rebuild.'))
    receipt=root/f'resume-preflight-{job_id}.json'
    receipt.write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--prior-job',required=True,type=int)
    args=parser.parse_args()
    state=predecessor_state(args.prior_job)
    result=prepare(args.run_dir,state,os.environ.get('SLURM_JOB_ID','manual'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
