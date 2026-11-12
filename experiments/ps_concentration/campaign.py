"""Ordered screening, pilots, matched episode budgets, and final evaluations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .baselines import decide
from .control import Handoff, check_handoff


def invoke(module, arguments, log, *, world=1, device=None):
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}']
    command += ['-m', f'experiments.ps_concentration.{module}', *map(str, arguments)]
    environment = os.environ.copy()
    if device is not None:
        environment['CUDA_VISIBLE_DEVICES'] = device
    log.parent.mkdir(parents=True, exist_ok=True)
    log.with_suffix('.command.json').write_text(json.dumps(dict(argv=command,
        cuda_visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
        cuda_graphs=environment.get('PS_CONCENTRATION_GRAPHS')), indent=2))
    handle = log.open('a')
    process = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    return process


def finish(processes):
    codes = [p.wait() for p in processes]
    check_handoff()
    if any(codes):
        raise RuntimeError(f'Campaign stage failed: exit codes {codes}; inspect saved stage logs')


def wait_json(path, timeout=7200):
    started = time.monotonic()
    while not path.exists():
        check_handoff()
        if time.monotonic()-started > timeout:
            raise TimeoutError(f'Waiting for required evidence {path}')
        print(json.dumps(dict(event='waiting', path=str(path))), flush=True)
        time.sleep(30)
    return json.loads(path.read_text())


def screening(model, output, conditions, devices, logs):
    assignments = [conditions[i::len(devices)] for i in range(len(devices))]
    processes = []
    for i, assigned in enumerate(assignments):
        if assigned:
            processes.append(invoke('baselines', ['--model',model,'--output',output,'--batch-size',8,
                                                  '--conditions',*assigned], logs/f'screen-{i}.log', device=devices[i]))
    finish(processes)


def train_run(model, attention, seed, updates, window, root, devices, logs):
    output = root/f'{attention}-seed{seed}'
    memory=torch.cuda.get_device_properties(0).total_memory
    batch=6 if attention=='full' and memory<160*1024**3 else 8
    if attention=='full':
        batch=int(os.environ.get('PS_CONCENTRATION_FULL_BATCH',batch))
    command = ['--model', model, '--attention', attention, '--seed', seed, '--updates',updates,
               '--window',window,'--batch-size',batch,'--output',output,'--allow-numerical-target-miss']
    resume = output/'latest_optimizer.pt'
    checkpoint = output/f'checkpoint-{updates:04d}'
    receipt = output/'latest_optimizer.json'
    if checkpoint.exists() and resume.exists() and receipt.exists() and json.loads(receipt.read_text())['update'] >= updates:
        return output
    if resume.exists():
        command += ['--resume',resume]
    finish([invoke('train',command,logs/f'{attention}-{seed}-to-{updates}.log',world=len(devices))])
    return output


def read_validation(run, update):
    rows = [json.loads(line) for line in (run/'validation.jsonl').read_text().splitlines() if line.strip()]
    return next(row for row in rows if row['update'] == update)


def qualify_checkpoint_reload(args,devices):
    allocation=Path(os.environ.get('ALLOCATION_ROOT',args.run_dir))
    receipt=allocation/'checkpoint-qualification.json'
    if receipt.exists():
        return
    output=allocation/f'checkpoint-smoke-{time.time_ns()}'
    finish([invoke('backward_smoke',['--model',args.model,'--output',output,'--allow-numerical-target-miss'],
                   allocation/'checkpoint-qualification.log',device=devices[0])])
    receipt.write_text((output/'result.json').read_text())


def pilot(args, devices):
    root, logs = Path(args.run_dir), Path(args.run_dir)/'logs'
    root.mkdir(parents=True, exist_ok=True)
    qualification = Path(args.qualification)
    graph = wait_json(qualification/'graphs.json')
    assert graph['status'] == 'GO'
    backward = wait_json(qualification/'backward/result.json')
    assert backward['status'] == 'GO'
    screen = Path(args.screening)
    conditions = ['neutral_ps1024','neutral_ps512','neutral_ps2048','prompted_ps1024','prompted_ps512']
    if args.own_full_screening:
        conditions += ['neutral_full','prompted_full']
    screening(args.model,screen,conditions, devices, logs/'screening')
    full = wait_json(screen/'neutral_full.summary.json')
    prompted = wait_json(screen/'prompted_full.summary.json')
    results = {name: json.loads((screen/f'{name}.summary.json').read_text()) for name in
               ['neutral_full','prompted_full','neutral_ps1024','neutral_ps512','neutral_ps2048',
                'prompted_ps1024','prompted_ps512']}
    assert all(r['episodes'] == 64 for r in results.values())
    initial = decide(results)
    (root/'baseline-gates.json').write_text(json.dumps(initial,indent=2))
    (root/'baseline-screening.json').write_text(json.dumps(dict(gate=initial,conditions=results),indent=2))
    skip_full = full['matched_fraction'] >= .80
    selected_window = initial['window']
    full_run = None
    pilot_screen = None
    if not skip_full:
        full_run = train_run(args.model,'full',11,32,1024,root,devices,logs)
        pilot_screen = root/'full-pilot-screening'
        screening(str(full_run/'checkpoint-0032'),pilot_screen,
                  ['neutral_full','neutral_ps1024','neutral_ps512'],devices,logs/'full-pilot-screening')
        pilot_full = json.loads((pilot_screen/'neutral_full.summary.json').read_text())
        gaps = {w:pilot_full['matched_fraction']-json.loads((pilot_screen/f'neutral_ps{w}.summary.json').read_text())['matched_fraction']
                for w in (1024,512)}
        if prompted['matched_fraction'] < .80:
            selected_window = next((w for w in (1024,512) if gaps[w] >= .10), None)
    else:
        pilot_full = full
        gaps = {w:full['matched_fraction']-results[f'neutral_ps{w}']['matched_fraction'] for w in (1024,512)}
    exploratory_window = selected_window is None
    window = selected_window or 1024
    freeze = dict(window=window, selected_by_original_rule=not exploratory_window,
                  stage='before PS RL pilot', skip_full_rl=skip_full, original_baseline_gates=initial,
                  pilot_gaps={str(w):g for w,g in gaps.items()}, exploratory=exploratory_window,
                  reason='User permits diagnostic pilots on marginal gates; no task or prompt change' if exploratory_window else 'Largest qualifying candidate')
    window_file = root/'frozen_window.json'
    if window_file.exists():
        frozen = json.loads(window_file.read_text())
        window = frozen['window']
        assert window in (512,1024), 'Window cannot change after PS pilot starts'
        assert skip_full == frozen['skip_full_rl']
    else:
        window_file.write_text(json.dumps(freeze,indent=2))
    ps_run = train_run(args.model,'ps',11,32,window,root,devices,logs)
    ps0, ps32 = read_validation(ps_run,0), read_validation(ps_run,32)
    full_competence = read_validation(full_run,32)['matched_fraction'] if full_run else full['matched_fraction']
    gap = gaps[window]
    format_rate = max(ps32['format_error_rate'],pilot_full['format_error_rate'])
    improvement = ps32['native_return']-ps0['native_return']
    exact_go = full_competence >= .80 and gap >= .10 and format_rate < .01 and improvement >= .05
    marginal = full_competence >= .75 and gap >= .07 and format_rate < .02 and improvement >= -.01
    status = dict(status='GO' if exact_go else 'HOLD', exploratory_main_allowed=not exact_go and marginal,
                  major_failure=not exact_go and not marginal, window=window, skip_full_rl=skip_full,
                  full_matched_fraction=full_competence, sliding_gap=gap, format_error_rate=format_rate,
                  ps_return_improvement=improvement,
                  original_gates=dict(full_matched_fraction=.80,sliding_gap=.10,format_error_rate=.01,ps_return_improvement=.05),
                  exploratory_rule=dict(full_matched_fraction=.75,sliding_gap=.07,format_error_rate=.02,ps_return_improvement=-.01),
                  numerical_checks=backward['parity'], training_seed=11, updates_per_arm=32)
    (root/'pilot-status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2),flush=True)
    finish([invoke('report',['--run-dir',root,'--tokenizer',args.model],logs/'pilot-report.log')])


def main_matrix(args,devices):
    root, logs = Path(args.run_dir), Path(args.run_dir)/'logs'
    if not (root/'pilot-status.json').exists():
        args.own_full_screening = True
        pilot(args,devices)
    status = wait_json(root/'pilot-status.json',timeout=24*3600)
    if status['major_failure']:
        print(json.dumps(dict(status='HOLD',reason='Major pilot failure',gates=status)),flush=True)
        return
    arms = ['ps'] if status['skip_full_rl'] else ['ps','full']
    window = status['window']
    for target in (64,128,256):
        for seed in (11,22,33):
            for arm in arms:
                train_run(args.model,arm,seed,target,window,root,devices,logs)
    for seed in (11,22,33):
        for arm in arms:
            run = root/f'{arm}-seed{seed}'
            selected = json.loads((run/'selected.json').read_text())['checkpoint']
            finish([invoke('evaluate',['--checkpoint',selected,'--split','test','--window',window,'--window-sweep'],
                           logs/f'test-{arm}-{seed}.log',world=len(devices))])
            finish([invoke('evaluate',['--checkpoint',selected,'--split','larger','--window',window,'--attention','ps'],
                           logs/f'larger-{arm}-{seed}.log',world=len(devices))])
    for prompted in (False,True):
        command = ['--checkpoint',args.model,'--split','test','--window',window,
                   '--output',root/('prompted-original' if prompted else 'original')/'evaluation'/'test']
        if prompted:
            command += ['--prompted']
        finish([invoke('evaluate',command,logs/f'original-prompted{prompted}.log',world=len(devices))])
    interventions = []
    for index,seed in enumerate((11,22,33)):
        selected = json.loads((root/f'ps-seed{seed}'/'selected.json').read_text())['checkpoint']
        interventions.append(invoke('intervene',['--checkpoint',selected,'--window',window],
                                     logs/f'intervene-{seed}.log',device=devices[index%len(devices)]))
    finish(interventions)
    finish([invoke('report',['--run-dir',root,'--tokenizer',args.model],logs/'final-report.log')])
    (root/'completed').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['screen','pilot','main'],required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--qualification',default='/hai/scratch/gelberg/ps/concentration/qualification-a1')
    parser.add_argument('--screening',default='/hai/scratch/gelberg/ps/concentration/screening-qualified-a1')
    parser.add_argument('--own-full-screening',action='store_true')
    args=parser.parse_args()
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not devices[0]:
        devices=[str(i) for i in range(torch.cuda.device_count())]
    assert len(devices) in ((1,4,8) if args.phase=='screen' else (4,8))
    root=Path(args.run_dir)
    root.mkdir(parents=True,exist_ok=True)
    marker=root/'handoff.requested'
    os.environ['PS_CONCENTRATION_HANDOFF']=str(marker)
    allocation=Path(os.environ.get('ALLOCATION_ROOT',str(root)))
    with (root/'campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            owner_path=root/'allocation-owner.json'
            owner=json.loads(owner_path.read_text()) if owner_path.exists() else dict(gpus=8)
            if args.phase=='screen' or (args.phase!='main' and len(devices)<=owner['gpus']):
                (allocation/'completed').touch()
                return
            temporary=marker.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(owner_job=owner.get('job_id','*'),requester_job=os.environ.get('SLURM_JOB_ID'),
                                                   requester_gpus=len(devices))))
            temporary.replace(marker)
            print(json.dumps(dict(event='requesting_allocation_handoff',gpus=len(devices))),flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
        marker.unlink(missing_ok=True)
        (root/'allocation-owner.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),
                                                                phase=args.phase,gpus=len(devices))))
        try:
            if args.phase=='screen':
                finish([invoke('graph_check',['--model',args.model,'--output',allocation/'graphs.json'],allocation/'graph-check.log')])
                screening(args.model,Path(args.screening),['neutral_full','prompted_full','neutral_ps1024',
                          'neutral_ps512','neutral_ps2048','prompted_ps1024','prompted_ps512'],devices,allocation/'screening-logs')
                (allocation/'completed').touch()
            elif args.phase=='pilot':
                qualify_checkpoint_reload(args,devices)
                pilot(args,devices)
            else:
                qualify_checkpoint_reload(args,devices)
                main_matrix(args,devices)
        except Handoff:
            (allocation/'handoff.completed').touch()
            print(json.dumps(dict(event='allocation_handoff_complete')),flush=True)


if __name__ == '__main__':
    main()
