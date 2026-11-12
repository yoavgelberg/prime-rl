"""Supplementary W1024 memory interventions after the frozen paired main study."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ps_concentration.hero_exports import file_hash
from experiments.ps_concentration.hero_memory import execute, save_fixed
from experiments.ps_concentration.hero_pair_final import (
    branch_configs, completed_branches, condition_result, execute_entries,
    prepare_condition, require_idle_devices,
)
from experiments.ps_concentration.hero_v2 import read, utc, write
from experiments.ps_concentration.rng_report import analyze_condition

MAIN_PLAN_SHA256 = '1231c5121548630f1ef3e89be62bb2c345c8b4b1ec5a4d35a773736f2d3fd71d'


def validate_main(plan_path, main):
    if file_hash(plan_path) != MAIN_PLAN_SHA256:
        raise ValueError('Use the original frozen paired study')
    status = read(Path(main) / 'status.json')
    if status['status'] != 'PAIRED_RESULTS_AND_INTERVENTIONS_COMPLETE':
        raise ValueError('Complete the main study and interventions first')
    final = read(Path(main) / 'final/plan.json')
    selection = read(Path(main) / 'selection.json')
    if final['plan_sha256'] != MAIN_PLAN_SHA256 or final['selection'] != selection:
        raise ValueError('Use exactly the main validation-selected policies')
    if set(selection['selected']) != {'w1024', 'w256'}:
        raise ValueError('Both prespecified arms are required')
    return final, selection


def prepare(plan_path, selection, output, *, gpus):
    entries = [prepare_condition(plan_path, output,
        name=f'{arm}-ps-w1024-intervention-reference', arm=arm,
        version=row['version'], attention='ps', window=1024,
        split='intervention', count=512, gpus=gpus)
        for arm, row in selection['selected'].items()]
    return entries


def run(plan_path, main, output, protocol_path, *, gpus):
    plan_path, main, output = map(Path, (plan_path, main, output))
    protocol = read(protocol_path)
    if (protocol['main_plan_sha256'] != MAIN_PLAN_SHA256 or protocol['window'] != 1024
            or protocol['collection_boards_per_arm'] != 512 or protocol['maximum_states'] != 128
            or protocol['boundary'] != 'before_thinking' or protocol['samples_per_branch'] != 32
            or protocol['arms'] != ['w1024', 'w256']
            or protocol['collection_seed_range'] != [4000000, 4000511]
            or protocol['example_seed_range'] != [2000000, 2000019]
            or protocol['checkpoint_selection'] != 'reuse_main_validation_selection'):
        raise ValueError('Use the predeclared reference-window protocol')
    final, selection = validate_main(plan_path, main)
    plan = read(plan_path)
    completed_branches(plan)
    if os.environ.get('SLURM_JOB_PARTITION') != 'yejin':
        raise ValueError('Use only regular yejin')
    require_idle_devices(gpus)
    output.mkdir(parents=True, exist_ok=True)
    fixed = dict(main_plan_sha256=MAIN_PLAN_SHA256,
        main_final_plan_sha256=file_hash(main / 'final/plan.json'),
        selection_sha256=file_hash(main / 'selection.json'),
        protocol_sha256=file_hash(protocol_path), helper_sha256=file_hash(__file__))
    session_path = output / 'session.json'
    if session_path.exists():
        session = read(session_path)
        if {k:v for k,v in session.items() if k != 'layout'} != fixed:
            raise ValueError('Resume the same supplemental policies, protocol and implementation')
    else:
        session = dict(fixed, layout=gpus)
        save_fixed(session_path, session)
    entries = prepare(plan_path, selection, output / 'conditions', gpus=session['layout'])
    save_fixed(output / 'plan.json', dict(session=session, conditions=entries))
    execute_entries(plan_path, entries, output / 'execution', gpus=gpus)
    configs = branch_configs(plan)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs['w1024'].model)
    summaries = {}
    for entry in entries:
        arm = entry['arm']
        condition_result(entry)
        episodes = Path(entry['summary']).parent / 'episodes.jsonl'
        collection = output / 'memory' / arm / 'collection'
        inputs = dict(episodes=str(episodes), episodes_sha256=file_hash(episodes),
            checkpoint=entry['checkpoint'], window=1024, maximum=128, boundary='before_thinking')
        save_fixed(collection.parent / 'collection-inputs.json', inputs)
        if not (collection / 'collection-summary.json').exists():
            subprocess.run([sys.executable, '-m', 'experiments.ps_concentration.rng_intervention_states',
                '--episodes', str(episodes), '--tokenizer', configs[arm].model,
                '--output', str(collection), '--maximum', '128', '--boundary', 'before_thinking'], check=True)
        with gzip.open(collection / 'states.jsonl.gz', 'rt') as handle:
            states = [json.loads(line) for line in handle if line.strip()]
        if any(state['window'] != 1024 for state in states):
            raise ValueError('All supplemental intervention states must use W1024')
        summary = execute(entry['checkpoint'], collection / 'states.jsonl.gz',
            configs[arm].model, configs[arm].source, output / 'memory' / arm, gpus=gpus)
        example_entry = next(row for row in final['conditions'] if row['condition'] == f'{arm}-ps-w1024-test')
        example_condition = condition_result(example_entry)
        example_condition['rows'] = sorted(example_condition['rows'], key=lambda row: row['board_seed'])[:20]
        if [row['board_seed'] for row in example_condition['rows']] != list(range(2000000,2000020)):
            raise ValueError('Use the first20fixed final-board seeds without outcome selection')
        examples = analyze_condition(example_condition, tokenizer, examples=output / 'examples' / arm)
        summaries[arm] = dict(policy_version=entry['policy_version'], checkpoint=entry['checkpoint'],
            shared_parent_selected=entry['policy_version']==128, collection=entry['summary'],
            collection_sha256=file_hash(episodes), intervention_summary=summary,
            example_scope='First20fixed final-board seeds underW1024; not a new512-board task score',
            examples=examples)
    result = dict(utc=utc(), status='REFERENCE_W1024_MEMORY_COMPLETE', window=1024,
        primary_comparison_unchanged=True, session=session, arms=summaries,
        scope='Supplementary reference-window evidence, predeclared before final evaluation. '
              'Original validation selection is reused; no new checkpoint/window selection. '
              'One shared training seed; fewer than32qualified states is insufficient for textual memory.')
    write(output / 'results.json', result)
    lines = ['# Supplementary readable-memory checks at W1024', '',
        'Both policies use the original main-study validation selections. Main comparison and scores remain unchanged. '
        'This reference-window investigation was predeclared before final test evaluation. '
        'The same cold rebuild, false-association swap, and sham protocol applies. '
        'Fewer than 32 qualified states is insufficient for a textual-memory conclusion.', '',
        '| Arm | Selected update | Qualified states | Executed states | Causal textual-memory criterion |',
        '|---|---:|---:|---:|---|']
    for arm, row in summaries.items():
        s=row['intervention_summary']
        lines.append(f"| {arm} | {row['policy_version']} | {s['qualified_states']} | {s['executed_states']} | {s['causal_textual_memory_established']} |")
    lines += ['', '[Exact results and provenance](results.json)', '',
        'Per-arm tables and exact edits are in memory/. Twenty fixed-order greedy thinking traces per arm are in examples/.']
    (output / 'README.md').write_text('\n'.join(lines)+'\n')
    return dict(status=result['status'], output=str(output), arms=list(summaries))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--main-output',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--gpus',type=int,choices=(4,8),required=True)
    args=parser.parse_args()
    plan=read(args.plan)
    with (Path(plan['parent_run']).parent / 'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(run(args.plan,args.main_output,args.output,args.protocol,gpus=args.gpus)))


if __name__=='__main__':
    main()
