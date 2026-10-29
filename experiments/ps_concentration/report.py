import argparse
import csv
import gzip
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .metrics import paired_bootstrap, summarize


def compact_records(paths):
    records = []
    excluded = {'tokens', 'positions', 'roles', 'loss_mask', 'old_logps', 'old_tempered_logps', 'turns'}
    for path in paths:
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                record = json.loads(line)
                records.append({k: v for k, v in record.items() if k not in excluded})
    assert len({r['board_seed'] for r in records}) == len(records), 'Duplicate final board evaluations'
    return records


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def transcript(record, tokenizer):
    ids, roles = record['tokens'], record['roles']
    p, w = record['prefix_length'], record['window']
    lines = [f'board_seed={record["board_seed"]} prefix={p} W={w} native_return={record["native_return"]}',
             f'origin={record.get("transcript_origin", "final test")}',
             f'For every query i >= {p+w}, suffix keys before i-{w}+1 have been evicted; prefix [0,{p}) stays pinned.']
    start = 0
    for end in range(1, len(ids)+1):
        if end == len(ids) or roles[end] != roles[start] or (end >= p+w and (end-p) % w == 0):
            lines.append(f'\n[tokens {start}:{end}; role={roles[start]}; retained suffix begins {max(p,end-w)}]\n')
            lines.append(tokenizer.decode(ids[start:end], skip_special_tokens=False))
            start = end
    return '\n'.join(lines)


def generate(args):
    root = Path(args.run_dir)
    out = Path(args.output or root/'report')
    out.mkdir(parents=True, exist_ok=True)
    catalog, tables = {}, []
    baseline_path = root/'baseline-screening.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline:
        write_csv(out/'baseline-screening.csv',[dict(condition=k,**{a:b for a,b in v.items() if a!='known_match'})
                                                for k,v in baseline['conditions'].items()])
    for metadata_path in sorted(root.rglob('evaluation.json')):
        meta = json.loads(metadata_path.read_text())
        experiment = meta.get('experiment')
        training = experiment['config']['attention'] if experiment else ('prompted_base' if meta['prompted'] else 'base')
        seed = experiment['config']['seed'] if experiment else None
        parent = metadata_path.parent
        conditions = sorted({p.name.split('-rank')[0] for p in parent.glob('*.jsonl.gz')})
        for condition in conditions:
            records = compact_records(sorted(parent.glob(f'{condition}-rank*.jsonl.gz')))
            if not records:
                continue
            attention, window = records[0]['attention'], records[0]['window']
            metric = summarize(records)
            expected = 256 if meta['split'] == 'larger' else 512 if window == meta['config']['window'] else 128
            row = dict(training=training, training_seed=seed, split=meta['split'], attention=attention, window=window,
                       expected_boards=expected, complete=len(records) == expected,
                       **{k:v for k,v in metric.items() if k != 'known_match'})
            tables.append(row)
            catalog[(training, seed, meta['split'], attention, window)] = (records, metric)
    write_csv(out/'main-comparison.csv', [r for r in tables if r['split'] == 'test'])
    write_csv(out/'size-generalization.csv', [r for r in tables if r['split'] == 'larger'])
    differences = []
    for (training, seed, split, attention, window), (records, metric) in catalog.items():
        if training != 'ps' or split != 'test' or attention != 'ps' or len(records) != 512:
            continue
        comparison = catalog.get(('full', seed, split, 'ps', window))
        label = 'Full RL under PS'
        full = catalog.get(('full', seed, split, 'full', window))
        if comparison is None:
            comparison = catalog.get(('base', None, split, 'ps', window))
            full = catalog.get(('base', None, split, 'full', window))
            label = 'Untrained base under PS; rule learning remains uncontrolled'
        if comparison is None:
            continue
        paired = paired_bootstrap(records, comparison[0], seed=seed)
        gap = full[1]['native_return'] - comparison[1]['native_return'] if full else None
        recovery = paired['difference']/gap if gap is not None and gap >= .10 else None
        differences.append(dict(training_seed=seed, comparator=label, window=window, **paired,
                                full_context_gap=gap, recovery=recovery))
    (out/'paired-differences.json').write_text(json.dumps(differences, indent=2))
    plt.figure(figsize=(7,4))
    curves = []
    pilot_rows = []
    for path in sorted(root.rglob('validation.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        label = path.parent.name
        plt.plot([r['episodes'] for r in rows], [r['native_return'] for r in rows], marker='o', label=label)
        curves.append(label)
        pilot_rows.extend(dict(run=label,**row) for row in rows)
    write_csv(out/'pilot-validation.csv',pilot_rows)
    plt.xlabel('Training episodes'); plt.ylabel('Validation native return')
    if curves: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/'return-vs-episodes.png', dpi=180); plt.close()
    plt.figure(figsize=(7,4))
    training_rows=[]
    for path in sorted(root.rglob('updates.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            label=path.parent.name
            plt.plot([r['episodes'] for r in rows],[r['native_return'] for r in rows],label=label)
            training_rows.extend(dict(run=label,update=r['update'],episodes=r['episodes'],native_return=r['native_return']) for r in rows)
    write_csv(out/'sampled-training-returns.csv',training_rows)
    plt.xlabel('Training episodes'); plt.ylabel('Fresh sampled-batch native return')
    if training_rows: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/'sampled-return-vs-episodes.png',dpi=180); plt.close()
    plt.figure(figsize=(7,4))
    grouped = {}
    sweep_rows=[]
    for (training,seed,split,attention,window),(records,metric) in catalog.items():
        if split == 'test' and attention == 'ps':
            subset=[r for r in records if 2_000_000 <= r['board_seed'] < 2_000_128]
            if len(subset)==128:
                row=dict(training=training,training_seed=seed,window=window,episodes=128,
                         native_return=float(np.mean([r['native_return'] for r in subset])))
                grouped.setdefault((training,seed),[]).append(row)
                sweep_rows.append(row)
    write_csv(out/'window-sweep-first128.csv',sweep_rows)
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda r:r['window'])
        plt.plot([r['window'] for r in rows], [r['native_return'] for r in rows], marker='o', label=str(label))
    if not grouped and baseline:
        rows = [(int(k.removeprefix('neutral_ps')),v['native_return']) for k,v in baseline['conditions'].items()
                if k.startswith('neutral_ps')]
        rows.sort()
        plt.plot([r[0] for r in rows],[r[1] for r in rows],marker='o',label='Original checkpoint screening')
        plt.title('Screening only; final test not run')
    plt.xscale('log', base=2); plt.xlabel('Rolling window W (tokens)'); plt.ylabel('Native return')
    if grouped or baseline: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/'return-vs-window.png', dpi=180); plt.close()
    plt.figure(figsize=(8,4))
    eligible = [(key, value[1]['known_match']) for key,value in catalog.items()
                if key[2] == 'test' and key[3] == 'ps' and len(value[0]) == 512]
    if not eligible:
        paths=sorted((root/'ps-seed11'/'validation').glob('update-0032-rank-*.jsonl.gz'))
        if paths:
            records=compact_records(paths)
            eligible=[(('PS RL pilot validation',11),summarize(records)['known_match'])]
    for index, (key, bins) in enumerate(eligible):
        x = np.arange(3) + (index-(len(eligible)-1)/2)*.8/max(1,len(eligible))
        plt.bar(x, [bins[b]['utilization'] if bins[b]['utilization'] is not None else np.nan
                    for b in ('<=W','(W,2W]','>2W')], width=.8/max(1,len(eligible)), label=f'{key[0]} seed={key[1]}')
    plt.xticks(range(3), ['≤W', '(W,2W]', '>2W']); plt.ylim(0,1)
    plt.xlabel('Age of latest direct environment reveal'); plt.ylabel('Known-match utilization')
    if eligible: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/'known-match-vs-age.png', dpi=180); plt.close()
    interventions = []
    for path in sorted(root.rglob('summary.json')):
        data = json.loads(path.read_text())
        if 'qualified_states' in data:
            interventions.append(dict(path=str(path), **data))
    (out/'interventions.json').write_text(json.dumps(interventions, indent=2))
    write_csv(out/'intervention-table.csv', [dict(source=r['path'], qualified_states=r['qualified_states'], branch=branch, **values)
                                            for r in interventions for branch,values in r['branches'].items()])
    aggregates = []
    for training in ('ps', 'full'):
        for attention in ('ps', 'full'):
            windows = sorted({r['window'] for r in tables if r['training'] == training and r['attention'] == attention})
            for window in windows:
                rows = [r for r in tables if r['training'] == training and r['attention'] == attention
                        and r['window'] == window and r['split'] == 'test']
                if rows:
                    aggregates.append(dict(training=training, attention=attention, window=window,
                                           training_seeds=len(rows), mean_native_return=float(np.mean([r['native_return'] for r in rows])),
                                           std_native_return=float(np.std([r['native_return'] for r in rows], ddof=1)) if len(rows)>1 else None))
    write_csv(out/'training-seed-aggregates.csv', aggregates)
    text_examples = 0
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        examples = []
        for path in sorted(root.rglob('ps-w*-rank*.jsonl.gz')):
            if 'test' not in path.parts:
                continue
            metadata = json.loads((path.parent/'evaluation.json').read_text())
            experiment = metadata.get('experiment')
            if not experiment or experiment['config']['attention'] != 'ps' or experiment['config']['seed'] != 11:
                continue
            if not path.name.startswith(f'ps-w{metadata["config"]["window"]}-rank'):
                continue
            with gzip.open(path, 'rt') as handle:
                for line in handle:
                    record = json.loads(line)
                    if 2_000_000 <= record['board_seed'] < 2_000_020:
                        examples.append(record)
            if len(examples) >= 20: break
        if not examples:
            for path in sorted((root/'ps-seed11'/'validation').glob('update-0032-rank-*.jsonl.gz')):
                with gzip.open(path,'rt') as handle:
                    for line in handle:
                        record=json.loads(line)
                        if 1_100_000 <= record['board_seed'] < 1_100_020:
                            record['transcript_origin']='pilot validation, final test not run'
                            examples.append(record)
        (out/'transcripts').mkdir(exist_ok=True)
        for index, record in enumerate(sorted(examples, key=lambda r:r['board_seed'])):
            (out/'transcripts'/f'{index:02d}-seed{record["board_seed"]}.txt').write_text(transcript(record, tokenizer))
        text_examples = len(examples)
    pilot_status_path=root/'pilot-status.json'
    pilot_status=json.loads(pilot_status_path.read_text()) if pilot_status_path.exists() else None
    result = dict(final_conditions=tables, paired_differences=differences, intervention_results=interventions,
                  pilot_status=pilot_status, baseline_screening=baseline,
                  training_seed_aggregates=aggregates,
                  training_curves=curves, example_transcripts=text_examples,
                  status='HOLD' if pilot_status and pilot_status['major_failure'] else
                         'COMPLETE_TABLES' if tables and all(r['complete'] for r in tables) else 'INCOMPLETE',
                  training_variability='Three training seeds determine training variability; bootstrap intervals are over boards only')
    (out/'results.json').write_text(json.dumps(result, indent=2))
    body = '<h1>Concentration memory RL results</h1>'
    body += f'<p>Status: {result["status"]}. Tables and plots are derived from saved episode and update records.</p>'
    for name in ('return-vs-episodes','sampled-return-vs-episodes','return-vs-window','known-match-vs-age'):
        body += f'<img src="{name}.png" style="max-width:800px"><br>'
    body += '<pre>'+html.escape(json.dumps(result, indent=2))+'</pre>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8">'+body)
    print(json.dumps(dict(output=str(out), status=result['status'], conditions=len(tables)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output')
    parser.add_argument('--tokenizer')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
