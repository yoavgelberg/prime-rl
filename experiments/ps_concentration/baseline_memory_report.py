"""Offline memory diagnostics for the completed historical 64-board paired baseline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .rng_memory_metrics import ANALYSIS_VERSION, known_match_decisions, summarize
from .rng_metrics import score_prefix


TOKENIZER = None
BINS = ['<=W', '(W,2W]', '>2W']


def initialize_tokenizer(path):
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)


def analyze(task):
    root, output, attention, seed = task
    root, output = Path(root) / attention, Path(output) / attention
    output.mkdir(parents=True, exist_ok=True)
    path = root / f'seed-{seed}-tokens.json.gz'
    game_path = root / f'seed-{seed}/game.json'
    receipt_path = root / f'seed-{seed}-result.json'
    with gzip.open(path, 'rt') as handle:
        record = json.load(handle)
    game = json.loads(game_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if record['attention'] != attention or record['window'] != 1024 or game['config']['seed'] != seed:
        raise ValueError('Historical baseline identities differ from the frozen paired comparison')
    decisions = known_match_decisions(record, game, TOKENIZER)
    prefix = record['tokens'][:record['prefix_length']]
    result = dict(attention=attention, board_seed=seed, known_match=summarize(decisions),
        prefix_length=record['prefix_length'], prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
        board_identity_sha256=hashlib.sha256(game['config']['ground_truth_text'].encode()).hexdigest(),
        token_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        game_sha256=hashlib.sha256(game_path.read_bytes()).hexdigest(),
        source_token_record=str(path), source_native_game=str(game_path),
        censored=bool(receipt['censored']), native_score=score_prefix(game, 96),
        analysis_version=ANALYSIS_VERSION, completed_decisions=len(decisions))
    with gzip.open(output / f'board-{seed}-decisions.json.gz', 'wt', compresslevel=1) as handle:
        json.dump(dict(result, decisions=decisions), handle)
    (output / f'board-{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def summarize_rows(rows):
    groups = {attention:sorted([row for row in rows if row['attention'] == attention], key=lambda row:row['board_seed'])
              for attention in ('full', 'ps')}
    if any([row['board_seed'] for row in group] != list(range(1000000, 1000064)) for group in groups.values()):
        raise ValueError('The historical baseline must contain all64 paired board seeds in both conditions')
    prefixes = {row['prefix_sha256'] for row in rows}
    if len(prefixes) != 1:
        raise ValueError('Baseline system prefixes differ; inspect before aggregating memory diagnostics')
    for left, right in zip(groups['full'], groups['ps'], strict=True):
        if left['board_identity_sha256'] != right['board_identity_sha256']:
            raise ValueError('Paired policies did not reset to the identical native board')
    summary, differences = {}, {}
    rng = np.random.default_rng(0)
    draw = rng.integers(64, size=(10000, 64))
    for attention, group in groups.items():
        bins = {}
        for name in BINS:
            opportunities = sum(row['known_match'][name]['opportunities'] for row in group)
            correct = sum(row['known_match'][name]['correct'] for row in group)
            bins[name] = dict(opportunities=opportunities, correct=correct,
                boards_with_opportunities=sum(row['known_match'][name]['opportunities'] > 0 for row in group),
                utilization=correct / opportunities if opportunities else None)
        summary[attention] = dict(boards=64, context_censors=sum(row['censored'] for row in group), known_match=bins)
    for name in BINS:
        values = {}
        for attention, group in groups.items():
            correct = np.array([row['known_match'][name]['correct'] for row in group])
            count = np.array([row['known_match'][name]['opportunities'] for row in group])
            denominator = count[draw].sum(-1)
            ratios = np.full(10000, np.nan)
            np.divide(correct[draw].sum(-1), denominator, out=ratios, where=denominator > 0)
            values[attention] = ratios
        contrast = values['full'] - values['ps']
        finite = contrast[np.isfinite(contrast)]
        a, b = summary['full']['known_match'][name]['utilization'], summary['ps']['known_match'][name]['utilization']
        differences[name] = dict(full_minus_ps=a - b if a is not None and b is not None else None,
            ci95=[float(value) for value in np.quantile(finite, [.025, .975])] if len(finite) else None,
            requested_resamples=10000, defined_resamples=len(finite),
            uncertainty='Paired board bootstrap of pooled opportunity ratios; policies visit different decision states; no training variability')
    return dict(conditions=summary, paired_utilization_difference=differences,
        analysis_version=ANALYSIS_VERSION, system_prefix_sha256=next(iter(prefixes)),
        scope='Historical UNTRAINED sampled baseline, custom StreamingQwen arithmetic; not native-vLLM greedy qualification or an RL result',
        decoding='Temperature .6, top-p .95, top-k20; native thinking retained; Action-only final;96attempt cap;262144token cap',
        censoring='Completed responses scored through their actual stop; unfinished context-censored responses excluded; no imputed96attempt endpoints',
        age='Original query before FIRST coordinate token minus latest direct environment rank-token reveal; notes do not refresh age')


def save_report(result, output):
    output = Path(output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'known-match-table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['attention', 'age_bin', 'opportunities', 'correct', 'boards_with_opportunities', 'utilization'])
        writer.writeheader()
        for attention, summary in result['conditions'].items():
            for name, values in summary['known_match'].items():
                writer.writerow(dict(attention=attention, age_bin=name, **values))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for attention in ('full', 'ps'):
        ax.plot(BINS, [result['conditions'][attention]['known_match'][name]['utilization'] for name in BINS],
                marker='o', label=attention)
    ax.set(ylim=(0, 1.05), xlabel='Age of direct environment reveal', ylabel='Known-match utilization',
           title='Untrained sampled baseline:64pairedboards')
    ax.legend(frameon=False)
    fig.text(.5, .01, 'Policies visit different states. Censored games contribute completed responses only.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(output / 'known-match-utilization.png', dpi=180); plt.close(fig)
    lines = ['# Historical baseline memory diagnostic', '', result['scope'], '', result['decoding'], '', result['censoring'], '',
        '| Attention | Reveal age | Opportunities | Correct | Utilization | Boards with opportunities |',
        '|---|---|---:|---:|---:|---:|']
    for attention, summary in result['conditions'].items():
        for name, values in summary['known_match'].items():
            utilization = f'{values["utilization"]:.4f}' if values['utilization'] is not None else 'undefined'
            lines.append(f'| {attention} | {name} | {values["opportunities"]} | {values["correct"]} | {utilization} | {values["boards_with_opportunities"]} |')
    lines += ['', result['age'], '', 'Intervals resample paired boards, retaining all decisions within each board. Opportunity sets differ between policies; this is a behavioral diagnostic, not a shared-state causal comparison. No learned-memory conclusion follows from this untrained baseline.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    tasks = [(args.run_dir, args.output, attention, seed) for attention in ('full', 'ps') for seed in range(1000000, 1000064)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_tokenizer, initargs=(args.tokenizer,)) as pool:
        futures = [pool.submit(analyze, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=len(tasks), attention=row['attention'], board_seed=row['board_seed'])), flush=True)
    result = summarize_rows(rows)
    result['cpu_analysis_seconds'] = time.perf_counter() - started
    save_report(result, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
