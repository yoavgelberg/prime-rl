"""Prepare held-out commands and report inputs after deferred validation selection."""

import argparse
import hashlib
import json
from pathlib import Path

from .overlap_config import RunConfig, SPLITS, board_protocol
from .rng_select import argument, read_json, select_checkpoint


def prepare_final_plan(validation_plan, run, selection_path, output, *, gpus, validation_gpus):
    if gpus not in (4, 8):
        raise ValueError('Final evaluation requires the reserved four- or eight-B200 layout')
    validation_plan, run, selection_path, output = map(Path, (validation_plan, run, selection_path, output))
    selected = select_checkpoint(validation_plan, run, gpus=validation_gpus)
    if read_json(selection_path) != selected:
        raise ValueError('Selection receipt differs from the complete prespecified validation results')
    plan = read_json(validation_plan)
    base = next(entry for entry in plan['layouts'][str(validation_gpus)] if entry['condition'] == 'base-ps')
    config = RunConfig(**read_json(argument(base['argv'], '--config')))
    trained = selected['update'] > 0
    checkpoint = selected['checkpoint'] if trained else None
    files, commands, conditions = {}, [], []

    def condition(name, *, policy, attention, split, count, window=None, include_report=True):
        folder = output / name
        config_path = output / 'configs' / f'{name}.json'
        current = config.model_dump() | dict(output=str(folder), gpus=gpus,
            evaluation_concurrency=4 if attention == 'full' else 128 // gpus)
        RunConfig(**current)
        data = (json.dumps(current, indent=2, allow_nan=False) + '\n').encode()
        files[config_path] = data
        argv = ['python', '-m', 'experiments.ps_concentration.rng_overlap', 'evaluate',
            '--config', str(config_path), '--attention', attention, '--split', split, '--count', str(count)]
        if policy == 'PS RL':
            argv.extend(['--checkpoint', checkpoint])
        if window is not None:
            argv.extend(['--window', str(window)])
        effective_window = config.window if window is None else window
        summary = folder / 'evaluation' / split / 'summary.json'
        start, maximum = SPLITS[split]
        if not 1 <= count <= maximum:
            raise ValueError('Final evaluation must remain within its reserved board split')
        commands.append(dict(condition=name, argv=argv, config_sha256=hashlib.sha256(data).hexdigest(),
            policy=policy, policy_version=selected['update'] if policy == 'PS RL' else 0,
            attention=attention, window=effective_window, split=split, count=count,
            first_board_seed=start, last_board_seed=start + count - 1,
            board=board_protocol(split).identity(), summary=str(summary)))
        if include_report:
            conditions.append(dict(name=name, policy=policy, training_seed=config.seed if policy == 'PS RL' else None,
                attention=attention, window=effective_window, split=split, greedy=True, expected_boards=count,
                policy_version=selected['update'] if policy == 'PS RL' else 0,
                policy_identity=str(Path(checkpoint).resolve()) if policy == 'PS RL' else config.revision,
                controller={key: current[key] for key in
                    ('thinking_token_limit', 'action_token_limit', 'thinking_budget_transition')},
                episodes=str(summary.parent / 'episodes.jsonl'), summary=str(summary)))
        return name

    base_full = condition('base-full-test', policy='base', attention='full', split='test', count=512)
    base_ps = condition('base-ps-test', policy='base', attention='ps', split='test', count=512)
    policy = 'PS RL' if trained else 'base'
    if trained:
        selected_full = condition('selected-full-test', policy=policy, attention='full', split='test', count=512)
        selected_ps = condition('selected-ps-test', policy=policy, attention='ps', split='test', count=512)
    else:
        selected_full, selected_ps = base_full, base_ps
    for window in (256, 512, 1024, 2048, 4096):
        condition(f'selected-ps-window-{window}', policy=policy, attention='ps', split='test', count=128, window=window)
    condition('base-ps-larger', policy='base', attention='ps', split='larger', count=256)
    if trained:
        condition('selected-ps-larger', policy=policy, attention='ps', split='larger', count=256)
        condition('selected-ps-intervention-collection', policy=policy, attention='ps',
            split='intervention', count=512, include_report=False)
    report = dict(conditions=conditions,
        comparisons=[dict(treatment=selected_ps, sliding_control=base_ps, full_control=base_full)],
        checkpoint_selection=dict(path=str(selection_path),
            sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest()),
        training_runs=[dict(run_dir=str(run), training_seed=config.seed,
            validation_summaries=[row['summary'] for row in selected['candidates']])],
        full_rl_was_required=selected['full_context_matched_fraction'] < .8,
        interventions=[],
        interpretation='One PS training seed. Original checkpoint is the control; Full RL is unexecuted. '
            'Window sweeps and final tests cannot select a checkpoint. Interventions remain pending.')
    final = dict(status='PREPARED_NOT_EXECUTED', training_barrier=plan['training_barrier'],
        source_root=plan['source_root'], source_sha256=plan['source_sha256'],
        gpus=gpus, validation_gpus=validation_gpus, selected=selected,
        selection_receipt=str(selection_path),
        selection_receipt_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        commands=commands, selected_test=dict(full=selected_full, ps=selected_ps),
        report_manifest=str(output / 'report-manifest.json'),
        intervention_status='PENDING_COLLECTION_AND_VALIDITY_AUDIT' if trained else 'NO_TRAINED_CHECKPOINT_SELECTED',
        guard='Run only after all training engines have exited and this allocation holds the campaign lock. '
            'Verify policy-export file hashes before use. This command prepares files and launches no model.',
        scope='Complete fixed test, diagnostic window sweep, parameter-scaled board and separate intervention '
            'collection for the validation-selected policy. Original checkpoint remains eligible and is never labeled PS RL.')
    files[output / 'report-manifest.json'] = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    files[output / 'plan.json'] = (json.dumps(final, indent=2, allow_nan=False) + '\n').encode()
    if any(path.exists() and path.read_bytes() != data for path, data in files.items()):
        raise ValueError('Refusing to replace a different final evaluation plan or configuration')
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-plan', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    parser.add_argument('--validation-gpus', type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    plan = prepare_final_plan(args.validation_plan, args.run_dir, args.selection, args.output,
        gpus=args.gpus, validation_gpus=args.validation_gpus)
    print(json.dumps(dict(status=plan['status'], selected_update=plan['selected']['update'],
        commands=len(plan['commands']), intervention_status=plan['intervention_status'])))


if __name__ == '__main__':
    main()
