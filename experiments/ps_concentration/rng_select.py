"""Select a deferred PS checkpoint from complete, prespecified validation runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .overlap_config import RunConfig
from .rng_report import load_condition
from .hero_protocol import is_hero, training_identity


def read_json(path):
    return json.loads(Path(path).read_text())


def argument(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f'Expected exactly one {name} in the evaluation command')
    return argv[argv.index(name) + 1]


def select_checkpoint(plan_path, run, *, gpus):
    plan_path, run = Path(plan_path), Path(run)
    plan, state = read_json(plan_path), read_json(run / 'state.json')
    if state['policy_version'] < plan['training_barrier']:
        raise ValueError('Deferred checkpoint selection requires the training barrier')
    conditions = plan['layouts'][str(gpus)]
    expected_versions = {0, *plan['exports']['versions']}
    if max(expected_versions) != plan['training_barrier'] or min(expected_versions) < 0:
        raise ValueError('Candidate versions must end at the prespecified training barrier')
    candidates, controls, identities, sources = [], [], [], []
    for entry in conditions:
        argv = entry['argv']
        if argument(argv, '--split') != 'validation' or entry['count'] != 128:
            raise ValueError('Checkpoint selection uses only the fixed 128-board validation split')
        config_path = Path(argument(argv, '--config'))
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if digest != entry['config_sha256']:
            raise ValueError('Prespecified evaluation configuration changed')
        config = RunConfig(**read_json(config_path))
        identity = training_identity(run, state, config)
        if is_hero(state) and plan.get('training_identity') != identity:
            raise ValueError('Validation plan must pin the complete hero training identity')
        attention, version = entry['attention'], entry['policy_version']
        if attention not in ('ps', 'full') or attention != argument(argv, '--attention'):
            raise ValueError('Planned attention and evaluation command differ')
        checkpoint = Path(argument(argv, '--checkpoint')) if version else Path(config.model)
        if version:
            metadata = read_json(checkpoint / 'experiment.json')
            if metadata['policy_version'] != version or any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError('Exported checkpoint has a different policy or controller identity')
        policy_identity = str(checkpoint.resolve()) if version else config.revision
        summary_path = Path(entry['summary'])
        summary = read_json(summary_path)
        protocol = read_json(summary_path.parent / 'evaluation-protocol.json')
        actual_identity = RunConfig(**protocol['config']).scientific_identity()
        if actual_identity != config.scientific_identity() or summary.get('split') != 'validation':
            raise ValueError('Saved evaluation protocol differs from the planned validation run')
        condition = load_condition(dict(name=entry['condition'], policy='PS RL' if version else 'base',
            training_seed=config.seed if version else None, split='validation', attention=attention,
            window=config.window, greedy=True, expected_boards=128,
            episodes=str(summary_path.parent / 'episodes.jsonl'), summary=str(summary_path)))
        for row in [summary, *condition['rows']]:
            if row.get('policy_version') != version or row.get('policy_identity') != policy_identity:
                raise ValueError('Validation records belong to another checkpoint')
        for row in condition['rows']:
            if any(row.get(key) != value for key, value in (
                ('thinking_token_limit', config.thinking_token_limit),
                ('action_token_limit', config.action_token_limit),
                ('thinking_budget_transition', config.thinking_budget_transition),
            )):
                raise ValueError('Validation episodes used a different generation controller')
        for key in ('mean_return', 'matched_fraction', 'completion_rate', 'mean_attempts'):
            if not math.isfinite(condition[key]) or abs(condition[key] - summary[key]) > 1e-10:
                raise ValueError('Validation summary disagrees with its episode records')
        row = dict(update=version, checkpoint=str(checkpoint), summary=str(summary_path),
            validation_return=condition['mean_return'], matched_fraction=condition['matched_fraction'],
            board_seeds_sha256=condition['board_seeds_sha256'])
        (candidates if attention == 'ps' else controls).append(row)
        identities.append(identity)
        sources.extend(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (config_path, summary_path, summary_path.parent / 'episodes.jsonl',
                         summary_path.parent / 'evaluation-protocol.json'))
    if len(candidates) != len(expected_versions) or {row['update'] for row in candidates} != expected_versions:
        raise ValueError('Every prespecified PS checkpoint must have a complete validation result')
    if len(controls) != 1 or controls[0]['update'] != 0 or any(identity != identities[0] for identity in identities):
        raise ValueError('Selection requires one matching original full-context control')
    if len({row['board_seeds_sha256'] for row in [*candidates, *controls]}) != 1:
        raise ValueError('Validation board sets differ')
    selected = max(candidates, key=lambda row: (row['validation_return'], -row['update']))
    base_ps = next(row for row in candidates if row['update'] == 0)
    full_fraction = controls[0]['matched_fraction']
    forgetting_gap = full_fraction - base_ps['matched_fraction']
    return dict(**selected, selection='Highest complete PS validation efficiency return; earlier update on ties',
        split='validation', episodes=128, attention='ps', window=identities[0]['window'],
        training_seed=config.seed,
        training_barrier=plan['training_barrier'], committed_training_version=state['policy_version'],
        candidates=sorted(candidates, key=lambda row: row['update']), full_context_control=controls[0],
        competence_and_forgetting_status='GO' if full_fraction >= .8 and forgetting_gap >= .1 else 'HOLD',
        full_context_matched_fraction=full_fraction, full_minus_base_ps_matched_fraction=forgetting_gap,
        competence_threshold=.8, forgetting_threshold=.1, plan=str(plan_path),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), source_files=sources,
        selection_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Post-training validation selection for one training seed. Final test boards never enter selection; no training-variability or causal textual-memory claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--gpus', type=int, choices=(4, 8), required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = select_checkpoint(args.plan, args.run_dir, gpus=args.gpus)
    output = Path(args.output) if args.output else Path(args.run_dir) / 'selected-checkpoint.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({key: result[key] for key in ('update', 'checkpoint', 'validation_return', 'competence_and_forgetting_status')}))


if __name__ == '__main__':
    main()
