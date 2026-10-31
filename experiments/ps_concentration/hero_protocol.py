"""Keep hero training identity distinct from the shared inference configuration."""

import json
from pathlib import Path

from .hero_v2 import HeroConfig, identity


THINKING_TRANSITION = '\n\nOkay, time to give the answer.</think>\n\n'


def is_hero(state):
    return state.get('identity', {}).get('variant') == 'hero-v2'


def inference_config(hero, *, output, gpus):
    from .overlap_config import RunConfig

    return RunConfig(model=hero.model, source=hero.source, output=str(output),
        source_sha256=hero.source_sha256, gpus=gpus, seed=hero.seed, updates=hero.updates,
        window=hero.window, transcript_cap=hero.transcript_cap, max_responses=hero.max_responses,
        thinking_token_limit=hero.thinking_token_limit, action_token_limit=hero.action_token_limit,
        thinking_budget_transition=THINKING_TRANSITION, temperature=hero.temperature,
        kl_coefficient=hero.kl_coefficient, wandb_id=hero.wandb_id, revision=hero.revision)


def training_identity(run, state, config):
    if not is_hero(state):
        expected = config.scientific_identity()
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('Validation and training scientific identities differ')
        return expected
    hero = HeroConfig(**json.loads((Path(run) / 'resolved_config.json').read_text()))
    expected = identity(hero)
    if state['identity'] != expected:
        raise ValueError('Hero committed state and resolved training identity differ')
    reference = inference_config(hero, output=config.output, gpus=config.gpus)
    fields = ('model', 'source', 'revision', 'seed', 'updates', 'window', 'transcript_cap',
        'max_responses', 'thinking_token_limit', 'action_token_limit', 'thinking_budget_transition',
        'temperature', 'kl_coefficient')
    if any(getattr(config, key) != getattr(reference, key) for key in fields):
        raise ValueError('Validation uses a different hero model or generation controller')
    return expected


def batch_size(state):
    size = state['identity']['global_batch_size'] if is_hero(state) else 64
    if type(size) is not int or size < 1:
        raise ValueError('Training batch size must be a positive integer')
    return size
