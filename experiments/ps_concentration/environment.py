import re

import numpy as np
from popgym.envs.concentration import Concentration, ConcentrationHard

ACTION_LINE = re.compile(r"^FLIP ([+-]?\d+)[ \t]*$", re.MULTILINE)


def make_env(larger=False):
    return Concentration(num_decks=2, deck_type="ranks") if larger else ConcentrationHard()


def serialize(observation, hidden, step=0, limit=104, last_action=None, reward=0.):
    phase = "second" if step and reward == 0 else "first"
    action = "none" if last_action is None else str(last_action)
    visible = ",".join(f"{i}:{int(rank)}" for i, rank in enumerate(observation) if rank != hidden)
    return (f"step={step} limit={limit} next_pick={phase} last_action={action} reward={reward:.8f}\n"
            f"visible={{{visible}}}")


def deserialize(text, cards, hidden):
    entries = text.split("\nvisible={", 1)[1].removesuffix("}")
    obs = np.full(cards, hidden, dtype=np.int64)
    if entries:
        positions = []
        for item in entries.split(","):
            position, rank = map(int, item.split(":"))
            assert 0 <= position < cards and rank != hidden
            positions.append(position)
            obs[position] = rank
        assert positions == sorted(set(positions))
    return obs


def parse_action(text, cards=52, stop_condition='eos'):
    if stop_condition=='length' and not completed_action(text):
        return 0,True
    lines = ACTION_LINE.findall(text)
    action_lines = re.findall(r'^FLIP\b.*$', text, re.MULTILINE)
    valid = (len(lines) == len(action_lines) == 1 and text.rstrip().splitlines()[-1].strip() == f"FLIP {lines[0]}"
             and 0 <= int(lines[0]) < cards)
    return (int(lines[0]), False) if valid else (0, True)


def completed_action(text):
    """Only a newline closes a line before EOS; a partial integer may continue."""
    return re.search(r"(?:^|\n)FLIP [+-]?\d+[ \t]*\n", text) is not None


class ScriptedMemory:
    def __init__(self, random=False, seed=0):
        self.memory = {}
        self.matched = set()
        self.first = None
        self.random = random
        self.rng = np.random.default_rng(seed)

    def observe(self, observation, hidden, action=None, reward=0.):
        for p, r in enumerate(observation):
            if r != hidden:
                self.memory[p] = int(r)
        if action is None:
            return
        if reward > 0:
            self.matched.update((self.first, action))
        self.first = action if reward == 0 else None

    def act(self, cards):
        available = [p for p in range(cards) if p not in self.matched and p != self.first]
        if self.random:
            return int(self.rng.choice(available))
        if self.first is not None:
            rank = self.memory[self.first]
            partners = [p for p in available if self.memory.get(p) == rank]
            if partners:
                return min(partners)
        else:
            groups = {}
            for p, rank in self.memory.items():
                if p not in self.matched:
                    groups.setdefault(rank, []).append(p)
            for positions in groups.values():
                if len(positions) >= 2:
                    return min(positions)
        unseen = [p for p in available if p not in self.memory]
        return min(unseen or available)


def scripted_episode(seed, random=False):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    policy = ScriptedMemory(random=random, seed=seed)
    policy.observe(obs, env.facedown_card)
    total = 0.
    for step in range(1, env.episode_length + 1):
        action = policy.act(env.action_space.n)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.array_equal(deserialize(serialize(obs, env.facedown_card, step=step,
                                                  last_action=action, reward=reward),
                                          env.action_space.n, env.facedown_card), obs)
        policy.observe(obs, env.facedown_card, action, reward)
        total += reward
        if terminated or truncated:
            break
    return dict(seed=seed, native_return=total, matched_fraction=len(policy.matched)/env.action_space.n,
                completed=bool(terminated), flips=step)
