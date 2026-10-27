"""Finite-lifetime scheduling for recorded prefix-sliding reverse derivatives."""

from dataclasses import dataclass, field


@dataclass
class RecordedBlock:
    start: int
    end: int
    query_done: list[bool]
    kv_done: list[bool]
    payload: object = None


@dataclass
class RetirementSchedule:
    depth: int
    window: int
    prefix: int
    chunk: int
    policy_version: int
    blocks: list[RecordedBlock] = field(default_factory=list, init=False)
    frontier: list[int] = field(init=False)
    end: int = field(default=0, init=False)
    peak_live_token_layers: int = field(default=0, init=False)
    retired_tokens: int = field(default=0, init=False)
    closed: bool = field(default=False, init=False)

    def __post_init__(self):
        if min(self.depth, self.window, self.prefix, self.chunk) < 1:
            raise ValueError("Depth, window, prefix and chunk must be positive")
        self.frontier = [self.prefix - 1] * self.depth

    def append(self, end, *, policy_version, payload=None):
        if self.closed or policy_version != self.policy_version:
            raise ValueError("Closed or stale-policy retirement schedule")
        if end <= self.end or (self.end == 0 and end != self.prefix):
            raise ValueError("The first block must contain the complete prefix; blocks must advance")
        if self.end and end - self.end > self.chunk:
            raise ValueError("Recording block exceeds configured chunk size")
        block = RecordedBlock(self.end, end, [False] * self.depth, [False] * self.depth, payload)
        self.end = end
        self.blocks.append(block)
        live = sum((b.end - b.start) * sum(not done for done in b.kv_done) for b in self.blocks)
        self.peak_live_token_layers = max(self.peak_live_token_layers, live)
        return block

    @staticmethod
    def _groups(blocks, max_tokens):
        group = []
        for block in blocks:
            if group and (
                block.start != group[-1].end or group[0].start == 0 or block.end - group[0].start > max_tokens
            ):
                yield group
                group = []
            group.append(block)
        if group:
            yield group

    def transitions(self, *, policy_version, finished=False, max_tokens=1):
        """Yield dependency-ready work; commit it when the caller resumes iteration."""
        if self.closed or policy_version != self.policy_version:
            raise ValueError("Closed or stale-policy retirement schedule")
        for layer in reversed(range(self.depth)):
            ready = [
                block
                for block in self.blocks
                if (layer == self.depth - 1 or block.kv_done[layer + 1]) and not block.query_done[layer]
            ]
            for group in self._groups(ready, max_tokens):
                yield "query", group, layer
                for block in group:
                    block.query_done[layer] = True
                    if block.start >= self.prefix:
                        self.frontier[layer] = block.end - 1
            ready = []
            for block in self.blocks:
                deadline = block.end + self.window - 2
                if finished:
                    deadline = min(deadline, self.end - 1)
                complete = finished if block.start == 0 else self.frontier[layer] >= deadline
                if block.query_done[layer] and not block.kv_done[layer] and complete:
                    ready.append(block)
            for group in self._groups(ready, max_tokens):
                yield "kv", group, layer
                for block in group:
                    block.kv_done[layer] = True
                    yield "release_layer", [block], layer
        for block in self.blocks:
            if block.kv_done[0]:
                yield "embedding", [block], 0
                self.retired_tokens += block.end - block.start
        self.blocks = [block for block in self.blocks if not block.kv_done[0]]
        if finished:
            if self.blocks:
                raise RuntimeError("EOF did not close every recorded derivative")
            self.closed = True

    def advance(self, consume, *, policy_version, finished=False):
        for action, blocks, layer in self.transitions(policy_version=policy_version, finished=finished):
            for block in blocks:
                consume(action, block, layer)

    @property
    def activation_bound(self):
        rounded_halo = ((self.window - 1 + self.chunk - 1) // self.chunk) * self.chunk
        return self.prefix * self.depth + self.chunk * self.depth + rounded_halo * self.depth * (self.depth + 1) // 2
