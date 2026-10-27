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

    def advance(self, consume, *, policy_version, finished=False):
        """Commit each transition only after its derivative consumer returns."""
        if self.closed or policy_version != self.policy_version:
            raise ValueError("Closed or stale-policy retirement schedule")
        for layer in reversed(range(self.depth)):
            for block in self.blocks:
                ready = layer == self.depth - 1 or block.kv_done[layer + 1]
                if ready and not block.query_done[layer]:
                    consume("query", block, layer)
                    block.query_done[layer] = True
                    if block.start >= self.prefix:
                        self.frontier[layer] = block.end - 1
            for block in self.blocks:
                prefix_block = block.start == 0
                deadline = block.end - 1 + self.window - 1
                if finished:
                    deadline = min(deadline, self.end - 1)
                complete = finished if prefix_block else self.frontier[layer] >= deadline
                if block.query_done[layer] and not block.kv_done[layer] and complete:
                    consume("kv", block, layer)
                    block.kv_done[layer] = True
                    consume("release_layer", block, layer)
        remaining = []
        for block in self.blocks:
            if block.kv_done[0]:
                consume("embedding", block, 0)
                self.retired_tokens += block.end - block.start
            else:
                remaining.append(block)
        self.blocks = remaining
        if finished:
            if self.blocks:
                raise RuntimeError("EOF did not close every recorded derivative")
            self.closed = True

    @property
    def activation_bound(self):
        rounded_halo = ((self.window - 1 + self.chunk - 1) // self.chunk) * self.chunk
        return self.prefix * self.depth + self.chunk * self.depth + rounded_halo * self.depth * (self.depth + 1) // 2
