"""Finish the bounded PS dependency band with larger layer-wise VJPs."""

from copy import copy
from dataclasses import replace

import torch

from .retirement import RecordedBlock


class TailRecords:
    def __init__(self, stream):
        self.stream, self.records, self.complete = stream, {}, set()
        self.ready = None

    def prefetch(self, index, copying):
        with torch.cuda.stream(copying):
            for block in self.stream.schedule.blocks:
                if block.kv_done[index]:
                    continue
                key = index, block.start
                fields = None if not block.query_done[index] else ("input", "norm_input", "norm1", "qkv", "key")
                self.records[key] = self.stream._read_rec(
                    index,
                    block.start,
                    block.end,
                    fields,
                    copy_mode=self.stream.tail_prefetch,
                )
                if fields is None:
                    self.complete.add(key)
            self.ready = torch.cuda.Event()
            self.ready.record(copying)
        return self

    def consume(self):
        current = torch.cuda.current_stream()
        current.wait_event(self.ready)
        seen = set()
        for record in self.records.values():
            for value in record.values():
                pointer = value.untyped_storage().data_ptr()
                if pointer not in seen:
                    value.record_stream(current)
                    seen.add(pointer)

    def read(self, index, start, end, fields):
        pieces = []
        for block in self.stream.schedule.blocks:
            begin, stop = max(start, block.start), min(end, block.end)
            if begin >= stop:
                continue
            key = index, block.start
            record = self.records.setdefault(key, {})
            missing = (
                None
                if fields is None
                else tuple(field for field in fields if field not in record and key not in self.complete)
            )
            if (fields is None and key not in self.complete) or missing:
                if block.kv_done[index]:
                    raise RuntimeError("Tail attempted to load retired layer activations")
                record.update(self.stream._read_rec(index, block.start, block.end, missing))
                if fields is None:
                    self.complete.add(key)
            selected = record.keys() if fields is None else (name for name in fields if name in record)
            pieces.append({name: record[name][begin - block.start : stop - block.start] for name in selected})
        if not pieces or sum(next(iter(piece.values())).shape[0] for piece in pieces) != end - start:
            raise RuntimeError("Tail activation range crosses retired tokens")
        if len(pieces) == 1:
            return pieces[0]
        return {name: torch.cat([piece[name] for piece in pieces]) for name in pieces[0]}


def _groups(blocks, max_tokens):
    group = []
    for block in blocks:
        if group and (block.start != group[-1].end or group[-1].start == 0 or block.end - group[0].start > max_tokens):
            yield group
            group = []
        group.append(block)
    if group:
        yield group


def _joined(blocks, depth, layer, action):
    payload = dict(dx={}, dkv={}, dq={})
    slot = layer + 1 if action == "query" else layer
    payload["dx"][slot] = torch.cat([block.payload["dx"][slot] for block in blocks])
    if action == "kv":
        payload["dkv"][layer] = tuple(
            torch.cat([block.payload["dkv"][layer][i] for block in blocks], 2) for i in range(2)
        )
        if layer in blocks[0].payload.get("dq", {}):
            payload["dq"][layer] = torch.cat([block.payload["dq"][layer] for block in blocks], 2)
    return RecordedBlock(blocks[0].start, blocks[-1].end, [False] * depth, [False] * depth, payload)


def iter_finish_layerwise(stream, *, release=True):
    schedule = stream.schedule
    if schedule.closed:
        raise ValueError("Recorded tail is already closed")
    graphs = stream.graph_cache
    stream.graph_cache = None
    pending = None
    if stream.tail_prefetch is not None:
        copying = stream.model._ps_tail_copy_stream
        pending = TailRecords(stream).prefetch(stream.model.depth - 1, copying)
    try:
        for index in reversed(range(stream.model.depth)):
            stream._tail_records = pending if pending is not None else TailRecords(stream)
            if pending is not None:
                pending.consume()
                pending = TailRecords(stream).prefetch(index - 1, copying) if index else None
            queries = [block for block in schedule.blocks if not block.query_done[index]]
            for group in _groups(queries, stream.tail_block_tokens):
                if any(index + 1 < stream.model.depth and not block.kv_done[index + 1] for block in group):
                    raise RuntimeError("Tail query is missing an upper-layer derivative")
                joined = _joined(group, stream.model.depth, index, "query")
                stream._query(joined, index)
                offset = 0
                for block in group:
                    count = block.end - block.start
                    block.payload["dx"].pop(index + 1)
                    block.payload["dx"][index] = joined.payload["dx"][index][offset : offset + count]
                    if stream.joint_qkv:
                        block.payload["dq"][index] = joined.payload["dq"][index][:, :, offset : offset + count]
                    block.query_done[index] = True
                    offset += count
            schedule.frontier[index] = schedule.end - 1
            producers = [block for block in schedule.blocks if not block.kv_done[index]]
            for group in _groups(producers, stream.tail_block_tokens):
                joined = _joined(group, stream.model.depth, index, "kv")
                stream._kv(joined, index)
                offset = 0
                for block in group:
                    count = block.end - block.start
                    block.payload["dkv"].pop(index)
                    if stream.joint_qkv:
                        block.payload["dq"].pop(index)
                    block.payload["dx"][index] = joined.payload["dx"][index][offset : offset + count]
                    block.kv_done[index] = True
                    if release:
                        stream._consume("release_layer", block, index)
                    offset += count
            if stream.tail_prefetch is not None:
                # Bound allocations to this layer and one prefetch, including queued GPU work.
                consumed = torch.cuda.Event()
                consumed.record(torch.cuda.current_stream())
                yield consumed
            else:
                yield None
            stream._tail_records = None
        schedule.advance(stream._consume, policy_version=schedule.policy_version, finished=True)
    finally:
        stream.graph_cache = graphs
        stream._tail_records = None


def finish_layerwise(stream, *, release=True):
    for consumed in iter_finish_layerwise(stream, release=release):
        if consumed is not None:
            consumed.synchronize()


def warm_layerwise(stream, *, policy_version):
    """Compile tail shapes using copied adjoints; preserve the original warmup for graph capture."""
    preview = copy(stream)
    preview.schedule = copy(stream.schedule)
    preview.schedule.frontier = list(stream.schedule.frontier)
    preview.schedule.blocks = [
        replace(
            block,
            query_done=list(block.query_done),
            kv_done=list(block.kv_done),
            payload={
                **block.payload,
                "dx": {key: value.clone() for key, value in block.payload["dx"].items()},
                "dq": {key: value.clone() for key, value in block.payload.get("dq", {}).items()},
                "dkv": {key: tuple(value.clone() for value in values) for key, values in block.payload["dkv"].items()},
            },
        )
        for block in stream.schedule.blocks
    ]
    preview.gradients = {
        name: torch.zeros_like(p) if stream.direct_gradients else None for name, p in stream.parameters.items()
    }
    preview.kv_cache = None
    if stream.direct_gradients:
        preview.gradient_pointers = torch.tensor(
            [preview.gradients[name].data_ptr() for name in stream.parameters],
            dtype=torch.int64,
            device=stream.model.device,
        )
    with preview._active(policy_version):
        finish_layerwise(preview, release=False)
