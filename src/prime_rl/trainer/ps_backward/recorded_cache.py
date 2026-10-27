"""Bounded K/V reuse at each layer's advancing backward frontier."""


class KVWorkingSet:
    def __init__(self, depth):
        self.records = [{} for _ in range(depth)]
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def read(self, index, start, end, fields):
        if set(fields) != {"key", "value"}:
            return None
        for begin, (stop, record, _) in self.records[index].items():
            if begin <= start and end <= stop:
                self.hits += 1
                return {name: value[start - begin : end - begin] for name, value in record.items()}
        self.misses += 1
        return None

    def contains(self, index, start, end):
        return any(begin <= start and end <= stop for begin, (stop, _, _) in self.records[index].items())

    def retain(self, index, start, end, record, **_):
        self.release(index, start)
        saved = {name: record[name].detach().clone() for name in ("key", "value")}
        size = sum(value.numel() * value.element_size() for value in saved.values())
        self.records[index][start] = end, saved, size
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def evict_before(self, index, start):
        for begin, (end, _, _) in list(self.records[index].items()):
            if begin and end <= start:
                self.release(index, begin)

    def release(self, index, start):
        record = self.records[index].pop(start, None)
        if record is not None:
            self.bytes -= record[2]

    def clear(self):
        for records in self.records:
            records.clear()
        self.bytes = 0

    def report(self):
        return dict(
            live_bytes=self.bytes,
            peak_bytes=self.peak_bytes,
            hits=self.hits,
            misses=self.misses,
        )
