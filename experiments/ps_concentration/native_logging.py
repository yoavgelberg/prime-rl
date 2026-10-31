"""Defer repeated native-history serialization while preserving final logs."""


def buffered_logger(base, state):
    class BufferedLogger(base):
        def _flush(self):
            state.logger = self
            self._pending = bool(self._data['turns']) and self._data['result'] is None
            if not self._pending:
                base._flush(self)

        def flush_pending(self):
            if self._pending:
                base._flush(self)
                self._pending = False

    return BufferedLogger
