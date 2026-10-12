"""Standard-library stage evidence; no model imports, downloads or execution on import."""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc)


class StageProgress:
    def __init__(self, directory, clock=time.monotonic, utc_clock=utc_now):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.utc_clock = utc_clock
        self.started = clock()
        self.sequence = 0

    def emit(self, phase, state='event', **details):
        self.sequence += 1
        event = dict(phase=phase, state=state, sequence=self.sequence,
                     timestamp_utc=self.utc_clock().isoformat(),
                     elapsed_seconds=round(self.clock()-self.started, 6), **details)
        line = json.dumps(event, allow_nan=False)
        with (self.directory/'stage_progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(line+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary = self.directory/'latest_progress.json.tmp'
        temporary.write_text(line+'\n', encoding='utf-8', newline='\n')
        temporary.replace(self.directory/'latest_progress.json')
        print(line, flush=True)
        return event

    @contextmanager
    def stage(self, phase):
        start = self.clock()
        self.emit(phase, state='start')
        try:
            yield
        except BaseException as error:
            # Exception text can contain URLs or data; retain class only in progress.
            self.emit(phase, state='failed', duration_seconds=self.clock()-start,
                      error_type=type(error).__name__)
            raise
        else:
            self.emit(phase, state='complete', duration_seconds=self.clock()-start)


def run_stage(observer, phase, operation, *args, **kwargs):
    if observer is None:
        return operation(*args, **kwargs)
    with observer.stage(phase):
        return operation(*args, **kwargs)
