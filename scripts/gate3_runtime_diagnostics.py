"""Optional future-run instrumentation; importing this module starts no job."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class Progress:
    def __init__(self, directory, budget_seconds=1500, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.budget = budget_seconds
        self.path = Path(directory) / 'diagnostic_progress.jsonl'
        # Refuse to append to a previous session's evidence.
        with self.path.open('x', encoding='utf-8'):
            pass

    def event(self, phase, **fields):
        row = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=self.clock() - self.started, phase=phase, **fields)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        print('GATE3_PROGRESS', json.dumps(row, allow_nan=False), flush=True)
        return row

    def check_deadline(self):
        if self.clock() - self.started >= self.budget:
            self.event('DEADLINE_EXCEEDED', completed_epoch_not_implied=True)
            raise TimeoutError('Diagnostic time budget exhausted; no automatic retry')

    def run(self, args, timeout_seconds, **kwargs):
        self.check_deadline()
        remaining = self.budget - (self.clock() - self.started)
        self.event('EXTERNAL_OPERATION_START', executable=str(args[0]))
        try:
            result = subprocess.run(args, timeout=min(timeout_seconds, remaining), check=True, **kwargs)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            self.event('EXTERNAL_OPERATION_FAILED', error_type=type(error).__name__)
            raise
        self.event('EXTERNAL_OPERATION_COMPLETE', executable=str(args[0]))
        return result
