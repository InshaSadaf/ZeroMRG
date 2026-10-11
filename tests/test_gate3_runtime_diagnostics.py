"""CPU-only tests: diagnostic logging/deadlines, no torch or Kaggle requests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gate3_runtime_diagnostics import Progress


class DiagnosticTests(unittest.TestCase):
    def test_deadline_records_failure_without_implying_epoch_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            now = [0.0]
            progress = Progress(folder, budget_seconds=1, clock=lambda: now[0])
            now[0] = 1.0
            with self.assertRaises(TimeoutError):
                progress.check_deadline()
            row = json.loads(progress.path.read_text().splitlines()[-1])
            self.assertEqual(row['phase'], 'DEADLINE_EXCEEDED')
            self.assertTrue(row['completed_epoch_not_implied'])
            with self.assertRaises(FileExistsError):
                Progress(folder)

    def test_blocking_external_process_is_terminated_and_failure_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            progress = Progress(folder)
            with self.assertRaises(subprocess.TimeoutExpired):
                progress.run([sys.executable, '-c', 'import time; time.sleep(5)'], timeout_seconds=0.05)
            row = json.loads(progress.path.read_text().splitlines()[-1])
            self.assertEqual(row['error_type'], 'TimeoutExpired')


if __name__ == '__main__':
    unittest.main()
