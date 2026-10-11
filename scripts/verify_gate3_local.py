"""Run lightweight repository tests on CPU and retain the exact result."""
import json
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    started = time.monotonic()
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    output = result.stdout + result.stderr
    directory = ROOT / 'artifacts/gate3'
    directory.mkdir(exist_ok=True)
    log = directory / 'lightweight_test_log.txt'
    log.write_text(output, encoding='utf-8')
    match = re.search(r'Ran (\d+) tests', output)
    record = dict(status='PASS' if result.returncode == 0 else 'FAIL', exit_code=result.returncode,
                  tests=int(match[1]) if match else None, elapsed_seconds=time.monotonic()-started,
                  command=command[1:], scope='CPU fixtures/static checks only; CUDA_VISIBLE_DEVICES=-1',
                  gate3_recovery_tests=sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in
                      ast.walk(ast.parse((ROOT / 'tests/test_gate3_recovery.py').read_text(encoding='utf-8')))),
                  real_decoder_training=False, models_downloaded=False,
                  log=str(log.relative_to(ROOT)))
    (ROOT / 'docs/evidence/gate3_local_test_results.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    if result.returncode:
        print(output[-6000:])
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
