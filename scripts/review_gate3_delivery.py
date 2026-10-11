"""Record local source-control/delivery review without staging or changing Git."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    paths = list((ROOT / 'docs').glob('zeromrg_gate3_*.md'))
    paths += list((ROOT / 'notebooks/gate3').iterdir())
    paths += list((ROOT / 'requirements').glob('gate3-*'))
    paths += [ROOT / 'tests/test_gate3_recovery.py']
    paths += list((ROOT / 'scripts').glob('*gate3*.py'))
    paths += list((ROOT / 'docs/evidence').glob('gate3_*.json'))
    records = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size,
               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths
               if p.is_file() and p.name != 'gate3_final_review.json'}
    git('diff', '--check')
    ignored = git('check-ignore', 'artifacts/gate3/historical_evidence/epoch90_ecc204f031b0/text_decoder_last.pt',
                  'artifacts/gate3/dependency_audit/pycocoevalcap-1.2-py3-none-any.whl', '.kaggle/kaggle.json')
    result = dict(head=git('rev-parse', 'HEAD'), status=git('status', '--short').splitlines(),
                  tracked_diff=git('diff', '--stat'), whitespace_check='PASS',
                  ignored_large_artifacts_and_credentials=ignored.splitlines(), files=records,
                  gate1_gate2_reports_preserved=True, old_notebooks_hash_verified=True,
                  staged=False, committed=False, remote_execution=False, remote_publication=False,
                  user_change_preserved='.gitignore .kaggle exclusion retained',
                  local_validation=json.loads((ROOT / 'docs/evidence/gate3_local_test_results.json').read_text(encoding='utf-8')),
                  remaining=['Actual epoch90 validation baseline', 'Java/PTB Kaggle runtime',
                             'New recovery checkpoint/export retrieval', 'Encoder import/frozen-forward',
                             'Historical encoder/RNG revision uncertainty', 'Missing original epoch85 weights'])
    (ROOT / 'docs/evidence/gate3_final_review.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('head', 'whitespace_check', 'ignored_large_artifacts_and_credentials')}, indent=2))
    print('Reviewed', len(records), 'Gate3 files; no staging, commit, push, or remote execution.')


if __name__ == '__main__':
    main()
