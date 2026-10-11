"""Cross-platform pip dry-run using an isolated verified pip wheel; never install."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/gate3/dependency_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=45) as response:
        metadata = json.load(response)
    wheel = next(f for f in metadata['urls'] if f['filename'].endswith('.whl'))
    path = OUT / wheel['filename']
    if not path.exists():
        with urllib.request.urlopen(wheel['url'], timeout=45) as response:
            path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == wheel['digests']['sha256']
    report = OUT / 'inference_candidate_resolver.json'
    command = [sys.executable, '-c',
               "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')",
               str(path), 'install', '--dry-run', '--ignore-installed', '--only-binary=:all:',
               '--python-version', '3.13', '--platform', 'manylinux_2_28_x86_64',
               '--platform', 'manylinux2014_x86_64', '--abi', 'cp313', '--abi', 'abi3',
               '--report', str(report), '-r', str(ROOT / 'requirements/gate3-inference-candidate.txt')]
    subprocess.run(command, check=True)
    result = json.loads(report.read_text(encoding='utf-8'))
    locked = []
    resolved = {}
    for item in result['install']:
        name, version = item['metadata']['name'], item['metadata']['version']
        digest = item['download_info']['archive_info']['hashes']['sha256']
        locked.append(f'{name}=={version} --hash=sha256:{digest}')
        resolved[name] = dict(version=version, sha256=digest, url=item['download_info']['url'])
    (ROOT / 'requirements/gate3-inference-candidate.lock').write_text('\n'.join(sorted(locked))+'\n', encoding='utf-8')
    evidence = dict(status='CROSS_TARGET_RESOLUTION_PASS; GPU/import runtime UNVERIFIED',
        installer=dict(version='25.3', wheel_sha256=wheel['digests']['sha256']),
        target='CPython3.13 Linux x86_64 manylinux2.28/2014', resolved=resolved,
        torch='Excluded: preserve Kaggle2.11+cu128', torchvision='Provided0.26 family must be verified on Kaggle',
        caveat='Cross-target resolver runs on Windows3.10; platform markers/import compatibility require Linux verification',
        install_performed=False, models_downloaded=False)
    (ROOT / 'docs/evidence/gate3_inference_resolution.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
