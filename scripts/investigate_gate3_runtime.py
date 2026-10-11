"""One bounded, read-only incident snapshot. Never submits or cancels jobs."""
import ast
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    now = datetime.now(timezone.utc)
    folder = ROOT / 'artifacts/gate3' / ('runtime_incident_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(exist_ok=False)
    active = json.loads((ROOT / 'artifacts/gate3/active_training_091_095.json').read_text())
    result = dict(target=TARGET, observed_at_utc=now.isoformat(), requested_timeout_seconds=active['timeout_seconds'],
                  remote_requests=[], preserved_previous_directory=active['directory'])
    command = ['powershell.exe', '-NoProfile', '-File', str(ROOT / 'scripts/kaggle.ps1'),
               '-KaggleArgs', 'kernels', 'status', TARGET]
    # PowerShell's native argument array binding needs an explicit literal array.
    command = ['powershell.exe', '-NoProfile', '-Command',
               ".\\scripts\\kaggle.ps1 -KaggleArgs @('kernels','status','" + TARGET + "')"]
    try:
        cli = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
        (folder / 'cli_status.txt').write_text(cli.stdout + cli.stderr, encoding='utf-8')
        result['cli_status'] = dict(exit_code=cli.returncode, output=(cli.stdout + cli.stderr).strip())
    except subprocess.TimeoutExpired:
        result['cli_status'] = dict(error='Local 90-second CLI request deadline exceeded; remote state unknown')
    result['remote_requests'].append('project CLI kernels status (one attempt)')
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest()
        req.user_name, req.kernel_slug = TARGET.split('/')
        response = client.kernels.kernels_api_client.get_kernel(req)
        result['remote_requests'].append('get exact current version and source')
        server = response.metadata
        source_path = folder / 'server_notebook.ipynb'
        source_path.write_text(response.blob.source, encoding='utf-8')
        old_source = ROOT / active['directory'] / 'server_source/notebook.ipynb'
        old_nb = json.loads(old_source.read_text(encoding='utf-8'))
        nb = json.loads(response.blob.source)
        cells = lambda book: [(c['cell_type'], ''.join(c.get('source', []))) for c in book['cells']]
        last_run = server.last_run_time
        elapsed = None
        if last_run:
            elapsed = (datetime.now(timezone.utc) - last_run.replace(tzinfo=last_run.tzinfo or timezone.utc)).total_seconds()
        result['server'] = dict(id=server.id, version=server.current_version_number, private=server.is_private,
            enable_gpu=server.enable_gpu, datasets=server.dataset_data_sources, kernel_sources=server.kernel_data_sources,
            last_run_time=str(last_run), elapsed_since_last_run_timestamp_seconds=elapsed,
            elapsed_is_not_verified_gpu_execution_time=True, sha256=digest(source_path),
            original_server_sha256=digest(old_source), source_cells_identical=cells(nb) == cells(old_nb))
        assert server.current_version_number == 1 and server.is_private is True, 'Unexpected version or privacy'
        output_req = ApiListKernelSessionOutputRequest()
        output_req.user_name, output_req.kernel_slug = TARGET.split('/')
        outputs = client.kernels.kernels_api_client.list_kernel_session_output(output_req)
        result['remote_requests'].append('list existing outputs and execution log (one page)')
        result['outputs'] = dict(files=[f.file_name for f in outputs.files], log_characters=len(outputs.log or ''),
                                 additional_page_present=bool(outputs.next_page_token))
        if outputs.log:
            (folder / 'execution_log.json').write_text(outputs.log, encoding='utf-8')
        # No signed download URLs or authentication material are persisted.
    audit = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                label = ast.unparse(node.func)
                if any(term in label for term in ('subprocess.', 'PTBTokenizer', 'optimizer.step', 'DataLoader',
                                                   'export_bundle', 'generate', 'range', 'persist_epoch')):
                    calls.append(dict(line=node.lineno, call=ast.unparse(node)[:650]))
        audit.append(dict(cell=i, lines=len(source.splitlines()), relevant_calls=calls))
    (folder / 'source_static_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    baseline = json.loads((ROOT / 'docs/evidence/gate3_baseline_verification.json').read_text())
    export_dir = ROOT / baseline['verified_export_directory']
    original_dir = ROOT / 'artifacts/reconciliation/gate1_20261010_checkpoint_audit/epoch90'
    checks = []
    for name, expected in baseline['historical_record']['files'].items():
        path = original_dir / name
        checks.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                           sha256=digest(path), expected_sha256=expected))
    for key in ('latest_checkpoint', 'best_checkpoint'):
        item = baseline[key]
        path = export_dir / item['file']
        checks.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                           sha256=digest(path), expected_sha256=item['sha256']))
    zip_path = next((ROOT / baseline['retrieval_directory']).rglob(baseline['zip']['file']))
    checks.append(dict(path=str(zip_path.relative_to(ROOT)), bytes=zip_path.stat().st_size,
                       sha256=digest(zip_path), expected_sha256=baseline['zip']['sha256']))
    for item in checks:
        item['hash_matches_prior_verified_evidence'] = item['sha256'] == item['expected_sha256']
    result['preserved_checkpoint_checks'] = checks
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    (folder / 'investigation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(directory=str(folder.relative_to(ROOT)), **result), indent=2))


if __name__ == '__main__':
    main()
