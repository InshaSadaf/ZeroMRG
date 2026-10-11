"""Record provenance for an already verified baseline; no network or model work."""
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified = read(ROOT / 'docs/evidence/gate3_baseline_verification.json')
    submitted = read(ROOT / 'artifacts/gate3/active_baseline_record.json')
    submission_dir = ROOT / submitted['directory']
    retrieval_dir = ROOT / verified['retrieval_directory']
    status_path = retrieval_dir / 'terminal_status.txt'
    status = status_path.read_text(encoding='utf-8-sig').strip()
    assert 'KernelWorkerStatus.COMPLETE' in status
    assert verified['status'] == 'PASS_VERIFIED_RECOVERY_BASELINE'
    assert submitted['actual_target'] == verified['target']
    assert submitted['server_private'] is True and submitted['server_source_matches'] is True
    notebook = submission_dir / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb'
    metadata = submission_dir / 'kernel-metadata.json'
    assert digest(notebook) == submitted['notebook_sha256']
    assert digest(metadata) == submitted['metadata_sha256']
    assert digest(ROOT / 'notebooks/gate3' / notebook.name) == submitted['notebook_sha256']
    assert digest(ROOT / 'notebooks/gate3/kernel-metadata.json') == submitted['metadata_sha256']
    server = submission_dir / 'server_source/zeromrg-covctr-gate3-recovery-baseline.ipynb'
    assert digest(server) == submitted['server_source_sha256']
    def source_cells(path):
        return [(cell['cell_type'], ''.join(cell['source'])) for cell in read(path)['cells']]
    assert len(source_cells(notebook)) == 11
    assert source_cells(notebook) == source_cells(server)
    server_metadata = read(submission_dir / 'server_source/kernel-metadata.json')
    assert server_metadata['is_private'] is True
    assert set(server_metadata['dataset_sources']) == set(submitted['datasets'])
    pins = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(''.join(read(notebook)['cells'][3]['source'])).body
    }
    for name, expected in pins['HISTORICAL_REPORT_HASHES'].items():
        assert digest(ROOT / 'docs' / ('evidence/' + name if name.endswith('.json') else name)) == expected, name
    snapshots = read(submission_dir / 'pre_execution_docs/sha256_manifest.json')
    # The preparation snapshot format is retained as originally created.
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    evidence = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS', scope='Existing baseline version1 verification only',
        actual_target=verified['target'], version=1, id_no=submitted['id_no'],
        terminal_status=status, terminal_status_sha256=digest(status_path),
        private=True, datasets=submitted['server_datasets'], all_11_cells_match=True,
        submitted_notebook_sha256=submitted['notebook_sha256'],
        prepared_metadata_sha256=submitted['metadata_sha256'],
        server_notebook_sha256=submitted['server_source_sha256'],
        notebook_and_metadata_unchanged_locally=True,
        historical_gate1_gate2_reports_unchanged=True,
        prepared_requested_target=submitted['target'],
        target_slug_warning_preserved=True,
        verification_sha256=digest(ROOT / 'docs/evidence/gate3_baseline_verification.json'),
        preparation_snapshot_manifest=snapshots,
        local_first_verification_failure='Result-path formatting: relative path versus absolute root. '
            'Corrected input resolution; preserved first extraction; second full verification passed.',
        local_provenance_check_correction='Compare cell type/source rather than Kaggle-reserialized cell metadata.',
        kaggle_runtime_failures=[],
        local_validation='CPU-only state/hash/schema comparison; helper syntax compilation and git diff --check',
        git_head=head, working_tree=tree.splitlines(), diff_check='PASS',
        preserved_user_modification='.gitignore (existing .kaggle/ ignore)',
        no_resubmission=True, no_additional_kaggle_execution=True,
        no_dataset_publication=True, no_training=True, no_git_commit_or_push_by_this_session=True,
        next_action='Explicit review/approval of a new private recovery attachment, then separate bounded epoch91 training authorization',
    )
    output = ROOT / 'docs/evidence/gate3_baseline_execution_audit.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(evidence, handle, indent=2)
        handle.write('\n')
    print('PASS: submission, privacy, source integrity, original reports, terminal status and working-tree review recorded.')


if __name__ == '__main__':
    main()
