"""Record unresolved existing-run verification. No network or model operations."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_gate3_training_091_095 import ROOT, TARGET
from zeromrg_gate1_runtime import require, sha256


def main():
    active = json.loads((ROOT / 'artifacts/gate3/active_training_091_095.json').read_text())
    directory = ROOT / active['directory']
    latest_status_path = max(directory.glob('status_*.json'))
    status = json.loads(latest_status_path.read_text())
    require('RUNNING' in status['status'] and status['listed_outputs'] == [],
            'State changed; inspect actual outputs instead of recording pending')
    require(active['submitted'] and active['version'] == 1 and active['server_private'] and active['server_source_matches'],
            'Submitted source/privacy verification missing')
    baseline = json.loads((ROOT / 'artifacts/gate3/active_baseline_record.json').read_text())
    require(sha256(ROOT / 'notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb') == baseline['notebook_sha256'] and
            sha256(ROOT / 'notebooks/gate3/kernel-metadata.json') == baseline['metadata_sha256'],
            'Original baseline source/metadata changed')
    git_status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    require(diff.returncode == 0, 'Diff check failed')
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        decision='BLOCKED_PENDING_RUNTIME_AND_PERSISTENCE_EVIDENCE', target=TARGET, version=1,
        id_no=active['id_no'], actual_status=status['status'], observation=status,
        observation_sha256=sha256(latest_status_path), source_private=True, source_matches=True,
        submitted_notebook_sha256=active['notebook_sha256'], metadata_sha256=active['metadata_sha256'],
        server_notebook_sha256=active['server_notebook_sha256'], datasets=active['datasets'],
        kernel_sources=active['kernel_sources'], submission_count_by_this_session=1,
        requested_execution_timeout_seconds=7200, execution_timeout_enforcement='UNVERIFIED',
        local_monitor_limit_seconds=7800,
        local_monitor_exit='RuntimeError: Status monitoring bound reached; job status unresolved. '
            'This is not a Kaggle notebook traceback or proof of GPU execution failure.',
        outputs_listed=0, runtime_logs_available=False, actual_training_completion='UNVERIFIED',
        final_completed_epoch=None, epochs_91_to_95_metrics=None, final_patience=None,
        final_recovery_best=None, historical_best_exceeded=None, new_checkpoint_hashes=None,
        persistence='UNVERIFIED: no new artifact returned; no next-session recovery readiness claimed',
        actual_gpu_type_and_runtime='UNVERIFIED for this session; baseline T4 environment is the required runtime signature',
        heldout_or_image_execution='Absent from audited submitted source; no runtime output exists to independently audit execution',
        baseline_notebook_metadata_unchanged=True,
        last_verified_recovery=active['input_baseline'], cpu_recovery_tests=14, guard_tests=4,
        no_additional_job=True, no_resubmission=True, no_cancellation=True,
        no_dataset_publication_or_mutation=True, no_git_push=True,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree=git_status.splitlines(), diff_check='PASS',
        evidence_directory=active['directory'],
        next_action='Read-only status/output retrieval of this existing version1 when execution ends; never resubmit it',
    )
    output = ROOT / 'docs/evidence/gate3_training_091_095_execution_audit.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('decision', 'target', 'version', 'actual_status', 'outputs_listed',
                                                  'final_completed_epoch', 'persistence')}, indent=2))


if __name__ == '__main__':
    main()
