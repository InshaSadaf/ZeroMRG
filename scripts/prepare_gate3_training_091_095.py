"""Prepare the single authorized recovery session; never execute or submit it."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from build_gate1_notebooks import cell
from build_gate3_notebook import TRAIN
from zeromrg_gate1_runtime import CheckpointReportDecoder, load_checkpoint, sha256
from zeromrg_gate3_runtime import validate_recovery_files
import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95'
SOURCE_TARGET = 'inshasadaf114/zeromrg-covctr-gate3-recovery-baseline'
FOLDER = ROOT / 'notebooks/gate3/training_091_095'

INPUT = r'''
# Exact existing baseline export, independently downloaded/verified before this run.
import stat
require(Path('/kaggle/input').is_dir(), 'Kaggle execution only')
candidates = list(Path('/kaggle/input').rglob(BASELINE_ZIP_NAME))
require(len(candidates) == 1, 'Expected exactly one attached baseline export with the approved filename')
BASELINE_INPUT_ZIP = candidates[0]
require(BASELINE_INPUT_ZIP.stat().st_size == BASELINE_ZIP_BYTES and
        sha256(BASELINE_INPUT_ZIP) == BASELINE_ZIP_SHA256, 'Attached recovery export differs from approved baseline')
INPUT_RECOVERY_DIR = Path('/kaggle/working') / ('verified_recovery_input_' + uuid.uuid4().hex[:12])
INPUT_RECOVERY_DIR.mkdir(exist_ok=False)
with zipfile.ZipFile(BASELINE_INPUT_ZIP) as bundle:
    entries = bundle.infolist()
    require(len(entries) == len(set(item.filename for item in entries)), 'Duplicate input archive paths')
    require(sum(item.file_size for item in entries) < 5 * 2**30, 'Input export exceeds audited size bound')
    require(shutil.disk_usage(INPUT_RECOVERY_DIR).free > sum(item.file_size for item in entries) + 2**30,
            'Insufficient space for safe input restoration')
    for item in entries:
        require((INPUT_RECOVERY_DIR / item.filename).resolve().is_relative_to(INPUT_RECOVERY_DIR.resolve()),
                'Unsafe input archive path')
        require(stat.S_IFMT(item.external_attr >> 16) != stat.S_IFLNK, 'Symlink input refused')
    require(bundle.testzip() is None, 'Input archive CRC failure')
    bundle.extractall(INPUT_RECOVERY_DIR)
input_manifest = json.loads((INPUT_RECOVERY_DIR / 'sha256_manifest.json').read_text())
for name, item in input_manifest['files'].items():
    require(sha256(INPUT_RECOVERY_DIR / name) == item['sha256'] and
            (INPUT_RECOVERY_DIR / name).stat().st_size == item['bytes'], 'Input manifest differs: ' + name)
require(sha256(INPUT_RECOVERY_DIR / 'recovery_latest.pt') == BASELINE_LATEST_SHA256 and
        sha256(INPUT_RECOVERY_DIR / 'recovery_best_epoch_090.pt') == BASELINE_BEST_SHA256,
        'Approved epoch90 latest/best weights are absent or changed')
print('Approved private baseline export attached and verified:', BASELINE_INPUT_ZIP)
'''

BOUNDARY = r'''
require(checkpoint.get('epoch') == 90 and checkpoint.get('schema_version') == RECOVERY_SCHEMA,
        'This authorization initializes only the reviewed full epoch90 recovery checkpoint')
require(checkpoint['original_stop_tracking'] == dict(best_epoch=85, best_validation_bleu4=56.44423922571333,
        epochs_without_improvement=5, patience=50), 'Historical starting patience/best differs')
require(checkpoint['recovery_best']['epoch'] == 90 and
        checkpoint['recovery_best']['validation_bleu4'] == 54.21140850351637 and
        checkpoint['recovery_best']['sha256'] == BASELINE_BEST_SHA256, 'Recovery baseline selection differs')
require(checkpoint['continuation_history'] == [], 'Starting checkpoint already contains training updates')
'''

EPOCH_RECORD = r'''
        torch.cuda.synchronize()
        completion = dict(epoch=epoch, completion_status='COMPLETED_AND_LATEST_SAVED',
            training_loss=metadata_state['continuation_history'][-1]['training_loss'],
            validation_bleu4=measurement['validation_bleu4'], validation_count=71,
            immutable_historical_record=ORIGINAL_RECORD,
            original_stop_tracking=metadata_state['original_stop_tracking'],
            recovery_best=dict(selected_reference), learning_rate=optimizer.param_groups[0]['lr'],
            momentum=optimizer.param_groups[0]['momentum'], optimizer_updates_this_epoch=len(losses),
            epoch_elapsed_seconds=time.monotonic()-epoch_started,
            latest_file='recovery_latest.pt', latest_sha256=sha256(RUN_ROOT / 'recovery_latest.pt'),
            latest_bytes=(RUN_ROOT / 'recovery_latest.pt').stat().st_size,
            predictions_sha256=measurement['predictions_sha256'])
        write_json_exclusive(RUN_ROOT / f'epoch_completion_{epoch:03d}.json', completion)
        print('EPOCH_COMPLETED', json.dumps(completion, sort_keys=True))
'''

FINAL_RECORD = r'''
    final_state = load_checkpoint(RUN_ROOT / 'recovery_latest.pt')
    validate_recovery_files(final_state, decoder, original_history, CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids)
    require(final_state['epoch'] <= AUTHORIZED_END_EPOCH, 'Authorization ceiling exceeded')
    verify_saved_scores(final_state)
    restored_optimizer, final_checks = restore_weights_optimizer(legacy_view(final_state), decoder, DEVICE)
    require(all(torch.equal(restored_optimizer.state[p]['momentum_buffer'], optimizer.state[p]['momentum_buffer'])
                for p in decoder.parameters()), 'Final optimizer restoration differs')
    del restored_optimizer
    completion_files = sorted(RUN_ROOT.glob('epoch_completion_*.json'))
    session_report = dict(status='TRAINING_CHUNK_COMPLETE_EXPORT_PENDING_RETRIEVAL',
        authorization=SESSION_AUTHORIZATION, start_epoch=AUTHORIZED_START_EPOCH,
        final_completed_epoch=final_state['epoch'], epoch_ceiling=AUTHORIZED_END_EPOCH,
        epoch_records=[json.loads(p.read_text()) for p in completion_files],
        original_historical_record=ORIGINAL_RECORD, original_stop_tracking=final_state['original_stop_tracking'],
        recovery_best=final_state['recovery_best'], environment=ENVIRONMENT,
        session_elapsed_seconds=time.monotonic()-session_started,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        full_state_restoration_checks=final_checks, no_image_inference=True, test_scoring=False,
        durable_retrieval_verified=False)
    write_json_exclusive(RUN_ROOT / 'training_session_report.json', session_report)
    del final_state
    export_bundle()
    print('AUTHORIZED_TRAINING_SESSION_COMPLETE', json.dumps(session_report, sort_keys=True))
'''


def main():
    verified = json.loads((ROOT / 'docs/evidence/gate3_baseline_verification.json').read_text())
    assert verified['status'] == 'PASS_VERIFIED_RECOVERY_BASELINE'
    assert verified['baseline_validation_bleu4'] == 54.21140850351637
    baseline_dir = ROOT / verified['verified_export_directory']
    baseline = load_checkpoint(baseline_dir / 'recovery_latest.pt')
    assert sha256(baseline_dir / 'recovery_latest.pt') == verified['latest_checkpoint']['sha256']
    assert sha256(baseline_dir / verified['best_checkpoint']['file']) == verified['best_checkpoint']['sha256']
    import csv
    with (baseline_dir / 'dependencies/validation_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        ids = [row['image_id'] for row in csv.DictReader(handle)]
    with torch.device('meta'):
        decoder = CheckpointReportDecoder()
    validate_recovery_files(baseline, decoder, baseline['original_history'], baseline['dependency_hashes'], baseline_dir, ids)
    del baseline, decoder
    submitted = json.loads((ROOT / 'artifacts/gate3/active_baseline_record.json').read_text())
    base = ROOT / 'notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb'
    assert sha256(base) == submitted['notebook_sha256']
    notebook = json.loads(base.read_text())
    FOLDER.mkdir(exist_ok=False)
    pins = dict(BASELINE_ZIP_NAME=verified['zip']['file'], BASELINE_ZIP_BYTES=verified['zip']['bytes'],
        BASELINE_ZIP_SHA256=verified['zip']['sha256'], BASELINE_LATEST_SHA256=verified['latest_checkpoint']['sha256'],
        BASELINE_BEST_SHA256=verified['best_checkpoint']['sha256'], AUTHORIZED_START_EPOCH=91,
        AUTHORIZED_END_EPOCH=95, SESSION_AUTHORIZATION='One private Kaggle recovery session, epochs91-95 inclusive only')
    config = ''.join(notebook['cells'][4]['source']).replace('ENABLE_APPROVED_RECOVERY_TRAINING = False',
                                                         'ENABLE_APPROVED_RECOVERY_TRAINING = True')
    config += '\n' + '\n'.join(k + ' = ' + repr(v) for k, v in pins.items()) + '\n'
    notebook['cells'][4] = cell(config)
    setup = ''.join(notebook['cells'][5]['source'])
    # INPUT needs these imports before the inherited SETUP cell imports them again.
    setup = 'import gc, platform, shutil, sys, tempfile, time, unicodedata, uuid, zipfile\n' + INPUT + setup
    setup = setup.replace("CHECKPOINT_PATH = Path('/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip/text_decoder_last.pt')",
                          "CHECKPOINT_PATH = INPUT_RECOVERY_DIR / 'recovery_latest.pt'")
    notebook['cells'][5] = cell(setup)
    notebook['cells'][8] = cell(BOUNDARY + ''.join(notebook['cells'][8]['source']))
    persist = ''.join(notebook['cells'][9]['source'])
    persist = persist.replace("sources += [(RUN_ROOT / name, name) for name in sorted(names)]",
        "names |= {p.name for p in RUN_ROOT.glob('epoch_completion_*.json')}\n"
        "    if (RUN_ROOT / 'training_session_report.json').exists():\n"
        "        names.add('training_session_report.json')\n"
        "    sources += [(RUN_ROOT / name, name) for name in sorted(names)]")
    persist = persist.replace('approved_training=ENABLE_APPROVED_RECOVERY_TRAINING)',
        'approved_training=ENABLE_APPROVED_RECOVERY_TRAINING, session_authorization=SESSION_AUTHORIZATION, '
        'authorized_start_epoch=AUTHORIZED_START_EPOCH, authorized_end_epoch=AUTHORIZED_END_EPOCH, '
        'baseline_input_zip_sha256=BASELINE_ZIP_SHA256, baseline_latest_sha256=BASELINE_LATEST_SHA256)')
    notebook['cells'][9] = cell(persist)
    training = TRAIN.replace("require(start == recovery_checks['next_epoch'],",
        "require(start == AUTHORIZED_START_EPOCH == 91 and AUTHORIZED_END_EPOCH == 95 and MAX_NEW_EPOCHS == 5, "
        "'This session is authorized only for epochs91-95')\n    require(start == recovery_checks['next_epoch'],")
    training = training.replace('min(200, start + MAX_NEW_EPOCHS - 1)',
                                'min(AUTHORIZED_END_EPOCH, 200, start + MAX_NEW_EPOCHS - 1)')
    training = training.replace('    restore_recovery_rng(resume_rng)',
        '    session_started = time.monotonic()\n    torch.cuda.reset_peak_memory_stats()\n'
        '    restore_recovery_rng(resume_rng)')
    training = training.replace('        decoder.train()',
        '        require(91 <= epoch <= 95, "Epoch outside authorized range")\n'
        '        epoch_started = time.monotonic()\n        decoder.train()')
    training = training.replace("        print('Recovery', epoch,", EPOCH_RECORD + "\n        print('Recovery', epoch,")
    training = training.replace("    export_bundle()\n    print('Chunk complete.",
        FINAL_RECORD + "\n    print('Chunk complete.")
    notebook['cells'][10] = cell(training)
    notebook['cells'][0] = cell('# ZeroMRG Gate3 controlled recovery: epochs91–95 only\n\n'
        'One explicitly authorized private session. Restores the independently verified physical90 baseline and full SGD/RNG. '
        'No image inference or test scoring. Historical85/56.44423922571333 and inherited stopping remain independent. '
        'Strict validation selection saves real recovery-window weights even below the missing85 score. '
        'Do not rerun or extend this notebook without separate authorization.', 'markdown')
    for i, value in enumerate(notebook['cells']):
        value['id'] = f'gate3-authorized-091-095-{i:02d}'
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), f'cell-{i}', 'exec')
            value.update(execution_count=None, outputs=[])
    filename = 'ZeroMRG_COV_CTR_Recovery_Epochs_091_095_Kaggle.ipynb'
    (FOLDER / filename).write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    metadata = json.loads((ROOT / 'notebooks/gate3/kernel-metadata.json').read_text())
    metadata.update(id=TARGET, title='ZeroMRG COVCTR Gate3 Recovery Epochs 91 95',
                    code_file=filename, kernel_sources=[SOURCE_TARGET], is_private=True)
    (FOLDER / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(ROOT / 'notebooks/gate3/requirements-bleu.lock', FOLDER / 'requirements-bleu.lock')
    record = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(), status='PREPARED_NOT_SUBMITTED',
        target=TARGET, folder=str(FOLDER.relative_to(ROOT)), start_epoch=91, max_epoch=95,
        private=True, datasets=metadata['dataset_sources'], kernel_sources=metadata['kernel_sources'],
        input_baseline=verified, notebook_sha256=sha256(FOLDER / filename),
        metadata_sha256=sha256(FOLDER / 'kernel-metadata.json'),
        baseline_notebook_unchanged=sha256(base) == submitted['notebook_sha256'],
        timeout_seconds=7200, authorization=SESSION_AUTHORIZATION if False else pins['SESSION_AUTHORIZATION'])
    path = ROOT / 'artifacts/gate3/training_091_095_preparation.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: record[k] for k in ('status', 'target', 'folder', 'notebook_sha256', 'metadata_sha256')}, indent=2))


if __name__ == '__main__':
    main()
