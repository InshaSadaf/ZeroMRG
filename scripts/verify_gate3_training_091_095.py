"""CPU-only independent verification of a retrieved controlled training export."""
import ast
import csv
import json
import os
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from zeromrg_gate1_runtime import ARCHITECTURE, TRAINING, CheckpointReportDecoder, load_checkpoint, require, sha256, restore_weights_optimizer
from zeromrg_gate3_runtime import ORIGINAL_FILES, ORIGINAL_RECORD, validate_recovery_files, historical_stop_record
from verify_gate3_baseline_export import model_digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(directory):
    torch.set_num_threads(1)
    directory = Path(directory).resolve()
    index = read(directory / 'retrieval_index.json')
    require(index['complete'] and index['export_present'] and 'COMPLETE' in index['terminal_status'],
            'Completed execution/export retrieval required; failure is not recovery-ready')
    archives = list(directory.rglob('recovery_export_epoch_*.zip'))
    require(len(archives) == 1, 'Expected one selected final export')
    archive = archives[0]
    sidecar = read(archive.with_suffix('.sha256.json'))
    require(sha256(archive) == sidecar['sha256'] and archive.stat().st_size == sidecar['bytes'], 'ZIP SHA/size differs')
    require(91 <= sidecar['epoch'] <= 95, 'Export epoch outside authorized completed training window')
    destination = directory / ('verified_export_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)) and sum(row.file_size for row in bundle.infolist()) < 5*2**30,
                'Duplicate paths or unbounded export')
        for row in bundle.infolist():
            require((destination / row.filename).resolve().is_relative_to(destination.resolve()), 'Unsafe export path')
            require(stat.S_IFMT(row.external_attr >> 16) != stat.S_IFLNK, 'Symlink export refused')
        require(bundle.testzip() is None, 'ZIP CRC failure')
        bundle.extractall(destination)
    manifest = read(destination / 'sha256_manifest.json')
    require(set(names) == set(manifest['files']) | {'scripts/zeromrg_gate1_runtime.py',
            'scripts/zeromrg_gate3_runtime.py', 'sha256_manifest.json'}, 'Manifest membership differs')
    for name, item in manifest['files'].items():
        require(sha256(destination / name) == item['sha256'] and (destination / name).stat().st_size == item['bytes'],
                'Manifest integrity failure: '+name)
    submitted = read(ROOT / 'artifacts/gate3/active_training_091_095.json')
    snapshot = ROOT / submitted['directory']
    meta = read(snapshot / 'kernel-metadata.json')
    notebook = read(snapshot / meta['code_file'])
    pins = {row.targets[0].id: ast.literal_eval(row.value) for row in ast.parse(''.join(notebook['cells'][3]['source'])).body}
    require(submitted['server_private'] and submitted['server_source_matches'] and submitted['version'] == 1,
            'Server source/privacy verification absent')
    require(manifest['runtime_hashes'] == pins['RUNTIME_HASHES'], 'Runtime pin drift')
    for scope, name in (('gate1', 'zeromrg_gate1_runtime.py'), ('gate3', 'zeromrg_gate3_runtime.py')):
        require(sha256(destination / 'scripts' / name) == pins['RUNTIME_HASHES'][scope], 'Archived runtime differs')
    for name, value in ORIGINAL_FILES.items():
        require(sha256(destination / 'historical_evidence' / name) == value, 'Original checkpoint/history altered')
    for name, value in pins['HISTORICAL_REPORT_HASHES'].items():
        require(sha256(destination / 'historical_evidence' / name) == value, 'Gate1/2 report altered')
    for name, value in pins['CANONICAL_DEPENDENCY_HASHES'].items():
        require(sha256(destination / 'dependencies' / name) == value, 'Dependency drift')
    configuration = read(destination / 'configuration.json')
    require(configuration['approved_training'] is True and configuration['authorized_start_epoch'] == 91 and
            configuration['authorized_end_epoch'] == 95 and configuration['architecture'] == ARCHITECTURE and
            configuration['training'] == TRAINING and configuration['baseline_latest_sha256'] ==
            submitted['input_baseline']['latest_checkpoint']['sha256'], 'Configuration/scope differs')
    latest_path = destination / 'recovery_latest.pt'
    latest = load_checkpoint(latest_path)
    with torch.device('meta'):
        decoder = CheckpointReportDecoder()
    with (destination / 'dependencies/validation_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        validation = list(csv.DictReader(handle))
    require(len(validation) == 71, 'Validation population differs')
    ids = [row['image_id'] for row in validation]
    references = [row['report'] for row in validation]
    original_history = read(destination / 'historical_evidence/text_training_history.json')
    inspection = validate_recovery_files(latest, decoder, original_history, pins['CANONICAL_DEPENDENCY_HASHES'], destination, ids)
    require(latest['epoch'] == sidecar['epoch'] <= 95 and latest['historical_record'] == ORIGINAL_RECORD and
            latest['original_history'] == original_history, 'Epoch/history differs')
    require(latest['metric_fingerprint'] == submitted['input_baseline']['metric_fingerprint'] ==
            configuration['metric_fingerprint'], 'Metric drift')
    require(latest['provenance']['environment'] == configuration['environment'] ==
            submitted['input_baseline']['environment'], 'Environment drift')
    history = read(destination / 'recovery_history.json')
    for name in history:
        require(history[name] == latest[name], 'Ancillary/latest history differs: '+name)
    best = load_checkpoint(destination / latest['recovery_best']['file'])
    winners = []
    running_score = -1.0
    for row in latest['recovery_history']:
        prediction = read(destination / row['predictions_file'])
        require(prediction['references'] == references and len(prediction['predictions']) == 71,
                'Validation reference/count mismatch')
        if row['validation_bleu4'] > running_score:
            winner_path = destination / f"recovery_best_epoch_{row['epoch']:03d}.pt"
            candidate = load_checkpoint(winner_path)
            require(candidate['epoch'] == row['epoch'] and candidate['validation_bleu4'] == row['validation_bleu4'] and
                    candidate['validation_predictions_sha256'] == row['predictions_sha256'] and
                    candidate['is_missing_epoch85_model'] is False and candidate['recovery_id'] == latest['recovery_id'] and
                    candidate['metric_fingerprint_sha256'] == row['metric_fingerprint_sha256'] and
                    candidate['dependency_hashes'] == latest['dependency_hashes'],
                    'Actual saved winner disagrees with validation/provenance')
            from zeromrg_gate1_runtime import validate_model_state
            validate_model_state(candidate, decoder)
            require(all(bool(torch.isfinite(tensor).all()) for tensor in candidate['decoder'].values()),
                    'Nonfinite selected weights')
            winners.append(dict(epoch=row['epoch'], validation_bleu4=row['validation_bleu4'],
                file=winner_path.name, bytes=winner_path.stat().st_size, sha256=sha256(winner_path)))
            running_score = row['validation_bleu4']
            del candidate
    baseline = submitted['input_baseline']
    require(winners[0]['epoch'] == 90 and winners[0]['sha256'] == baseline['best_checkpoint']['sha256'] and
            latest['recovery_history'][0]['validation_bleu4'] == 54.21140850351637, 'Starting selection changed')
    report = read(destination / 'training_session_report.json')
    require(report['final_completed_epoch'] == latest['epoch'] and report['epoch_ceiling'] == 95 and
            report['test_scoring'] is False and report['no_image_inference'] is True and
            report['recovery_best'] == latest['recovery_best'] and
            report['original_stop_tracking'] == latest['original_stop_tracking'], 'Final report/selection differs')
    epoch_rows = report['epoch_records']
    require([row['epoch'] for row in epoch_rows] == list(range(91, latest['epoch']+1)), 'Session completion records missing')
    for position, row in enumerate(epoch_rows):
        epoch = row['epoch']
        require(row == read(destination / f'epoch_completion_{epoch:03d}.json'), 'Completion record altered')
        measured = latest['continuation_history'][position]
        selected = max(latest['recovery_history'][:position+2], key=lambda item: item['validation_bleu4'])
        require(row['training_loss'] == measured['training_loss'] and row['validation_bleu4'] == measured['validation_bleu4'] and
            row['learning_rate'] == .001 and row['momentum'] == .99 and row['validation_count'] == 71 and
            row['optimizer_updates_this_epoch'] == 96 and row['immutable_historical_record'] == ORIGINAL_RECORD,
            'Per-epoch metric/config/update mismatch')
        require(row['original_stop_tracking'] == historical_stop_record(original_history, latest['continuation_history'][:position+1]) and
                row['recovery_best']['epoch'] == selected['epoch'] and row['recovery_best']['validation_bleu4'] == selected['validation_bleu4'],
                'Per-epoch stopping/selection mismatch')
        require(row['completion_status'] == 'COMPLETED_AND_LATEST_SAVED' and
                row['recovery_best']['sha256'] == sha256(destination / row['recovery_best']['file']) and
                row['predictions_sha256'] == latest['recovery_history'][position+1]['predictions_sha256'],
                'Per-epoch completion/checkpoint/prediction integrity mismatch')
    require(epoch_rows[-1]['latest_sha256'] == sha256(latest_path), 'Final latest differs from epoch completion SHA')
    require(latest['epoch'] == 95 or inspection['stop'], 'Early chunk termination without documented stopping')
    require(all(bool(torch.isfinite(value).all()) for value in latest['decoder'].values()), 'Nonfinite final decoder')
    decoder.load_state_dict(latest['decoder'], strict=True, assign=True)
    optimizer, restored = restore_weights_optimizer(__import__('zeromrg_gate3_runtime').legacy_view(latest), decoder, 'cpu')
    require(restored['momentum_buffers'] == 112, 'Incomplete actual CPU optimizer restoration')
    if best['epoch'] == latest['epoch']:
        require(all(torch.equal(best['decoder'][name], latest['decoder'][name]) for name in latest['decoder']),
                'Current winner/latest physical weights differ')
    old = load_checkpoint(ROOT / baseline['verified_export_directory'] / 'recovery_latest.pt')
    changed_weights = sum(not torch.equal(old['decoder'][name], latest['decoder'][name]) for name in latest['decoder'])
    changed_momentum = sum(not torch.equal(old['optimizer']['state'][key]['momentum_buffer'], state['momentum_buffer'])
                           for key, state in latest['optimizer']['state'].items())
    require(changed_weights > 0 and changed_momentum > 0, 'Training updates not evidenced in final state')
    require(latest['recovery_id'] == old['recovery_id'], 'Recovery identity changed')
    rng = latest['rng_state']
    require({'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= rng.keys() and
            len(rng['torch_cuda']) == 2 and len(rng['numpy']['keys']) == 624 and
            rng['torch_cpu'].dtype == torch.uint8 and rng['torch_cpu'].numel() > 0 and
            all(value.dtype == torch.uint8 and value.numel() > 0 for value in rng['torch_cuda']), 'Incomplete RNG export')
    del old
    log = read(directory / 'execution.log')
    log_text = '\n'.join(row['data'] for row in log)
    require('AUTHORIZED_TRAINING_SESSION_COMPLETE' in log_text and 'Traceback (most recent call last)' not in log_text,
            'Terminal success/log mismatch')
    result = dict(status='PASS_VERIFIED_CONTROLLED_TRAINING', target=index['target'], version=1,
        retrieval_directory=str(directory.relative_to(ROOT)), verified_export_directory=str(destination.relative_to(ROOT)),
        final_completed_epoch=latest['epoch'], next_epoch=latest['epoch']+1, epoch_records=epoch_rows,
        historical_record=latest['historical_record'], original_stop_tracking=latest['original_stop_tracking'],
        recovery_best=latest['recovery_best'], historical_best_exceeded=any(row['validation_bleu4'] > 56.44423922571333
            for row in latest['continuation_history']), actual_winner_files=winners,
        latest_checkpoint=dict(file=latest_path.name, epoch=latest['epoch'], bytes=latest_path.stat().st_size,
                               sha256=sha256(latest_path), tensor_content_sha256=model_digest(latest['decoder'])),
        zip=dict(file=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive), crc='PASS',
                 manifest_files_verified=len(manifest['files']), independently_retrieved=True),
        optimizer_restoration='PASS_CPU_FULL112_MOMENTUM', architecture=ARCHITECTURE,
        changed_decoder_tensors=changed_weights, changed_momentum_buffers=changed_momentum,
        rng_fields=sorted(rng), cuda_rng_states=len(rng['torch_cuda']), environment=report['environment'],
        session_elapsed_seconds=report['session_elapsed_seconds'], gpu_peak_allocated_bytes=report['gpu_peak_allocated_bytes'],
        gpu_peak_reserved_bytes=report['gpu_peak_reserved_bytes'], runtime_last_log_elapsed_seconds=max(row['time'] for row in log),
        log_sha256=sha256(directory / 'execution.log'), no_image_inference=True, no_test_scoring=True,
        local_validation='CPU state/schema/hash/optimizer restoration only; no forward/backward/step',
        score_recomputation='Executed on Kaggle from saved validation predictions; no local Java rerun',
        readiness='Verified next-epoch state; another session requires separate explicit authorization',
        remaining=['Original epoch85 weights unavailable', 'Historical encoder/RNG revisions unknown',
                   'No final zero-shot inference/test evaluation executed'])
    output = ROOT / 'docs/evidence/gate3_training_091_095_verification.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    (directory / 'training_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(destination / 'training_session_report.json', ROOT / 'docs/evidence/gate3_training_091_095_runtime_report.json')
    shutil.copy2(directory / 'execution.log', ROOT / 'docs/evidence/gate3_training_091_095_execution_log.json')
    print(json.dumps({key: result[key] for key in ('status', 'final_completed_epoch', 'original_stop_tracking',
        'recovery_best', 'latest_checkpoint', 'zip', 'session_elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
