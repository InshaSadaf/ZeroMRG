"""CPU-only integrity/state verification of a retrieved baseline. No model forward/update."""
import csv
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

# Never initialize a local CUDA context while checking saved state.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from zeromrg_gate1_runtime import ARCHITECTURE, CheckpointReportDecoder, load_checkpoint, sha256, require
from zeromrg_gate3_runtime import ORIGINAL_FILES, ORIGINAL_RECORD, validate_recovery_state, json_sha256

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_digest(state):
    value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        value.update(b'\0')
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def main(directory):
    torch.set_num_threads(1)
    directory = Path(directory).resolve()
    retrieval = read_json(directory / 'retrieval_index.json')
    require(retrieval['complete'] and retrieval['export_present'], 'Retrieval incomplete or export absent')
    archives = list(directory.rglob('recovery_export_epoch_090_*.zip'))
    require(len(archives) == 1, 'Expected exactly one baseline export')
    archive = archives[0]
    sidecar = read_json(archive.with_suffix('.sha256.json'))
    archive_sha = sha256(archive)
    require(archive_sha == sidecar['sha256'] and archive.stat().st_size == sidecar['bytes'], 'Export ZIP hash/size mismatch')
    require(sidecar['epoch'] == 90 and sidecar['file'] == archive.name, 'Export identity mismatch')
    destination = directory / 'verified_export'
    if destination.exists():
        # Preserve any earlier verification attempt; never overwrite its extraction.
        destination = directory / ('verified_export_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP paths')
        require(sum(info.file_size for info in handle.infolist()) < 5 * 2**30, 'Unbounded export size')
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            require(target.is_relative_to(destination.resolve()), 'Unsafe archive path')
            require(stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK, 'Symlink export member refused')
        require(handle.testzip() is None, 'ZIP CRC failure')
        handle.extractall(destination)
    manifest = read_json(destination / 'sha256_manifest.json')
    expected_members = set(manifest['files']) | {'scripts/zeromrg_gate1_runtime.py',
        'scripts/zeromrg_gate3_runtime.py', 'sha256_manifest.json'}
    require(set(names) == expected_members, 'ZIP and manifest members differ')
    for filename, data in manifest['files'].items():
        path = destination / filename
        require(path.stat().st_size == data['bytes'] and sha256(path) == data['sha256'], 'Manifest integrity failure: '+filename)
    for key, filename in {'gate1': 'zeromrg_gate1_runtime.py', 'gate3': 'zeromrg_gate3_runtime.py'}.items():
        require(sha256(destination / 'scripts' / filename) == manifest['runtime_hashes'][key], 'Archived runtime source mismatch')
        local_source = (ROOT / 'scripts' / filename).read_text(encoding='utf-8').encode()
        require(hashlib.sha256(local_source).hexdigest() == manifest['runtime_hashes'][key], 'Runtime differs from audited source')
    for filename, digest in ORIGINAL_FILES.items():
        require(sha256(destination / 'historical_evidence' / filename) == digest, 'Original evidence changed')
    dependency_hashes = {name: data['sha256'] for name, data in
        ((Path(name).name, data) for name, data in manifest['files'].items() if name.startswith('dependencies/'))}
    submission = read_json(ROOT / 'artifacts/gate3/active_baseline_record.json')
    submitted_notebook = read_json(ROOT / submission['directory'] / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb')
    pin_tree = ast.parse(''.join(submitted_notebook['cells'][3]['source']))
    pins = {node.targets[0].id: ast.literal_eval(node.value) for node in pin_tree.body}
    require(dependency_hashes == pins['CANONICAL_DEPENDENCY_HASHES'], 'Dependencies differ from submitted canonical pins')
    require(manifest['runtime_hashes'] == pins['RUNTIME_HASHES'], 'Runtime provenance differs from submitted source')
    for filename, digest in pins['HISTORICAL_REPORT_HASHES'].items():
        require(sha256(destination / 'historical_evidence' / filename) == digest, 'Historical report differs from submitted evidence')
    original = load_checkpoint(destination / 'historical_evidence/text_decoder_last.pt')
    history = read_json(destination / 'historical_evidence/text_training_history.json')
    latest_path = destination / 'recovery_latest.pt'
    best_path = destination / 'recovery_best_epoch_090.pt'
    latest, best = load_checkpoint(latest_path), load_checkpoint(best_path)
    with torch.device('meta'):
        model = CheckpointReportDecoder()
    inspection = validate_recovery_state(latest, model, history, dependency_hashes, best)
    require(latest['epoch'] == best['epoch'] == original['epoch'] == 90, 'Wrong checkpoint epoch')
    require(latest['continuation_history'] == [] and len(latest['recovery_history']) == 1, 'Unexpected training epochs')
    require(latest['historical_record'] == ORIGINAL_RECORD and latest['original_history'] == original['history'] == history,
            'Historical selection/history changed')
    require(latest['original_stop_tracking'] == dict(best_epoch=85, best_validation_bleu4=56.44423922571333,
        epochs_without_improvement=5, patience=50), 'Historical best/patience changed')
    require(set(best['decoder']) == set(latest['decoder']) == set(original['decoder']), 'Decoder keys changed')
    for name in original['decoder']:
        require(torch.equal(original['decoder'][name], latest['decoder'][name]) and
                torch.equal(original['decoder'][name], best['decoder'][name]), 'Original90 weights changed: '+name)
    original_opt, new_opt = original['optimizer'], latest['optimizer']
    require(set(original_opt['state']) == set(new_opt['state']), 'SGD state mapping changed')
    require(original_opt['param_groups'][0]['params'] == new_opt['param_groups'][0]['params'], 'SGD parameter order changed')
    for parameter_id, state in original_opt['state'].items():
        require(torch.equal(state['momentum_buffer'], new_opt['state'][parameter_id]['momentum_buffer']),
                'SGD momentum changed during baseline')
    selected = latest['recovery_best']
    require(selected['file'] == best_path.name and selected['sha256'] == sha256(best_path), 'Winner file hash mismatch')
    measurement = latest['recovery_history'][0]
    prediction_path = destination / measurement['predictions_file']
    require(sha256(prediction_path) == measurement['predictions_sha256'] == selected['predictions_sha256'], 'Prediction hash mismatch')
    prediction = read_json(prediction_path)
    require(prediction['epoch'] == 90 and measurement['stage'] == 'baseline', 'Wrong measurement epoch/stage')
    with (destination / 'dependencies/validation_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        validation = list(csv.DictReader(handle))
    require(len(validation) == len(prediction['image_ids']) == len(prediction['references']) == len(prediction['predictions']) == 71,
            'Validation population differs')
    require(prediction['image_ids'] == [row['image_id'] for row in validation] and
            prediction['references'] == [row['report'] for row in validation] and prediction['split'] == 'validation',
            'Validation IDs/references/split differ')
    require(measurement['metric_fingerprint_sha256'] == prediction['metric_fingerprint_sha256'] ==
            json_sha256(latest['metric_fingerprint']), 'Metric fingerprint mismatch')
    report = read_json(destination / 'recovery_readiness_report.json')
    configuration = read_json(destination / 'configuration.json')
    score = prediction['validation_bleu4']
    require(score == best['validation_bleu4'] == measurement['validation_bleu4'] == selected['validation_bleu4'] ==
            report['checks']['recovery_best_validation_bleu4'], 'Baseline scores differ across evidence')
    require(configuration['approved_training'] is False and report['test_scoring'] is False, 'Wrong execution scope')
    require(configuration['architecture'] == ARCHITECTURE and configuration['dependency_hashes'] == dependency_hashes,
            'Configuration/dependencies differ')
    require(configuration['metric_fingerprint'] == latest['metric_fingerprint'] == report['metric_fingerprint'],
            'Metric configuration differs')
    require(configuration['training'] == latest['training_config'] and
            configuration['recovery_policy'] == latest['recovery_policy'] and
            configuration['historical_record'] == ORIGINAL_RECORD, 'Training/policy/historical configuration differs')
    require(latest['metric_fingerprint']['implementation_sha256'] == pins['METRIC_IMPLEMENTATION'] and
            latest['metric_fingerprint']['wheel_sha256'] == pins['METRIC_WHEEL_SHA256'], 'Metric implementation pin differs')
    rng = latest['rng_state']
    require({'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= rng.keys() and len(rng['torch_cuda']) > 0,
            'Complete RNG export absent')
    require(isinstance(rng['torch_cpu'], torch.Tensor) and rng['torch_cpu'].dtype == torch.uint8 and
            rng['torch_cpu'].numel() > 0 and
            all(t.dtype == torch.uint8 and t.numel() > 0 for t in rng['torch_cuda']), 'Invalid Torch RNG state')
    require(isinstance(rng['python'], tuple) and len(rng['python']) == 3 and
            isinstance(rng['python'][1], tuple), 'Invalid Python RNG export')
    require({'name', 'keys', 'position', 'has_gauss', 'cached'} <= rng['numpy'].keys() and
            rng['numpy']['name'] == 'MT19937' and len(rng['numpy']['keys']) == 624 and
            0 <= rng['numpy']['position'] <= 624, 'Incomplete NumPy RNG export')
    log_path = directory / 'execution.log'
    log = read_json(log_path)
    log_text = '\n'.join(row['data'] for row in log)
    require('BASELINE/RESTORATION ONLY: no decoder updates; no images or held-out test scoring.' in log_text,
            'Final baseline-only branch not evidenced')
    require('Traceback (most recent call last)' not in log_text and 'PapermillExecutionError' not in log_text,
            'Runtime exception in complete run')
    result = dict(status='PASS_VERIFIED_RECOVERY_BASELINE', target=retrieval['target'], version=1,
        retrieval_directory=str(directory.relative_to(ROOT)), verified_export_directory=str(destination.relative_to(ROOT)),
        checkpoint_epoch=90, next_epoch=91, validation_count=71, baseline_validation_bleu4=score,
        historical_epoch90_validation_bleu4=history[-1]['validation_bleu4'],
        baseline_minus_historical_epoch90=score-history[-1]['validation_bleu4'],
        historical_record=latest['historical_record'], original_stop_tracking=latest['original_stop_tracking'],
        decoder_tensor_count=len(original['decoder']), parameter_count=inspection['parameter_count'],
        decoder_weights_equal_original90=True, decoder_tensor_content_sha256=model_digest(original['decoder']),
        optimizer_momentum_buffers=len(original_opt['state']), optimizer_momentum_equal_original90=True,
        optimizer_settings={key: new_opt['param_groups'][0][key] for key in ('lr', 'momentum', 'dampening', 'weight_decay', 'nesterov')},
        scheduler=latest['scheduler'], rng_fields=sorted(rng), cuda_rng_state_count=len(rng['torch_cuda']),
        saved_history_lengths=dict(original=len(history), continuation=0, recovery=1),
        best_checkpoint=dict(file=best_path.name, bytes=best_path.stat().st_size, sha256=sha256(best_path),
                             epoch=best['epoch'], validation_bleu4=best['validation_bleu4'],
                             selection_scope=best['selection_scope'], is_missing_epoch85_model=best['is_missing_epoch85_model']),
        latest_checkpoint=dict(file=latest_path.name, bytes=latest_path.stat().st_size, sha256=sha256(latest_path)),
        zip=dict(file=archive.name, bytes=archive.stat().st_size, sha256=archive_sha, crc='PASS',
                 manifest_members_verified=len(manifest['files']), independently_retrieved=True),
        predictions_sha256=sha256(prediction_path), log_sha256=sha256(log_path),
        runtime_last_elapsed_seconds=max(row['time'] for row in log), environment=report['environment'],
        gpu_memory_before_load_bytes=report['gpu_memory_before_load_bytes'], metric_fingerprint=latest['metric_fingerprint'],
        no_training_or_optimizer_updates=True, no_image_inference=True, no_test_scoring=True,
        local_verification='CPU tensor/hash/schema comparisons only; no model forward or optimizer update',
        score_recomputation='Performed on Kaggle from saved predictions; local Java/tokenizer rerun not performed',
        remaining=['Original epoch85 weights unavailable', 'Historical encoder/RNG revisions unknown',
                   'No new private recovery dataset publication authorized', 'Epoch91 training requires separate approval'])
    (directory / 'baseline_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'docs/evidence/gate3_baseline_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    shutil_copy = __import__('shutil').copy2
    shutil_copy(destination / 'recovery_readiness_report.json', ROOT / 'docs/evidence/gate3_baseline_runtime_report.json')
    shutil_copy(log_path, ROOT / 'docs/evidence/gate3_baseline_execution_log.json')
    print(json.dumps({key: result[key] for key in ('status', 'baseline_validation_bleu4', 'best_checkpoint',
        'latest_checkpoint', 'zip', 'runtime_last_elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
