"""Build, never execute, the separate Kaggle recovery protocol."""
import ast
import hashlib
import json
from pathlib import Path

from build_gate1_notebooks import ARTIFACT_CHECK, SETUP, cell

ROOT = Path(__file__).resolve().parents[1]

CONFIG = r'''
import subprocess
ENABLE_APPROVED_RECOVERY_TRAINING = False
MAX_NEW_EPOCHS = 5
# A future baseline-only execution must explicitly approve this isolated metric install.
INSTALL_LOCKED_BLEU_PACKAGE = True
'''

PREPARE = r'''
# Never install/upgrade Torch, encoders, or the root project requirements here.
require(platform.python_implementation() == 'CPython' and sys.version_info[:2] == (3, 13),
        'Reaudit dependencies for a different Python family')
require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Expected Kaggle Linux x86_64')
require(str(torch.__version__).startswith('2.11.') and torch.version.cuda == '12.8', 'Reaudit changed Torch/CUDA image')
require(versions['numpy'] == '2.1.3', 'Reaudit changed NumPy image')
require(shutil.which('java') is not None, 'Java is missing; stop without altering the environment')
java_result = subprocess.run(['java', '-version'], capture_output=True, text=True, check=True)
java_version = (java_result.stderr + java_result.stdout).strip()
metric_target = RUN_ROOT / 'metric_packages'
lock_path = RUN_ROOT / 'requirements-bleu.lock'
lock_path.write_text(METRIC_LOCK, encoding='utf-8')
require(INSTALL_LOCKED_BLEU_PACKAGE is True, 'Approve the exact isolated metric installation before executing')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(metric_target),
                '--no-deps', '--require-hashes', '--only-binary=:all:', '-r', str(lock_path)], check=True)
sys.path.insert(0, str(metric_target))
# pycocotools is deliberately excluded: this isolated BLEU/PTB path never imports it.
# This is not an installation for arbitrary COCO metrics.
for relative, digest in METRIC_IMPLEMENTATION.items():
    require(sha256(metric_target / relative) == digest, 'BLEU/PTB implementation differs: ' + relative)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu

def score_predictions(references, predictions):
    require(len(references) == len(predictions) > 0, 'Metric population mismatch')
    ptb = PTBTokenizer()
    refs = ptb.tokenize({i: [{'caption': text}] for i, text in enumerate(references)})
    preds = ptb.tokenize({i: [{'caption': text}] for i, text in enumerate(predictions)})
    score = 100.0 * float(Bleu(4).compute_score(refs, preds)[0][3])
    require(math.isfinite(score) and 0 <= score <= 100, 'Non-finite/out-of-range BLEU4')
    return score

toy = PTBTokenizer().tokenize({0: [{'caption': 'The lungs are clear . No pleural effusion .'}]})
require(toy[0] == ['the lungs are clear no pleural effusion'], 'PTB smoke test differs')
require(score_predictions(['the lungs are clear with no pleural effusion'],
                          ['the lungs are clear with no pleural effusion']) > 99.99, 'BLEU smoke test failed')
METRIC_FINGERPRINT = plain_metadata(dict(package='pycocoevalcap==1.2', wheel_sha256=METRIC_WHEEL_SHA256,
                                        implementation_sha256=METRIC_IMPLEMENTATION, java=java_version,
                                        tokenizer='Stanford PTB 3.4.1 lowercase punctuation removal',
                                        metric='corpus BLEU4 closest references x100'))
ENVIRONMENT = plain_metadata(dict(python=versions['python'], torch=str(torch.__version__),
                                  cuda=torch.version.cuda, numpy=versions['numpy'], gpu=gpu.name))
print('Locked BLEU/PTB smoke tests passed:', METRIC_FINGERPRINT)
'''

INITIALIZE = r'''
require(not ENABLE_APPROVED_RECOVERY_TRAINING or checkpoint.get('schema_version') == RECOVERY_SCHEMA,
        'Legacy90 initialization is baseline-only; training requires a retrieved and reviewed Gate3 checkpoint')
original_history = json.loads((ARTIFACT_DIR / 'text_training_history.json').read_text(encoding='utf-8'))
for name, digest in ORIGINAL_FILES.items():
    require(sha256(ARTIFACT_DIR / name) == digest, 'Historical source hash differs: ' + name)
historical_dir = RUN_ROOT / 'historical_evidence'
historical_dir.mkdir(exist_ok=False)
for name in ORIGINAL_FILES:
    destination = historical_dir / name
    shutil.copy2(ARTIFACT_DIR / name, destination)
    require(sha256(destination) == ORIGINAL_FILES[name], 'Historical copy integrity failure')
    destination.chmod(0o444)
for name, content in HISTORICAL_REPORTS.items():
    path = historical_dir / name
    path.write_bytes(content.encode('utf-8'))
    require(sha256(path) == HISTORICAL_REPORT_HASHES[name], 'Historical report copy differs')
    path.chmod(0o444)

validation_ids = [row['image_id'] for row in records['validation']]
validation_references = [row['report'] for row in records['validation']]
require(len(validation_ids) == 71, 'Validation population changed')
with torch.device('meta'):
    decoder = CheckpointReportDecoder()
validate_model_state(checkpoint, decoder)
decoder.load_state_dict(checkpoint['decoder'], strict=True, assign=True)

def write_json_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(plain_metadata(value), handle, indent=2, allow_nan=False)

def measure_validation(epoch, stage):
    decoder.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, 71, 6):
            rows = records['validation'][start:start + 6]
            output = decoder.generate(feature_batch(rows, 'validation'), max_length=160)
            predictions.extend(decode_report(ids, vocabulary) for ids in output.cpu())
    require(len(predictions) == 71, 'Not all validation reports were generated')
    score = score_predictions(validation_references, predictions)
    filename = f'validation_epoch_{epoch:03d}.json'
    measured = dict(epoch=epoch, split='validation', image_ids=validation_ids,
                    references=validation_references, predictions=predictions,
                    validation_bleu4=score, metric_fingerprint_sha256=json_sha256(METRIC_FINGERPRINT))
    write_json_exclusive(RUN_ROOT / filename, measured)
    return dict(epoch=epoch, stage=stage, validation_bleu4=score, predictions_file=filename,
                predictions_sha256=sha256(RUN_ROOT / filename),
                metric_fingerprint_sha256=json_sha256(METRIC_FINGERPRINT))

def verify_saved_scores(payload):
    for row in payload['recovery_history']:
        value = json.loads((RUN_ROOT / row['predictions_file']).read_text(encoding='utf-8'))
        require(value['references'] == validation_references, 'Saved references differ from validation manifest')
        rescored = score_predictions(value['references'], value['predictions'])
        require(math.isclose(rescored, row['validation_bleu4'], rel_tol=0, abs_tol=1e-10),
                'Saved BLEU4 does not reproduce from saved validation predictions')

if checkpoint.get('schema_version') == RECOVERY_SCHEMA:
    require(checkpoint['metric_fingerprint'] == METRIC_FINGERPRINT, 'Metric drift; no automatic recovery reset')
    require(checkpoint['provenance']['environment'] == ENVIRONMENT, 'Environment drift; reauditing required')
    optimizer, recovery_checks = restore_recovery(checkpoint, decoder, original_history,
        CANONICAL_DEPENDENCY_HASHES, CHECKPOINT_PATH.parent, validation_ids, DEVICE)
    # Copy only authoritative references; orphan candidates are never auto-selected.
    names = {checkpoint['recovery_best']['file']} | {r['predictions_file'] for r in checkpoint['recovery_history']}
    for name in names:
        shutil.copy2(CHECKPOINT_PATH.parent / safe_name(name), RUN_ROOT / name)
    metadata_state = metadata_from_payload(checkpoint)
    selected_reference = checkpoint['recovery_best']
    payload = checkpoint
    verify_saved_scores(payload)
else:
    require(sha256(CHECKPOINT_PATH) == ORIGINAL_FILES['text_decoder_last.pt'],
            'Legacy initialization must use the exact canonical epoch90 file')
    optimizer, recovery_checks = restore_weights_optimizer(checkpoint, decoder, DEVICE)
    require(recovery_checks['epoch'] == 90 and recovery_checks['epochs_without_improvement'] == 5,
            'Original stopping initialization differs')
    seed_legacy_recovery()  # New documented seed, not reconstruction of missing historical RNG.
    baseline = measure_validation(90, 'baseline')  # Fresh measured score; NEVER copy history[89].
    metadata_state = new_recovery_metadata(original_history, baseline, uuid.uuid4().hex,
        CANONICAL_DEPENDENCY_HASHES, METRIC_FINGERPRINT,
        dict(environment=ENVIRONMENT, legacy_rng_missing=True, restart_seed=42,
             source_sha256=ORIGINAL_FILES['text_decoder_last.pt'], runtime_sha256=RUNTIME_HASHES,
             paired_regime='0%', dataset='COV-CTR'))
    best_path = RUN_ROOT / 'recovery_best_epoch_090.pt'
    exclusive_save(recovery_best_payload(decoder, baseline, metadata_state['recovery_id'],
                                        CANONICAL_DEPENDENCY_HASHES), best_path)
    baseline_best = load_checkpoint(best_path)
    require(all(torch.equal(baseline_best['decoder'][k], checkpoint['decoder'][k])
                for k in checkpoint['decoder']), 'Baseline weights differ from physical epoch90 source')
    del baseline_best
    selected_reference = best_reference(best_path, baseline)
    payload = latest_recovery_payload(metadata_state, decoder, optimizer, selected_reference)
    verify_saved_scores(payload)
    print('Measured epoch90 recovery baseline:', baseline['validation_bleu4'])
    print('Original epoch90 recorded score (comparison only):', original_history[-1]['validation_bleu4'])
    print('Original selected85 score remains:', ORIGINAL_RECORD['best_validation_bleu4'])

validate_recovery_files(payload, decoder, original_history, CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids)
exclusive_save(payload, RUN_ROOT / 'recovery_latest.pt')
reloaded = load_checkpoint(RUN_ROOT / 'recovery_latest.pt')
validate_recovery_files(reloaded, decoder, original_history, CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids)
require(all(torch.equal(payload['decoder'][k].cpu(), reloaded['decoder'][k]) for k in payload['decoder']),
        'New schema save/reload model integrity failure')
require(all(torch.equal(payload['optimizer']['state'][k]['momentum_buffer'].cpu(),
                        reloaded['optimizer']['state'][k]['momentum_buffer'])
            for k in payload['optimizer']['state']), 'New schema save/reload momentum integrity failure')
optimizer, recovery_checks = restore_recovery(reloaded, decoder, original_history,
    CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids, DEVICE)
resume_rng = reloaded['rng_state']
del checkpoint, reloaded, payload
gc.collect()
print('Recovery state verified:', recovery_checks)
print('Historical stopping:', metadata_state['original_stop_tracking'])
write_json_exclusive(RUN_ROOT / 'recovery_readiness_report.json', dict(
    status='BASELINE_OR_RESTORATION_VERIFIED; NO_TRAINING_YET', epoch=metadata_state['epoch'],
    report_scope='immutable initialization snapshot; final completed epoch is in recovery_latest/history',
    checks=recovery_checks, historical_record=ORIGINAL_RECORD,
    original_stop_tracking=metadata_state['original_stop_tracking'], recovery_best=selected_reference,
    dataset_counts=dict(train=571, validation=71, test=72), test_scoring=False,
    environment=ENVIRONMENT, gpu_memory_before_load_bytes=dict(free=free_memory, total=total_memory),
    metric_fingerprint=METRIC_FINGERPRINT, durable_retrieval_verified=False))
'''

PERSIST = r'''
def persist_epoch():
    payload = latest_recovery_payload(metadata_state, decoder, optimizer, selected_reference)
    validate_recovery_files(payload, decoder, original_history, CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids)
    # Only this newly created run's latest is replaced atomically. Input/legacy artifacts never change.
    atomic_save(payload, RUN_ROOT / 'recovery_latest.pt')
    atomic_json_path = RUN_ROOT / 'recovery_history.json.tmp'
    atomic_json_path.write_text(json.dumps(metadata_state, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(atomic_json_path, RUN_ROOT / 'recovery_history.json')

def export_bundle():
    latest = load_checkpoint(RUN_ROOT / 'recovery_latest.pt')
    validate_recovery_files(latest, decoder, original_history, CANONICAL_DEPENDENCY_HASHES, RUN_ROOT, validation_ids)
    epoch = latest['epoch']
    del latest
    # Include historical evidence, all new-session winners, current winner, and all measurements.
    sources = [(p, 'historical_evidence/' + p.name) for p in historical_dir.iterdir()]
    sources += [(ARTIFACT_DIR / name, 'dependencies/' + name) for name in DEPENDENCIES]
    names = {'recovery_latest.pt', 'recovery_history.json', 'requirements-bleu.lock', 'configuration.json',
             'recovery_readiness_report.json'}
    names |= {p.name for p in RUN_ROOT.glob('recovery_best_epoch_*.pt')}
    names |= {r['predictions_file'] for r in metadata_state['recovery_history']}
    sources += [(RUN_ROOT / name, name) for name in sorted(names)]
    total_bytes = sum(p.stat().st_size for p, _ in sources)
    require(shutil.disk_usage(RUN_ROOT).free > total_bytes + 2**30,
            'Insufficient export disk space; latest is saved, stop and retrieve before another session')
    destination = RUN_ROOT / f'recovery_export_epoch_{epoch:03d}_{uuid.uuid4().hex[:8]}.zip'
    manifest = {relative: dict(sha256=sha256(path), bytes=path.stat().st_size) for path, relative in sources}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, relative in sources:
            archive.write(source, relative)
        archive.writestr('scripts/zeromrg_gate1_runtime.py', RUNTIME_SOURCE)
        archive.writestr('scripts/zeromrg_gate3_runtime.py', RECOVERY_RUNTIME_SOURCE)
        archive.writestr('sha256_manifest.json', json.dumps(dict(files=manifest, runtime_hashes=RUNTIME_HASHES), indent=2))
    write_json_exclusive(destination.with_suffix('.sha256.json'), dict(file=destination.name, sha256=sha256(destination),
        bytes=destination.stat().st_size, epoch=epoch, durable_retrieval_verified=False))
    print('Export prepared; durability UNVERIFIED until independently retrieved:', destination)

write_json_exclusive(RUN_ROOT / 'configuration.json', dict(architecture=ARCHITECTURE, training=TRAINING,
    recovery_policy=RECOVERY_POLICY, historical_record=ORIGINAL_RECORD, runtime_hashes=RUNTIME_HASHES,
    dependency_hashes=CANONICAL_DEPENDENCY_HASHES, metric_fingerprint=METRIC_FINGERPRINT,
    environment=ENVIRONMENT, approved_training=ENABLE_APPROVED_RECOVERY_TRAINING))
write_json_exclusive(RUN_ROOT / 'recovery_history.json', metadata_state)
export_bundle()  # Baseline/restart export BEFORE any optional training.
'''

TRAIN = r'''
def run_approved_training_chunk():
    global metadata_state, selected_reference
    require(ENABLE_APPROVED_RECOVERY_TRAINING is True, 'Training is disabled')
    require(type(MAX_NEW_EPOCHS) is int and 1 <= MAX_NEW_EPOCHS <= 5, 'Maximum five updates per approved session')
    start = metadata_state['epoch'] + 1
    require(start == recovery_checks['next_epoch'], 'Already advanced; reload an explicit checkpoint before another chunk')
    require(start <= 200 and metadata_state['original_stop_tracking']['epochs_without_improvement'] < 50,
            'Original stopping condition met; no silent restart')
    require(not globals().get('_gate3_training_attempt_started', False),
            'Training already attempted in this kernel; restart from a complete verified checkpoint')
    # Persist across re-execution of this cell, including exceptions before a completed epoch.
    # Never retry with partly updated in-memory model/optimizer state.
    globals()['_gate3_training_attempt_started'] = True
    restore_recovery_rng(resume_rng)  # Reapply after diagnostics/export.
    for epoch in range(start, min(200, start + MAX_NEW_EPOCHS - 1) + 1):
        decoder.train()
        generator = torch.Generator().manual_seed(42 + epoch)
        loader = torch.utils.data.DataLoader(records['train'], batch_size=6, shuffle=True,
            generator=generator, num_workers=0, collate_fn=lambda rows: rows)
        losses = []
        for rows in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = decoder_loss(feature_batch(rows, 'train'), token_batch(rows))
            require(bool(torch.isfinite(loss)), 'Non-finite decoder loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        measurement = measure_validation(epoch, 'continuation')
        updated, improved = append_recovery_epoch(metadata_state, epoch, sum(losses)/len(losses), measurement)
        if improved:
            candidate_path = RUN_ROOT / f'recovery_best_epoch_{epoch:03d}.pt'
            exclusive_save(recovery_best_payload(decoder, measurement, updated['recovery_id'],
                                                CANONICAL_DEPENDENCY_HASHES), candidate_path)
            selected_reference = best_reference(candidate_path, measurement)
        metadata_state = updated
        persist_epoch()  # EVERY completed epoch; model/optimizer/RNG/ledgers/current-winner hash.
        print('Recovery', epoch, 'validation reconstruction BLEU4', measurement['validation_bleu4'],
              'recoverable best', selected_reference, 'historical stopping', metadata_state['original_stop_tracking'])
        if metadata_state['original_stop_tracking']['epochs_without_improvement'] >= 50:
            break
    export_bundle()
    print('Chunk complete. Retrieve and verify export before another separately approved chunk.')

if ENABLE_APPROVED_RECOVERY_TRAINING:
    run_approved_training_chunk()
else:
    print('BASELINE/RESTORATION ONLY: no decoder updates; no images or held-out test scoring.')
'''


def build():
    runtime = (ROOT / 'scripts/zeromrg_gate1_runtime.py').read_text(encoding='utf-8')
    recovery = (ROOT / 'scripts/zeromrg_gate3_runtime.py').read_text(encoding='utf-8')
    lines = recovery.splitlines(keepends=True)
    imported = next(n for n in ast.parse(recovery).body if isinstance(n, ast.ImportFrom) and n.module == 'zeromrg_gate1_runtime')
    embedded_recovery = ''.join(lines[:imported.lineno-1] + lines[imported.end_lineno:])
    gate1 = json.loads((ROOT / 'notebooks/gate1/preflight/ZeroMRG_COV_CTR_Preflight_Only_Kaggle.ipynb').read_text())
    assignments = ast.parse(''.join(gate1['cells'][2]['source'])).body
    hashes = ast.literal_eval(next(n.value for n in assignments if isinstance(n, ast.Assign) and n.targets[0].id == 'CANONICAL_DEPENDENCY_HASHES'))
    metric = json.loads((ROOT / 'docs/evidence/gate3_metric_implementation.json').read_text())
    metric_lock = 'pycocoevalcap==1.2 --hash=sha256:' + metric['wheel']['digests']['sha256'] + '\n'
    reports = ['zeromrg_gate1_readiness.md', 'zeromrg_checkpoint_recovery.md', 'zeromrg_source_reconciliation.md',
               'zeromrg_checkpoint_persistence.md', 'zeromrg_gate2_preflight.md',
               'evidence/gate2_preflight_report.json', 'evidence/gate2_preflight_audit.json']
    report_contents = {Path(p).name: (ROOT / 'docs' / p).read_bytes().decode('utf-8') for p in reports}
    pins = '\n'.join(name + ' = ' + repr(value) for name, value in {
        'CANONICAL_DEPENDENCY_HASHES': hashes, 'RUNTIME_SOURCE': runtime, 'RECOVERY_RUNTIME_SOURCE': recovery,
        'RUNTIME_HASHES': {'gate1': hashlib.sha256(runtime.encode()).hexdigest(), 'gate3': hashlib.sha256(recovery.encode()).hexdigest()},
        'METRIC_LOCK': metric_lock, 'METRIC_WHEEL_SHA256': metric['wheel']['digests']['sha256'],
        'METRIC_IMPLEMENTATION': metric['implementation_sha256'], 'HISTORICAL_REPORTS': report_contents,
        'HISTORICAL_REPORT_HASHES': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in report_contents.items()}}.items())
    setup = SETUP.replace("ARTIFACT_DIR = CHECKPOINT_PATH.parent", "ARTIFACT_DIR = Path('/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip')")
    setup = setup.replace('zeromrg_gate1_', 'zeromrg_gate3_')
    checks = ARTIFACT_CHECK.replace("require(external_history == checkpoint['history'], 'External and checkpoint history disagree')",
        "require(external_history == checkpoint.get('original_history', checkpoint.get('history')), 'Original history differs')")
    intro = ('# ZeroMRG Gate 3: recovery baseline and separately approved continuation\n\n'
             'Default: measure the physical epoch90 decoder on the fixed71 validation reports, save a genuine recovery baseline, '
             'and export it. Training is disabled. Execution, metric installation and any later training require explicit approval. '
             'Never run locally. No image generation or held-out test scoring. Historical85/56.44423922571333 and patience5/50 remain '
             'independent and immutable. Recovery selection is a declared available-window deviation, not recovery of missing85 weights. '
             'A restart may select an explicit Gate3 recovery_latest.pt; original artifact dependencies remain attached unchanged.')
    cells = [cell(intro, 'markdown'), cell(runtime), cell(embedded_recovery), cell(pins), cell(CONFIG),
             cell(setup), cell(checks), cell(PREPARE), cell(INITIALIZE), cell(PERSIST), cell(TRAIN)]
    for index, value in enumerate(cells):
        value['id'] = f'gate3-recovery-{index:02d}'
    directory = ROOT / 'notebooks/gate3'
    directory.mkdir(parents=True, exist_ok=True)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')))
    (directory / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb').write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    metadata = json.loads((ROOT / 'notebooks/gate1/preflight/kernel-metadata.json').read_text())
    metadata.update(id='inshasadaf114/zeromrg-covctr-gate3-recovery', title='ZeroMRG COVCTR Gate3 Recovery Baseline',
                    code_file='ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb')
    (directory / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (directory / 'requirements-bleu.lock').write_text(metric_lock, encoding='utf-8')
    print('Built separate Gate3 notebook; did not execute or change Gate1 notebooks.')


if __name__ == '__main__':
    build()
