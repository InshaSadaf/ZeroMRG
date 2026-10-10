"""Build self-contained Gate 1 notebooks from the audited runtime. Never execute them."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(text, kind="code"):
    value = dict(cell_type=kind, metadata={}, source=text.strip().splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


SETUP = r'''
import gc, platform, shutil, sys, tempfile, time, unicodedata, uuid, zipfile
from collections import Counter, defaultdict
from importlib import metadata

# Select an exact checkpoint; no automatic newest-file selection or epoch assumption.
CHECKPOINT_PATH = Path('/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip/text_decoder_last.pt')
ARTIFACT_DIR = CHECKPOINT_PATH.parent
BEST_CHECKPOINT_PATH = ARTIFACT_DIR / 'text_decoder_best.pt'
DATA_ROOT = Path('/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports')
RUN_ROOT = Path('/kaggle/working') / ('zeromrg_gate1_' + uuid.uuid4().hex[:12])
require(Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir(), 'Execute this notebook on Kaggle only')
require(torch.cuda.is_available(), 'Enable a Kaggle GPU; no local CPU fallback')
DEVICE = torch.device('cuda:0')
require(CHECKPOINT_PATH.is_file(), 'Set CHECKPOINT_PATH to the exact mounted latest checkpoint')
require(DATA_ROOT.is_dir(), 'Set DATA_ROOT to the attached COV-CTR directory')
require(RUN_ROOT.resolve().is_relative_to(Path('/kaggle/working').resolve()), 'Output must stay in /kaggle/working')
RUN_ROOT.mkdir(exist_ok=False)
write_probe = RUN_ROOT / 'write_probe.txt'
write_probe.write_text('writable', encoding='utf-8')
require(write_probe.read_text(encoding='utf-8') == 'writable', 'Output write/read failed')
print('Mounted roots:', [str(p) for p in Path('/kaggle/input').iterdir()])
gpu = torch.cuda.get_device_properties(0)
free_memory, total_memory = torch.cuda.mem_get_info()
versions = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda}
for distribution in ('numpy', 'pandas', 'Pillow', 'clip', 'multilingual-clip', 'transformers', 'pycocoevalcap'):
    try:
        versions[distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        versions[distribution] = 'not installed'
print('GPU:', gpu.name, '| free/total GiB:', free_memory/2**30, total_memory/2**30)
print('Versions:', versions)
print('Selected checkpoint:', CHECKPOINT_PATH, '| output:', RUN_ROOT)
'''

ARTIFACT_CHECK = r'''
# Legacy recovery must match the audited dependencies exactly. New checkpoints carry
# the same dependency hashes; changed splits/features are refused.
checkpoint = load_checkpoint(CHECKPOINT_PATH)
expected_dependencies = checkpoint.get('dependency_hashes', CANONICAL_DEPENDENCY_HASHES)
artifacts = validate_artifacts(ARTIFACT_DIR, expected_dependencies)
records = artifacts['records']
vocabulary = artifacts['vocabulary']['vocabulary']
text_cache = artifacts['text']
prompt_cache = artifacts['prompt']
external_history = json.loads((ARTIFACT_DIR / 'text_training_history.json').read_text(encoding='utf-8'))
require(external_history == checkpoint['history'], 'External and checkpoint history disagree')

# Verify mounted raw population and saved target reports without rebuilding splits.
csv_candidates = list(DATA_ROOT.rglob('reports_ZH_EN.csv'))
require(len(csv_candidates) == 1, 'Expected one attached COV-CTR CSV')
with csv_candidates[0].open(encoding='utf-8-sig', newline='') as handle:
    raw_rows = list(csv.DictReader(handle))
require(len(raw_rows) == 726, 'Expected 726 raw report rows')
images = [p for p in DATA_ROOT.rglob('*') if p.is_file() and p.suffix.lower() in
          {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}]
require(len(images) == 746, 'Expected 746 physical image files')
raw_ids = Counter(row['image_id'].strip() for row in raw_rows)
require(sum(count > 1 for count in raw_ids.values()) == 6, 'Duplicate-ID audit differs')
raw_by_id = {row['image_id'].strip(): row for row in raw_rows if raw_ids[row['image_id'].strip()] == 1}
image_index = defaultdict(list)
for p in images:
    image_index[p.name.lower()].append(p)
for row in records['validated']:
    key = row['image_id']
    normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', raw_by_id[key]['reports_En'])).strip()
    require(normalized == row['report'], f'Saved report differs from raw source: {key}')
    require(len(image_index[key.lower()]) == 1, f'Image missing/ambiguous: {key}')
print('Dataset/split counts: 726 rows, 746 images, 714 usable; 571/71/72 partitions')
print('Training-only vocabulary: 368; prompt bank: 500 x 768 raw; normalized text: 571 + 71')

def token_batch(rows):
    sequences = [encode_report(row['report'], vocabulary) for row in rows]
    ids = torch.full((len(rows), max(map(len, sequences))), 0, dtype=torch.long, device=DEVICE)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, device=DEVICE)
    return ids

def feature_batch(rows, name):
    return torch.stack([text_cache[name][row['image_id']] for row in rows]).to(DEVICE)

def decoder_loss(condition, ids):
    logits = decoder(condition, ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, 368), ids[:, 1:].reshape(-1),
                           ignore_index=0, reduction='sum') / ids[:, 1:].ne(0).sum()
'''

LATEST_DIAGNOSTIC = r'''
with torch.device('meta'):
    decoder = CheckpointReportDecoder()
latest_inspection = inspect_training_state(checkpoint, decoder)
decoder.load_state_dict(checkpoint['decoder'], strict=True, assign=True)
optimizer, latest_inspection = restore_weights_optimizer(checkpoint, decoder, DEVICE)
print('Latest model/optimizer:', latest_inspection)
print('Saved best epoch from history:', latest_inspection['best_epoch'], '(weights still require verification)')

# One TRAINING sample; no optimizer.step, backward, validation scoring, or test inference.
probe_rows = records['train'][:1]
decoder.eval()
with torch.no_grad():
    conditions = feature_batch(probe_rows, 'train')
    ids = token_batch(probe_rows)
    probe_logits = decoder(conditions, ids[:, :-1])
    probe_loss = decoder_loss(conditions, ids)
require(bool(torch.isfinite(probe_logits).all()) and bool(torch.isfinite(probe_loss)), 'Non-finite diagnostic output')
require(all(parameter.grad is None for parameter in decoder.parameters()), 'Unexpected diagnostic gradients')
print('Finite latest-checkpoint TRAIN forward CE:', float(probe_loss))

# Disposable full model + optimizer round trip, in an isolated temporary directory.
with tempfile.TemporaryDirectory(prefix='disposable_restore_', dir=RUN_ROOT) as temporary:
    diagnostic_path = Path(temporary) / 'diagnostic.pt'
    atomic_save(checkpoint, diagnostic_path)
    reloaded = load_checkpoint(diagnostic_path)
    inspect_training_state(reloaded, decoder)
    for key in checkpoint['decoder']:
        require(torch.equal(reloaded['decoder'][key], checkpoint['decoder'][key]), 'Diagnostic weights changed')
    for pid in checkpoint['optimizer']['state']:
        require(torch.equal(reloaded['optimizer']['state'][pid]['momentum_buffer'],
                            checkpoint['optimizer']['state'][pid]['momentum_buffer']), 'Diagnostic momentum changed')
    optimizer, roundtrip_inspection = restore_weights_optimizer(reloaded, decoder, DEVICE)
    with torch.no_grad():
        restored_logits = decoder(conditions, ids[:, :-1])
    require(torch.equal(restored_logits, probe_logits), 'Forward output changed after diagnostic restoration')
    saved_hash = sha256(diagnostic_path)
    del reloaded
print('Disposable save/restore PASS; SHA-256:', saved_hash)
gc.collect()
'''

PREFLIGHT_BEST = r'''
# Gate 0's epoch-85 history entry is NOT evidence of epoch-85 model weights.
# The audited backup actually contains an epoch-8 best artifact. Report this clearly.
preflight = dict(kind='preflight_only', checkpoint_sha256=sha256(CHECKPOINT_PATH),
                 latest=latest_inspection, dependency_assertions='PASS',
                 finite_train_forward_loss=float(probe_loss), disposable_restore='PASS',
                 versions=versions, gpu=gpu.name, zero_shot='BLOCKED', overall='BLOCKED')
best = load_checkpoint(BEST_CHECKPOINT_PATH)
try:
    validate_selected_best(checkpoint, decoder, best)
    preflight['best_selection'] = 'PASS'
except ValueError as error:
    preflight['best_selection'] = str(error)
    (RUN_ROOT / 'preflight_report.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
    print(json.dumps(preflight, indent=2))
    raise ValueError('PREFLIGHT BLOCKED: validation-selected best weights unavailable. Latest restoration checks passed.') from error

# Reached only with a verified best artifact. Never substitute epoch-90 weights here.
# If absent, install ONLY the official OpenAI CLIP package on Kaggle, then rerun:
# %pip install 'clip @ git+https://github.com/openai/CLIP.git'
import clip
from PIL import Image
decoder.load_state_dict(best['decoder'], strict=True)
decoder.eval()
for parameter in decoder.parameters():
    parameter.requires_grad_(False)
clip_model, clip_preprocess = clip.load('ViT-L/14', device=DEVICE, jit=False)
vision = clip_model.visual.eval()
for parameter in vision.parameters():
    parameter.requires_grad_(False)
del clip_model
require(all(not p.requires_grad for p in vision.parameters()), 'Image encoder must be frozen')
require(all(not p.requires_grad for p in decoder.parameters()), 'Decoder must be frozen for zero-shot')
require(int(vision.output_dim) == 768, 'Image embedding dimension differs')
probe_image_path = image_index[probe_rows[0]['image_id'].lower()][0]
with Image.open(probe_image_path) as image:
    image_tensor = clip_preprocess(image.convert('RGB')).unsqueeze(0)
require(tuple(image_tensor.shape) == (1, 3, 224, 224), 'CLIP image preprocessing shape differs')
prompts = prompt_cache['embeddings'].to(DEVICE, dtype=torch.float32)
require(not prompts.requires_grad, 'Prompt bank must be frozen')
with torch.no_grad():
    image_embedding = vision(image_tensor.to(DEVICE, dtype=vision.conv1.weight.dtype)).float()
    phi, attention = enrich(image_embedding, prompts)
    require(torch.allclose(attention.sum(-1), torch.ones(1, device=DEVICE), atol=1e-5), 'Attention normalization failed')
    require(torch.allclose(phi.norm(dim=-1), torch.ones(1, device=DEVICE), atol=1e-5), 'Enriched feature normalization failed')
    image_logits = decoder(phi, torch.tensor([[1]], device=DEVICE))
    require(bool(torch.isfinite(image_logits).all()), 'Image-conditioned logits non-finite')
    generated = decoder.generate(phi, max_length=160).cpu()
preflight.update(zero_shot='PASS: one training image, no scoring', overall='PASS',
                 best_checkpoint_sha256=sha256(BEST_CHECKPOINT_PATH),
                 clip_weight_url=clip.clip._MODELS['ViT-L/14'],
                 diagnostic_image_id=probe_rows[0]['image_id'],
                 diagnostic_generation=decode_report(generated[0], vocabulary),
                 generated_length=int(generated.shape[1]))
(RUN_ROOT / 'preflight_report.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
print(json.dumps(preflight, indent=2))
print('PREFLIGHT PASS; no training or held-out test evaluation was performed')
'''

CONTINUATION_RESTORE = r'''
# Full continuation is blocked if the saved best weights do not match selection history.
best = load_checkpoint(BEST_CHECKPOINT_PATH)
optimizer, recovery = restore_training(checkpoint, decoder, best, DEVICE)
history = list(checkpoint['history'])
best_metric = float(checkpoint['best_validation_bleu4'])
bad_epochs = int(checkpoint['epochs_without_improvement'])
NEXT_EPOCH = recovery['next_epoch']
if 'rng_state' in checkpoint:
    restore_rng(checkpoint['rng_state'])
else:
    # The legacy RNG trajectory is irrecoverable. Record a reproducible restart seed,
    # without claiming bit-for-bit equivalence to the interrupted run.
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print('LEGACY RNG MISSING: seeded restart; not a bit-for-bit continuation')
SELECTED_BEST = RUN_ROOT / 'text_decoder_best.pt'
shutil.copy2(BEST_CHECKPOINT_PATH, SELECTED_BEST)
provenance = dict(source_sha256=sha256(CHECKPOINT_PATH), source_epoch=recovery['epoch'],
                  historical_rng_missing=recovery['legacy_rng_missing'],
                  source_runtime_sha256=RUNTIME_SHA256, versions=versions, gpu=gpu.name,
                  dataset='cov_ctr', paired_regime='0%', timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
print('VERIFIED continuation:', recovery)
'''

TRAINING_HELPERS = r'''
# Definitions only. No training begins in this cell.
def validation_bleu4():
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    require(shutil.which('java') is not None, 'COCO PTB tokenizer requires Java')
    decoder.eval()
    references, predictions = [], []
    with torch.no_grad():
        for start in range(0, 71, 6):
            rows = records['validation'][start:start + 6]
            output = decoder.generate(feature_batch(rows, 'validation'))
            references.extend(row['report'] for row in rows)
            predictions.extend(decode_report(row, vocabulary) for row in output.cpu())
    ptb = PTBTokenizer()
    refs = ptb.tokenize({i: [{'caption': text}] for i, text in enumerate(references)})
    preds = ptb.tokenize({i: [{'caption': text}] for i, text in enumerate(predictions)})
    return 100.0 * float(Bleu(4).compute_score(refs, preds)[0][3])

def export_bundle(epoch):
    # Immutable, epoch-labelled snapshot. Always read the CURRENT best file; no stale slim copy.
    bundle_dir = RUN_ROOT / ('export_epoch_' + str(epoch) + '_' + uuid.uuid4().hex[:8])
    bundle_dir.mkdir(exist_ok=False)
    for name in DEPENDENCIES:
        shutil.copy2(ARTIFACT_DIR / name, bundle_dir / name)
    for name in ('text_decoder_last.pt', 'text_decoder_best.pt', 'text_training_history.json', 'configuration.json'):
        shutil.copy2(RUN_ROOT / name, bundle_dir / name)
    (bundle_dir / 'gate1_runtime.py').write_text(RUNTIME_SOURCE, encoding='utf-8')
    manifest = {p.name: sha256(p) for p in bundle_dir.iterdir() if p.is_file()}
    (bundle_dir / 'sha256_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    bundle = bundle_dir.with_suffix('.zip')
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in bundle_dir.iterdir():
            archive.write(item, item.name)
    bundle.with_suffix('.zip.sha256').write_text(sha256(bundle) + '  ' + bundle.name + '\n', encoding='utf-8')
    print('EXPORT AVAILABLE, DURABILITY UNVERIFIED:', bundle)
    print('Download/publish only with authorization; retrieve and hash-check before marking durable')
    return bundle

def save_epoch(epoch):
    payload = training_payload(decoder, optimizer, epoch, history, best_metric, bad_epochs,
                               artifacts['dependency_hashes'], provenance)
    # Best and latest are checked as a pair before writing the epoch snapshot.
    inspect_training_state(payload, decoder, load_checkpoint(SELECTED_BEST))
    atomic_save(payload, RUN_ROOT / 'text_decoder_last.pt')
    (RUN_ROOT / 'text_training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (RUN_ROOT / 'configuration.json').write_text(json.dumps(
        dict(architecture=ARCHITECTURE, training=TRAINING, provenance=provenance,
             dependency_hashes=artifacts['dependency_hashes']), indent=2), encoding='utf-8')

def continue_training():
    global best_metric, bad_epochs
    require(ENABLE_APPROVED_TRAINING is True, 'Training remains disabled')
    require(bad_epochs < 50 and NEXT_EPOCH <= 200, 'Stopping condition already met; no restart')
    require(history[-1]['epoch'] == NEXT_EPOCH - 1, 'This cell already advanced; explicitly reload a new checkpoint before continuing')
    # Restore checkpoint RNG immediately before the first epoch; preflight must not consume it.
    if 'rng_state' in checkpoint:
        restore_rng(checkpoint['rng_state'])
    else:
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    session_final_epoch = min(200, NEXT_EPOCH + 4)
    for epoch in range(NEXT_EPOCH, session_final_epoch + 1):
        decoder.train()
        generator = torch.Generator().manual_seed(42 + epoch)
        losses = []
        loader = torch.utils.data.DataLoader(records['train'], batch_size=6, shuffle=True,
                                             generator=generator, num_workers=0,
                                             collate_fn=lambda rows: rows)
        for rows in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = decoder_loss(feature_batch(rows, 'train'), token_batch(rows))
            require(bool(torch.isfinite(loss)), f'Non-finite loss at epoch {epoch}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = validation_bleu4()
        require(math.isfinite(score), 'Non-finite validation BLEU-4')
        history.append(dict(epoch=epoch, training_loss=sum(losses)/len(losses), validation_bleu4=score))
        if score > best_metric:
            best_metric, bad_epochs = score, 0
            atomic_save(dict(epoch=epoch, decoder=decoder.state_dict(), validation_bleu4=score,
                             vocabulary_size=368, max_tokens=160, architecture=ARCHITECTURE,
                             dependency_hashes=artifacts['dependency_hashes']), SELECTED_BEST)
        else:
            bad_epochs += 1
        save_epoch(epoch)  # latest local session checkpoint EVERY completed epoch
        print(epoch, 'text reconstruction BLEU-4:', score, 'best:', best_metric, 'patience:', bad_epochs, '/50')
        if epoch % 5 == 0 or bad_epochs >= 50 or epoch == session_final_epoch:
            export_bundle(epoch)
        if bad_epochs >= 50:
            break
    print('Session chunk complete. Retrieve and hash-verify the export before another chunk.')
'''


def build():
    import hashlib
    runtime = (ROOT / "scripts/zeromrg_gate1_runtime.py").read_text(encoding="utf-8")
    audit = json.loads((ROOT / "artifacts/reconciliation/gate1_20261010_checkpoint_audit/checkpoint_audit.json").read_text(encoding="utf-8"))
    canonical = next(item for item in audit["archives"] if item.get("verification", {}).get("epoch") == 90)
    pinned = "CANONICAL_DEPENDENCY_HASHES = " + repr(canonical["dependency_hashes"])
    pinned += "\nRUNTIME_SHA256 = " + repr(hashlib.sha256(runtime.encode()).hexdigest())
    pinned += "\nRUNTIME_SOURCE = " + repr(runtime)
    for kind in ("preflight", "continuation"):
        directory = ROOT / "notebooks" / "gate1" / kind
        directory.mkdir(parents=True, exist_ok=True)
        filename = "ZeroMRG_COV_CTR_" + ("Preflight_Only" if kind == "preflight" else "Checkpoint_Continuation") + "_Kaggle.ipynb"
        intro = ("# ZeroMRG Gate 1 COV-CTR 0%-paired " + kind + "\n\n"
                 "Kaggle-only execution. No held-out test generation or paired experiments. "
                 "The epoch-90 latest artifact is valid, but the saved best file is epoch 8, whereas history selects epoch 85. "
                 "This notebook reports/refuses that mismatch. No epoch-85 weights are claimed verified.\n\n"
                 "Set the exact checkpoint, artifact, best-checkpoint, and raw-dataset paths in the configuration cell. "
                 "Dependencies and vocabulary are reused, never rebuilt. The embedded runtime is generated from scripts/zeromrg_gate1_runtime.py. "
                 "Legacy checkpoints lack RNG state and exact encoder revisions; bit-for-bit continuation is unavailable.")
        cells = [cell(intro, "markdown"), cell(runtime), cell(pinned), cell(SETUP),
                 cell(ARTIFACT_CHECK), cell(LATEST_DIAGNOSTIC)]
        if kind == "preflight":
            cells += [cell("## Validate the selected best and frozen image path\n\n"
                           "The latest checkpoint diagnostic above is separate from validation-selected zero-shot readiness. "
                           "With current inputs the next cell writes a BLOCKED report and stops before loading CLIP. "
                           "After the correct best artifact is recovered, it uses one fixed training image, with no metrics or tuning.", "markdown"),
                      cell(PREFLIGHT_BEST)]
        else:
            cells += [cell(CONTINUATION_RESTORE), cell(TRAINING_HELPERS),
                      cell("## Explicitly authorized training only\n\n"
                           "Full training is disabled by default. Gate 1 must be unblocked and the owner must approve remote training "
                           "before changing the flag. Retain SGD, batch 6, patience 50 and the documented 200-epoch cap. "
                           "A Save Version/publish operation is not performed by this notebook.", "markdown"),
                      cell("ENABLE_APPROVED_TRAINING = False\nif ENABLE_APPROVED_TRAINING:\n    continue_training()\nelse:\n    print('Training disabled; checkpoint-driven restoration only')")]
        notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                                 language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
        for index, item in enumerate(cells):
            item["id"] = f"gate1-{kind}-{index:02d}"
        notebook_text = json.dumps(notebook, indent=1) + "\n"
        notebook_path = directory / filename
        if not notebook_path.exists() or notebook_path.read_text(encoding="utf-8") != notebook_text:
            notebook_path.write_text(notebook_text, encoding="utf-8")
        metadata_value = dict(id="inshasadaf114/zeromrg-covctr-gate1-" + kind,
                              title="ZeroMRG COVCTR Gate1 " + kind.title(), code_file=filename,
                              language="python", kernel_type="notebook", is_private=True,
                              enable_gpu=True, enable_tpu=False, enable_internet=True,
                              dataset_sources=["inshasadaf114/dataset", "inshasadaf114/zeromrg-checkpoint-backup5-zip"],
                              kernel_sources=[], competition_sources=[], model_sources=[])
        metadata_path = directory / "kernel-metadata.json"
        metadata_text = json.dumps(metadata_value, indent=2) + "\n"
        if not metadata_path.exists() or metadata_path.read_text(encoding="utf-8") != metadata_text:
            metadata_path.write_text(metadata_text, encoding="utf-8")
        print(directory / filename)


if __name__ == "__main__":
    build()
