"""Content-bound COV attachment resolution. No models, network or reference scoring."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
ATTACHMENT = dict(ref='inshasadaf114/dataset', dataset_id=11916862)


class CovDataError(ValueError):
    def __init__(self, code, **details):
        self.diagnostic = dict(code=code, **details)
        super().__init__(json.dumps(self.diagnostic, sort_keys=True))


def ensure(condition, code, **details):
    if not condition:
        raise CovDataError(code, **details)


def stream_hash(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return stream_hash(stream)


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def validate_contract(contract, train_ids, prompt_ids):
    ensure(contract['schema'] == 'zeromrg_cov_identity_v1' and
           contract['attachment'] == ATTACHMENT, 'ATTACHMENT_CONTRACT_MISMATCH')
    groups = contract['splits']
    for name, count in dict(validated=714, train=571, validation=71, test=72, prompt=500).items():
        ids = groups[name]
        ensure(len(ids) == len(set(ids)) == count, 'CANONICAL_ID_COUNT', split=name, expected=count)
        ensure(all(isinstance(i, str) and PurePosixPath(i).name == i and
                   i not in ('.', '..') and '\\' not in i for i in ids), 'UNSAFE_IMAGE_ID', split=name)
    train, validation, test = (set(groups[n]) for n in ('train', 'validation', 'test'))
    ensure(not (train & validation or train & test or validation & test) and
           train | validation | test == set(groups['validated']), 'SPLIT_MEMBERSHIP_MISMATCH')
    ensure(groups['train'] == list(train_ids), 'TRAIN_ORDER_OR_MEMBERSHIP_MISMATCH')
    ensure(groups['prompt'] == list(prompt_ids) and set(prompt_ids) <= train and
           ids_hash(prompt_ids) == contract['prompt_ids_sha256'], 'PROMPT_BANK_MEMBERSHIP_OR_ORDER_MISMATCH')
    for name, ids in groups.items():
        ensure(ids_hash(ids) == contract['split_ids_sha256'][name],
               'SPLIT_ID_FINGERPRINT_MISMATCH', split=name)
    ensure(set(groups['validated']) <= set(contract['images']) and
           len(contract['images']) == contract['physical_image_count'] == 746,
           'IMAGE_CONTRACT_POPULATION_MISMATCH')


def verify_checkpoint_manifests(archive, contract):
    """Only hash report-bearing CSV bytes; no held-out report parsing or scoring."""
    for name, expected in contract['manifest_sha256'].items():
        member = 'dependencies/' + name
        ensure(archive.namelist().count(member) == 1, 'CHECKPOINT_MANIFEST_MISSING_OR_AMBIGUOUS', file=name)
        with archive.open(member) as stream:
            actual = stream_hash(stream)  # reading to EOF also verifies ZIP CRC
        ensure(actual == expected, 'CHECKPOINT_MANIFEST_HASH_MISMATCH', file=name,
               expected=expected, actual=actual)


def _index(entries):
    index = {}
    for name, entry in entries:
        ensure(name not in index, 'DUPLICATE_IMAGE_IDENTIFIER', image_id=name)
        index[name] = entry
    return index


def _bounded_entries(root):
    entries = []
    for path in root.rglob('*'):
        ensure(len(entries) < 10000, 'DATASET_SCAN_BOUND', root=str(root))
        entries.append(path)
    return entries


def _validate_population(index, contract, hash_image, progress=None):
    expected = contract['images']
    missing, extra = sorted(set(expected) - set(index)), sorted(set(index) - set(expected))
    ensure(not missing and not extra, 'IMAGE_INVENTORY_MISMATCH', missing_count=len(missing),
           extra_count=len(extra), missing=missing[:10], extra=extra[:10])
    for count, identifier in enumerate(sorted(expected),1):
        actual = hash_image(index[identifier])
        ensure(actual == expected[identifier]['sha256'], 'IMAGE_CONTENT_MISMATCH',
               image_id=identifier, expected=expected[identifier]['sha256'], actual=actual)
        if progress and (count==1 or count%100==0 or count==len(expected)):
            progress('dataset_image_hash_progress', verified=count, total=len(expected))


def _directory_candidate(report, attachment_root, contract, progress=None):
    root = report.parent.resolve()
    ensure(report.resolve().is_relative_to(attachment_root), 'REPORT_ESCAPES_ATTACHMENT', path=str(report))
    ensure(file_hash(report) == contract['annotation']['sha256'], 'ANNOTATION_CONTENT_MISMATCH', path=str(report))
    files = _bounded_entries(root)
    images = [p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    for p in images:
        ensure(p.resolve().is_relative_to(root) and p.resolve().is_relative_to(attachment_root),
               'IMAGE_ESCAPES_DATA_ROOT', image_id=p.name)
    index = _index((p.name, p) for p in images)
    _validate_population(index, contract, file_hash, progress)
    return dict(kind='directory', root=root, report=report, images=index)


def _archive_candidate(path, contract, progress=None):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        ensure(len(entries) <= 10000 and len({e.filename for e in entries}) == len(entries),
               'ARCHIVE_ENTRY_BOUND_OR_DUPLICATE', path=str(path))
        reports = [e for e in entries if not e.is_dir() and PurePosixPath(e.filename).name == contract['annotation']['filename']]
        ensure(len(reports) == 1, 'ARCHIVE_REPORT_MISSING_OR_AMBIGUOUS', path=str(path), count=len(reports))
        prefix = PurePosixPath(reports[0].filename).parent
        for e in entries:
            name = PurePosixPath(e.filename)
            ensure(not name.is_absolute() and '..' not in name.parts and '\\' not in e.filename,
                   'UNSAFE_ARCHIVE_MEMBER', member=e.filename)
            ensure((e.external_attr >> 16) & 0o170000 != 0o120000,
                   'ARCHIVE_SYMLINK_REJECTED', member=e.filename)
        with archive.open(reports[0]) as stream:
            ensure(stream_hash(stream) == contract['annotation']['sha256'], 'ANNOTATION_CONTENT_MISMATCH', path=str(path))
        images = [e for e in entries if not e.is_dir() and
                  PurePosixPath(e.filename).is_relative_to(prefix) and
                  PurePosixPath(e.filename).suffix.lower() in IMAGE_EXTENSIONS]
        index = _index((PurePosixPath(e.filename).name, e) for e in images)
        def hash_image(entry):
            ensure(entry.file_size == contract['images'][PurePosixPath(entry.filename).name]['bytes'],
                   'IMAGE_SIZE_MISMATCH', image_id=PurePosixPath(entry.filename).name)
            with archive.open(entry) as stream:
                return stream_hash(stream)
        _validate_population(index, contract, hash_image, progress)
        return dict(kind='archive', root=path.resolve(), report=reports[0].filename,
                    images={k: v.filename for k, v in index.items()})


def resolve_cov_dataset(input_root, configured_root, contract, train_ids, prompt_ids, scratch, progress=None):
    """Resolve only approved owner/slug or Kaggle short-slug mounts, bound by bytes.

    Multiple candidates fail even if one validates. Ordering never chooses a root.
    CSV contents are hashed, never parsed here; no clinical target reaches decoding.
    """
    validate_contract(contract, train_ids, prompt_ids)
    input_root = Path(input_root).resolve()
    ensure(input_root.is_dir(), 'INPUT_ROOT_MISSING', path=str(input_root))
    ref = contract['attachment']['ref']; owner, slug = ref.split('/')
    layouts = [input_root/'datasets'/owner/slug, input_root/slug]
    mounts = []
    for path in layouts:
        if path.is_dir():
            resolved = path.resolve()
            ensure(resolved.is_relative_to(input_root), 'ATTACHMENT_ESCAPES_INPUT', path=str(path))
            if resolved not in mounts:
                mounts.append(resolved)
    ensure(mounts, 'APPROVED_ATTACHMENT_NOT_FOUND', attachment=ref,
           expected_layouts=[str(p) for p in layouts], configured_root=str(configured_root),
           observed_top_directories=sorted(p.name for p in input_root.iterdir() if p.is_dir())[:40])
    candidates = []
    for mount in mounts:
        # rglob is layout-independent; bounds reject unexpected enormous attachments.
        entries = _bounded_entries(mount)
        for path in entries:
            if not path.is_file():
                continue
            if path.name == contract['annotation']['filename']:
                candidates.append(('directory', path, mount))
            elif path.name == contract['archive_filename']:
                ensure(path.resolve().is_relative_to(mount), 'ARCHIVE_ESCAPES_ATTACHMENT', path=str(path))
                candidates.append(('archive', path, mount))
    unique = {(kind, path.resolve()): (kind, path, mount) for kind, path, mount in candidates}
    candidates = list(unique.values())
    ensure(len(candidates) == 1, 'DATASET_ROOT_MISSING_OR_AMBIGUOUS', count=len(candidates),
           candidates=[str(path) for _, path, _ in candidates][:10], attachment=ref,
           action='Keep one original COV annotation tree OR its original archive inside the approved attachment; do not substitute data')
    kind, path, mount = candidates[0]
    if progress: progress('dataset_candidate_selected',kind=kind,path=str(path))
    try:
        resolved = (_directory_candidate(path, mount, contract, progress) if kind == 'directory'
                    else _archive_candidate(path, contract, progress))
    except CovDataError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise CovDataError('DATASET_READ_OR_CRC_FAILURE', path=str(path), error_type=type(error).__name__) from None
    evidence = dict(status='PASS_CONTENT_BOUND_DATA_RESOLUTION', attachment=contract['attachment'],
                    attachment_root=str(mount), kind=kind, discovered_source=str(resolved['root']),
                    configured_root=str(configured_root), configured_root_exists=Path(configured_root).is_dir(),
                    annotation_sha256=contract['annotation']['sha256'], physical_images_verified=746,
                    valid_pairs=714, split_counts=dict(train=571, validation=71, test=72),
                    prompt_count=500, prompt_ids_sha256=contract['prompt_ids_sha256'],
                    report_text_parsed=False, images_decoded=False, scoring=False)
    if kind == 'archive':
        destination = Path(scratch)/'verified_cov_images'
        destination.mkdir(exist_ok=False, parents=True)
        # Materialize canonical images only into ephemeral scratch, never into input.
        with zipfile.ZipFile(path) as archive:
            for count, identifier in enumerate(contract['splits']['validated'],1):
                with archive.open(resolved['images'][identifier]) as source, (destination/identifier).open('xb') as out:
                    shutil.copyfileobj(source, out)
                ensure(file_hash(destination/identifier) == contract['images'][identifier]['sha256'],
                       'EXTRACTED_IMAGE_CONTENT_MISMATCH', image_id=identifier)
                if progress and (count==1 or count%100==0 or count==714):
                    progress('dataset_archive_extract_progress',completed=count,total=714)
        images = {i: destination/i for i in contract['splits']['validated']}
        root = destination
    else:
        root, images = resolved['root'], resolved['images']
    evidence['resolved_image_root'] = str(root)
    return dict(root=root, images=images, evidence=evidence)
