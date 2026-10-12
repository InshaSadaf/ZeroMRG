"""Construct ID/hash-only contract from existing verified local artifacts; no models."""
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from zeromrg_cov_data_resolver import ATTACHMENT, IMAGE_EXTENSIONS, file_hash, ids_hash, ensure, validate_contract

MANIFEST_HASHES = {
    'validated_manifest.csv': '1536c6eb78e2857e599da4f7b3854ed5fa01fd66e50f390dd315d72fe5572aee',
    'train_manifest.csv': '8902b8fea1163429aec6940a9744f7c792d1b7f9a00ca300a1fab17448ead973',
    'validation_manifest.csv': '94c555cb25a19485616065079139722bfe7122a17639267ba5e03d9e4cfa7c5d',
    'test_manifest.csv': 'fcbcd1bd12220b0a3a286c4b1a82a8543e34c0a3495c5d0b8c89c81033b3eb66',
    'prompt_manifest.csv': 'f60c45a1dc9beecba16917d767f1cf6b0f67231141a9226efdb771a27f9281de',
}
ANNOTATION_SHA = 'bfb892f3453276f652d8a14dcfdc6f215dd3eecd435394a5c05bdb8209a04489'


def build_contract(raw_archive, dependencies, config):
    records = {}
    for filename, expected in MANIFEST_HASHES.items():
        path = Path(dependencies)/filename
        ensure(file_hash(path) == expected, 'LOCAL_CANONICAL_MANIFEST_CHANGED', file=filename)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            records[filename.removesuffix('_manifest.csv')] = list(csv.DictReader(stream))
    groups = {name: [r['image_id'] for r in rows] for name, rows in records.items()}
    ensure(ids_hash(groups['prompt']) == config['prompt_ids_sha256'], 'LOCAL_PROMPT_ORDER_CHANGED')
    with zipfile.ZipFile(raw_archive) as archive:
        entries = archive.infolist()
        reports = [e for e in entries if not e.is_dir() and Path(e.filename).name == 'reports_ZH_EN.csv']
        ensure(len(reports) == 1, 'LOCAL_RAW_REPORT_AMBIGUOUS')
        content = archive.read(reports[0])
        ensure(hashlib.sha256(content).hexdigest() == ANNOTATION_SHA, 'LOCAL_RAW_ANNOTATION_CHANGED')
        reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
        rows = list(reader); counts = Counter(r['image_id'].strip() for r in rows)
        ensure(len(rows) == 726 and sum(n > 1 for n in counts.values()) == 6, 'LOCAL_RAW_POPULATION_CHANGED')
        unique = {r['image_id'].strip(): r for r in rows if counts[r['image_id'].strip()] == 1}
        images = {}
        for e in entries:
            if e.is_dir() or Path(e.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            identifier = Path(e.filename).name
            ensure(identifier not in images, 'LOCAL_IMAGE_AMBIGUOUS', image_id=identifier)
            content_image = archive.read(e)  # CRC + SHA, no decoding or report scoring
            images[identifier] = dict(bytes=len(content_image), sha256=hashlib.sha256(content_image).hexdigest())
        valid = {i for i, r in unique.items() if i in images and r['reports_En'].strip()}
        ensure(valid == set(groups['validated']) and len(images) == 746, 'LOCAL_VALID_PAIR_MEMBERSHIP_CHANGED')
        for row in records['validated']:
            normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', unique[row['image_id']]['reports_En'])).strip()
            ensure(normalized == row['report'], 'LOCAL_CANONICAL_REPORT_MISMATCH', image_id=row['image_id'])
        contract = dict(schema='zeromrg_cov_identity_v1', attachment=ATTACHMENT,
                        archive_filename='COV-CTR with English reports.zip',
                        annotation=dict(filename='reports_ZH_EN.csv', sha256=ANNOTATION_SHA,
                                        bytes=len(content), columns=reader.fieldnames, row_count=726),
                        images=images, physical_image_count=746, splits=groups,
                        split_ids_sha256={name:ids_hash(ids) for name,ids in groups.items()},
                        manifest_sha256=MANIFEST_HASHES, prompt_ids_sha256=ids_hash(groups['prompt']),
                        report_normalization='Historical NFC + whitespace collapse + strip',
                        exclusions=dict(conflicting_duplicate_keys=6, unannotated_images=26),
                        no_report_text_exported=True)
    validate_contract(contract, groups['train'], groups['prompt'])
    return contract
