"""Stable, JSON-serializable records for dataset validation manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "1.0"


def stable_id(dataset: str, record_type: str, source_key: str) -> str:
    """Create a portable ID from dataset-native relative identifiers."""

    material = f"{SCHEMA_VERSION}\0{dataset}\0{record_type}\0{source_key}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{dataset}:{record_type}:{digest}"


@dataclass(frozen=True)
class ValidSample:
    dataset: str
    sample_id: str
    report_id: str
    study_id: str | None
    image_ids: list[str]
    source_image_paths: list[str]
    report_text: str | None
    report_fields: dict[str, Any]
    image_metadata: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    record_type: str = field(default="valid_sample", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExcludedSample:
    dataset: str
    record_id: str
    source_key: str
    reason: str
    source_paths: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    record_type: str = field(default="excluded_sample", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConflictSample:
    dataset: str
    conflict_id: str
    source_key: str
    source_paths: list[str]
    row_ids: list[str]
    conflicting_fields: list[str]
    rows: list[dict[str, Any]]
    reason: str = "conflicting_annotation_rows"
    record_type: str = field(default="conflict_sample", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetAuditRecord:
    dataset: str
    audit_id: str
    status: str
    source_key: str
    source_paths: list[str]
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    record_type: str = field(default="dataset_audit", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitReadySample:
    """A validated sample eligible for a later, not-yet-created split."""

    dataset: str
    sample_id: str
    grouping_id: str
    image_ids: list[str]
    report_id: str
    record_type: str = field(default="split_ready_sample", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAwareSample:
    """Validated sample with deterministic split and normalized target text."""

    dataset: str
    sample_id: str
    membership_id: str
    report_id: str
    study_id: str | None
    split: str
    source_image_paths: list[str]
    image_ids: list[str]
    view_metadata: list[dict[str, Any]]
    raw_target_fields: dict[str, str]
    normalized_target_report: str
    normalization_version: str
    source_valid_manifest_sha256: str
    validity: dict[str, Any]
    provenance: dict[str, Any]
    record_type: str = field(default="split_aware_sample", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ManifestRecord = (
    ValidSample | ExcludedSample | ConflictSample | DatasetAuditRecord | SplitAwareSample
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_jsonl(records: Iterable[ManifestRecord], destination: str | Path) -> str:
    """Atomically write deterministic JSONL and return its SHA-256."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for record in records:
            line = canonical_json_bytes(record.to_dict())
            handle.write(line)
            digest.update(line)
    temporary.replace(destination)
    return digest.hexdigest()


def write_json(value: Mapping[str, Any], destination: str | Path) -> str:
    """Atomically write stable formatted JSON and return the file SHA-256."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return sha256_bytes(payload)
