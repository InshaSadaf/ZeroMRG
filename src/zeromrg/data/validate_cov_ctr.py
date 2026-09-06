"""Read-only validation for the supplied COV-CTR dataset."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from .archive_io import (
    DatasetSource,
    duplicate_content_groups,
    inspect_images,
    open_dataset_source,
    unique_member_by_basename,
)
from .schemas import ConflictSample, DatasetAuditRecord, ExcludedSample, ValidSample, stable_id

DATASET = "cov_ctr"
ANNOTATION_BASENAME = "reports_ZH_EN.csv"
REQUIRED_COLUMNS = (
    "image_id",
    "findings",
    "terminologies",
    "COVID",
    "impression",
    "reports_En",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class CovCtrValidation:
    valid_samples: tuple[ValidSample, ...]
    audit_records: tuple[DatasetAuditRecord | ExcludedSample | ConflictSample, ...]
    summary: dict[str, Any]


def _read_annotations(source: DatasetSource, annotation_path: str) -> tuple[bytes, list[dict[str, str]]]:
    raw = source.read_bytes(annotation_path)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = tuple(reader.fieldnames or ())
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"COV-CTR annotation table is missing required columns: {missing}")
    records: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        record = {column: (row.get(column) or "") for column in columns}
        record["_row_number"] = str(row_number)
        records.append(record)
    return raw, records


def _conflicting_fields(rows: list[dict[str, str]]) -> list[str]:
    return [
        column
        for column in REQUIRED_COLUMNS
        if len({row.get(column, "") for row in rows}) > 1
    ]


def _image_record(inspection: Any) -> dict[str, Any]:
    record = inspection.to_dict()
    record.pop("content_sha256", None)
    return record


def validate_cov_ctr(
    configured_path: str,
    expected: Mapping[str, int] | None = None,
) -> CovCtrValidation:
    """Validate all COV-CTR metadata and image members without writing source data."""

    with open_dataset_source(configured_path, [ANNOTATION_BASENAME]) as source:
        annotation_path = unique_member_by_basename(source, ANNOTATION_BASENAME)
        annotation_bytes, rows = _read_annotations(source, annotation_path)
        image_paths = [
            member.path
            for member in source.members
            if PurePosixPath(member.path).suffix.lower() in IMAGE_SUFFIXES
        ]
        inspections = inspect_images(source, image_paths)

        images_by_basename: dict[str, list[str]] = defaultdict(list)
        for path in image_paths:
            images_by_basename[PurePosixPath(path).name].append(path)
        rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_image[row["image_id"]].append(row)

        valid_samples: list[ValidSample] = []
        audit: list[DatasetAuditRecord | ExcludedSample | ConflictSample] = []
        conflict_keys: list[str] = []
        missing_image_keys: list[str] = []

        for image_id in sorted(rows_by_image):
            grouped_rows = rows_by_image[image_id]
            physical_paths = sorted(images_by_basename.get(image_id, []))
            row_ids = [f"{annotation_path}#row={row['_row_number']}" for row in grouped_rows]
            if len(grouped_rows) > 1:
                fields = _conflicting_fields(grouped_rows)
                if fields:
                    conflict_keys.append(image_id)
                    audit.append(
                        ConflictSample(
                            dataset=DATASET,
                            conflict_id=stable_id(DATASET, "conflict", image_id),
                            source_key=image_id,
                            source_paths=[annotation_path, *physical_paths],
                            row_ids=row_ids,
                            conflicting_fields=fields,
                            rows=[{key: row[key] for key in REQUIRED_COLUMNS} for row in grouped_rows],
                        )
                    )
                    continue
                audit.append(
                    ExcludedSample(
                        dataset=DATASET,
                        record_id=stable_id(DATASET, "excluded", image_id),
                        source_key=image_id,
                        reason="duplicate_annotation_rows",
                        source_paths=[annotation_path, *physical_paths],
                        details={"row_ids": row_ids},
                    )
                )
                continue
            if not physical_paths:
                missing_image_keys.append(image_id)
                audit.append(
                    ExcludedSample(
                        dataset=DATASET,
                        record_id=stable_id(DATASET, "excluded", image_id),
                        source_key=image_id,
                        reason="report_without_image",
                        source_paths=[annotation_path],
                        details={"row_id": row_ids[0]},
                    )
                )
                continue
            if len(physical_paths) > 1:
                audit.append(
                    ExcludedSample(
                        dataset=DATASET,
                        record_id=stable_id(DATASET, "excluded", image_id),
                        source_key=image_id,
                        reason="ambiguous_duplicate_image_basename",
                        source_paths=[annotation_path, *physical_paths],
                        details={"row_id": row_ids[0]},
                    )
                )
                continue

            row = grouped_rows[0]
            image_path = physical_paths[0]
            inspection = inspections[image_path]
            empty_fields = [column for column in REQUIRED_COLUMNS if not row[column].strip()]
            if inspection.corrupt or empty_fields:
                reasons = []
                if inspection.corrupt:
                    reasons.append("corrupt_or_unreadable_image")
                if "reports_En" in empty_fields:
                    reasons.append("empty_english_report")
                if any(field != "reports_En" for field in empty_fields):
                    reasons.append("missing_annotation_fields")
                audit.append(
                    ExcludedSample(
                        dataset=DATASET,
                        record_id=stable_id(DATASET, "excluded", image_id),
                        source_key=image_id,
                        reason=";".join(reasons),
                        source_paths=[annotation_path, image_path],
                        details={
                            "row_id": row_ids[0],
                            "empty_fields": empty_fields,
                            "image": _image_record(inspection),
                        },
                    )
                )
                continue

            sample_id = stable_id(DATASET, "sample", image_id)
            sample = ValidSample(
                dataset=DATASET,
                sample_id=sample_id,
                report_id=row_ids[0],
                study_id=None,
                image_ids=[image_id],
                source_image_paths=[image_path],
                report_text=row["reports_En"],
                report_fields={column: row[column] for column in REQUIRED_COLUMNS if column != "image_id"},
                image_metadata=[_image_record(inspection)],
                metadata={"annotation_path": annotation_path, "annotation_row": int(row["_row_number"]), "language": "en"},
            )
            valid_samples.append(sample)
            audit.append(
                DatasetAuditRecord(
                    dataset=DATASET,
                    audit_id=stable_id(DATASET, "audit", image_id),
                    status="valid",
                    source_key=image_id,
                    source_paths=[annotation_path, image_path],
                    reasons=[],
                    details={"sample_id": sample_id},
                )
            )

        unannotated_paths: list[str] = []
        for basename, paths in sorted(images_by_basename.items()):
            if basename not in rows_by_image:
                for path in sorted(paths):
                    unannotated_paths.append(path)
                    audit.append(
                        ExcludedSample(
                            dataset=DATASET,
                            record_id=stable_id(DATASET, "excluded_image", path),
                            source_key=basename,
                            reason="image_without_report",
                            source_paths=[path],
                            details={"image": _image_record(inspections[path])},
                        )
                    )

        actual_formats = Counter(item.format or "UNREADABLE" for item in inspections.values())
        modes = Counter(item.mode or "UNREADABLE" for item in inspections.values())
        extension_mismatches = [
            path
            for path, item in inspections.items()
            if not item.corrupt
            and ((PurePosixPath(path).suffix.lower() == ".png" and item.format != "PNG")
                 or (PurePosixPath(path).suffix.lower() in {".jpg", ".jpeg"} and item.format != "JPEG"))
        ]
        corrupt_paths = [path for path, item in inspections.items() if item.corrupt]
        duplicate_groups = duplicate_content_groups(inspections.values())
        empty_field_counts = {
            column: sum(not row[column].strip() for row in rows) for column in REQUIRED_COLUMNS
        }
        comparison = {}
        for key, expected_value in (expected or {}).items():
            actual_lookup = {
                "primary_usable_samples": len(valid_samples),
                "conflicting_duplicate_keys": len(conflict_keys),
                "unannotated_images": len(unannotated_paths),
            }
            if key in actual_lookup:
                comparison[key] = {
                    "expected": int(expected_value),
                    "actual": actual_lookup[key],
                    "matches": actual_lookup[key] == int(expected_value),
                }

        summary: dict[str, Any] = {
            "schema_version": "1.0",
            "dataset": DATASET,
            "source_kind": source.kind,
            "source_display_name": source.path.name,
            "structure": {
                "total_files": len(source.members),
                "annotation_path": annotation_path,
                "directory_file_counts": dict(
                    sorted(
                        Counter(
                            str(PurePosixPath(member.path).parent)
                            for member in source.members
                        ).items()
                    )
                ),
                "non_image_files": sorted(
                    member.path for member in source.members if member.path not in image_paths
                ),
                "image_extensions": dict(sorted(Counter(PurePosixPath(path).suffix.lower() for path in image_paths).items())),
            },
            "counts": {
                "physical_images": len(image_paths),
                "annotation_rows": len(rows),
                "unique_annotation_image_ids": len(rows_by_image),
                "valid_samples": len(valid_samples),
                "conflicting_duplicate_keys": len(conflict_keys),
                "duplicate_annotation_keys_total": sum(len(group) > 1 for group in rows_by_image.values()),
                "exact_duplicate_annotation_rows": len(rows)
                - len({tuple(row[column] for column in REQUIRED_COLUMNS) for row in rows}),
                "images_without_reports": len(unannotated_paths),
                "reports_without_images": len(missing_image_keys),
                "corrupt_images": len(corrupt_paths),
                "excluded_records": sum(not isinstance(record, DatasetAuditRecord) for record in audit),
            },
            "fields": {
                "columns": list(REQUIRED_COLUMNS),
                "empty_value_counts": empty_field_counts,
                "project_report_field": "reports_En",
                "project_report_language": "en",
            },
            "images": {
                "actual_formats": dict(sorted(actual_formats.items())),
                "modes": dict(sorted(modes.items())),
                "extension_mismatch_count": len(extension_mismatches),
                "extension_mismatch_paths": sorted(extension_mismatches),
                "corrupt_paths": sorted(corrupt_paths),
                "duplicate_content_groups": duplicate_groups,
            },
            "mapping": {
                "conflict_keys": sorted(conflict_keys),
                "unannotated_image_paths": sorted(unannotated_paths),
                "missing_image_keys": sorted(missing_image_keys),
            },
            "integrity": {
                "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                "image_inventory_sha256": source.inventory_hash(image_paths),
            },
            "reference_comparison": comparison,
        }
        return CovCtrValidation(tuple(valid_samples), tuple(audit), summary)
