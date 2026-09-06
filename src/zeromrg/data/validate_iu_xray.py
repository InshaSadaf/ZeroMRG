"""Read-only validation for the supplied IU-Xray dataset."""

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

DATASET = "iu_xray"
REPORT_BASENAME = "indiana_reports.csv"
PROJECTION_BASENAME = "indiana_projections.csv"
REPORT_COLUMNS = (
    "uid",
    "MeSH",
    "Problems",
    "image",
    "indication",
    "comparison",
    "findings",
    "impression",
)
PROJECTION_COLUMNS = ("uid", "filename", "projection")


@dataclass(frozen=True)
class IuXrayValidation:
    valid_studies: tuple[ValidSample, ...]
    audit_records: tuple[DatasetAuditRecord | ExcludedSample | ConflictSample, ...]
    summary: dict[str, Any]


def _read_csv(
    source: DatasetSource, path: str, required_columns: tuple[str, ...]
) -> tuple[bytes, list[dict[str, str]]]:
    raw = source.read_bytes(path)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = tuple(reader.fieldnames or ())
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(f"{PurePosixPath(path).name} is missing required columns: {missing}")
    records = []
    for row_number, row in enumerate(reader, start=2):
        record = {column: (row.get(column) or "") for column in columns}
        record["_row_number"] = str(row_number)
        records.append(record)
    return raw, records


def _image_record(inspection: Any, projection: str | None = None) -> dict[str, Any]:
    record = inspection.to_dict()
    record.pop("content_sha256", None)
    if projection is not None:
        record["projection"] = projection
    return record


def _conflicting_report_fields(rows: list[dict[str, str]]) -> list[str]:
    return [column for column in REPORT_COLUMNS if len({row[column] for row in rows}) > 1]


def validate_iu_xray(
    configured_path: str,
    expected: Mapping[str, int] | None = None,
) -> IuXrayValidation:
    """Validate IU-Xray tables and preserve one study record per UID."""

    with open_dataset_source(configured_path, [REPORT_BASENAME, PROJECTION_BASENAME]) as source:
        report_path = unique_member_by_basename(source, REPORT_BASENAME)
        projection_path = unique_member_by_basename(source, PROJECTION_BASENAME)
        report_bytes, reports = _read_csv(source, report_path, REPORT_COLUMNS)
        projection_bytes, projections = _read_csv(source, projection_path, PROJECTION_COLUMNS)
        image_paths = [
            member.path
            for member in source.members
            if PurePosixPath(member.path).suffix.lower() == ".png"
        ]
        inspections = inspect_images(source, image_paths)

        physical_by_basename: dict[str, list[str]] = defaultdict(list)
        for path in image_paths:
            physical_by_basename[PurePosixPath(path).name].append(path)
        reports_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reports:
            reports_by_uid[row["uid"]].append(row)
        projections_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in projections:
            projections_by_uid[row["uid"]].append(row)

        section_counts = Counter()
        for row in reports:
            has_findings = bool(row["findings"].strip())
            has_impression = bool(row["impression"].strip())
            if has_findings and has_impression:
                section_counts["both"] += 1
            elif has_findings:
                section_counts["findings_only"] += 1
            elif has_impression:
                section_counts["impression_only"] += 1
            else:
                section_counts["neither"] += 1

        valid_studies: list[ValidSample] = []
        audit: list[DatasetAuditRecord | ExcludedSample | ConflictSample] = []
        report_uids_without_projections: list[str] = []
        projection_rows_missing_images: list[dict[str, str]] = []

        for uid in sorted(reports_by_uid, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
            grouped_reports = reports_by_uid[uid]
            report_row_ids = [f"{report_path}#row={row['_row_number']}" for row in grouped_reports]
            uid_projections = projections_by_uid.get(uid, [])
            if len(grouped_reports) > 1:
                audit.append(
                    ConflictSample(
                        dataset=DATASET,
                        conflict_id=stable_id(DATASET, "conflict_report_uid", uid),
                        source_key=uid,
                        source_paths=[report_path, projection_path],
                        row_ids=report_row_ids,
                        conflicting_fields=_conflicting_report_fields(grouped_reports),
                        rows=[{column: row[column] for column in REPORT_COLUMNS} for row in grouped_reports],
                        reason="duplicate_report_uid",
                    )
                )
                continue

            report = grouped_reports[0]
            has_findings = bool(report["findings"].strip())
            has_impression = bool(report["impression"].strip())
            reasons: list[str] = []
            if not has_findings and not has_impression:
                reasons.append("missing_findings_and_impression")
            elif not has_findings:
                reasons.append("missing_findings")
            elif not has_impression:
                reasons.append("missing_impression")
            if not uid_projections:
                report_uids_without_projections.append(uid)
                reasons.append("report_without_projection")

            image_ids: list[str] = []
            source_paths: list[str] = []
            image_metadata: list[dict[str, Any]] = []
            projection_row_ids: list[str] = []
            for projection in uid_projections:
                filename = projection["filename"]
                projection_row_ids.append(
                    f"{projection_path}#row={projection['_row_number']}"
                )
                matches = sorted(physical_by_basename.get(filename, []))
                if not matches:
                    reasons.append("projection_missing_physical_image")
                    projection_rows_missing_images.append(
                        {"uid": uid, "filename": filename, "row": projection["_row_number"]}
                    )
                    continue
                if len(matches) > 1:
                    reasons.append("ambiguous_duplicate_image_basename")
                    continue
                image_path = matches[0]
                inspection = inspections[image_path]
                if inspection.corrupt:
                    reasons.append("corrupt_or_unreadable_image")
                image_ids.append(filename)
                source_paths.append(image_path)
                image_metadata.append(_image_record(inspection, projection["projection"]))

            reasons = sorted(set(reasons))
            if reasons:
                audit.append(
                    ExcludedSample(
                        dataset=DATASET,
                        record_id=stable_id(DATASET, "excluded_uid", uid),
                        source_key=uid,
                        reason=";".join(reasons),
                        source_paths=[report_path, projection_path, *source_paths],
                        details={
                            "report_row_id": report_row_ids[0],
                            "projection_row_ids": projection_row_ids,
                            "has_findings": has_findings,
                            "has_impression": has_impression,
                            "mapped_image_count": len(image_ids),
                        },
                    )
                )
                continue

            sample_id = stable_id(DATASET, "study", uid)
            sample = ValidSample(
                dataset=DATASET,
                sample_id=sample_id,
                report_id=uid,
                study_id=uid,
                image_ids=image_ids,
                source_image_paths=source_paths,
                report_text=None,
                report_fields={column: report[column] for column in REPORT_COLUMNS if column != "uid"},
                image_metadata=image_metadata,
                metadata={
                    "uid": uid,
                    "report_path": report_path,
                    "report_row": int(report["_row_number"]),
                    "projection_rows": [int(row["_row_number"]) for row in uid_projections],
                    "language": "en",
                    "view_count": len(image_ids),
                },
            )
            valid_studies.append(sample)
            audit.append(
                DatasetAuditRecord(
                    dataset=DATASET,
                    audit_id=stable_id(DATASET, "audit_uid", uid),
                    status="valid",
                    source_key=uid,
                    source_paths=[report_path, projection_path, *source_paths],
                    reasons=[],
                    details={"sample_id": sample_id, "view_count": len(image_ids)},
                )
            )

        projection_uids_without_reports = sorted(set(projections_by_uid) - set(reports_by_uid))
        for uid in projection_uids_without_reports:
            audit.append(
                ExcludedSample(
                    dataset=DATASET,
                    record_id=stable_id(DATASET, "excluded_projection_uid", uid),
                    source_key=uid,
                    reason="projection_uid_without_report",
                    source_paths=[projection_path],
                    details={"projection_rows": [int(row["_row_number"]) for row in projections_by_uid[uid]]},
                )
            )

        projected_filenames = {row["filename"] for row in projections}
        unmapped_image_paths: list[str] = []
        for basename, paths in sorted(physical_by_basename.items()):
            if basename not in projected_filenames:
                for path in sorted(paths):
                    unmapped_image_paths.append(path)
                    audit.append(
                        ExcludedSample(
                            dataset=DATASET,
                            record_id=stable_id(DATASET, "excluded_image", path),
                            source_key=basename,
                            reason="image_missing_from_projection_metadata",
                            source_paths=[path],
                            details={"image": _image_record(inspections[path])},
                        )
                    )

        formats = Counter(item.format or "UNREADABLE" for item in inspections.values())
        modes = Counter(item.mode or "UNREADABLE" for item in inspections.values())
        corrupt_paths = sorted(path for path, item in inspections.items() if item.corrupt)
        duplicate_groups = duplicate_content_groups(inspections.values())
        projection_values = Counter(row["projection"] for row in projections)
        views_per_uid = Counter(len(rows) for rows in projections_by_uid.values())
        empty_report_fields = {
            column: sum(not row[column].strip() for row in reports) for column in REPORT_COLUMNS
        }
        empty_projection_fields = {
            column: sum(not row[column].strip() for row in projections)
            for column in PROJECTION_COLUMNS
        }
        duplicate_projection_filenames = sorted(
            filename for filename, count in Counter(row["filename"] for row in projections).items() if count > 1
        )
        comparison = {}
        for key, expected_value in (expected or {}).items():
            actual_lookup = {"primary_usable_samples": len(valid_studies)}
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
                "report_path": report_path,
                "projection_path": projection_path,
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
            },
            "counts": {
                "physical_images": len(image_paths),
                "report_rows": len(reports),
                "unique_report_uids": len(reports_by_uid),
                "duplicate_report_uids": sum(len(group) > 1 for group in reports_by_uid.values()),
                "exact_duplicate_report_rows": len(reports)
                - len({tuple(row[column] for column in REPORT_COLUMNS) for row in reports}),
                "projection_rows": len(projections),
                "unique_projection_uids": len(projections_by_uid),
                "exact_duplicate_projection_rows": len(projections)
                - len({tuple(row[column] for column in PROJECTION_COLUMNS) for row in projections}),
                "valid_uids": len(valid_studies),
                "both_sections_uids": section_counts["both"],
                "findings_only_uids": section_counts["findings_only"],
                "impression_only_uids": section_counts["impression_only"],
                "neither_section_uids": section_counts["neither"],
                "images_missing_projection_metadata": len(unmapped_image_paths),
                "projection_rows_missing_images": len(projection_rows_missing_images),
                "report_uids_without_projections": len(report_uids_without_projections),
                "projection_uids_without_reports": len(projection_uids_without_reports),
                "corrupt_images": len(corrupt_paths),
                "excluded_records": sum(not isinstance(record, DatasetAuditRecord) for record in audit),
            },
            "fields": {
                "report_columns": list(REPORT_COLUMNS),
                "projection_columns": list(PROJECTION_COLUMNS),
                "empty_report_value_counts": empty_report_fields,
                "empty_projection_value_counts": empty_projection_fields,
                "project_target_fields": ["impression", "findings"],
                "project_report_language": "en",
            },
            "images": {
                "actual_formats": dict(sorted(formats.items())),
                "modes": dict(sorted(modes.items())),
                "corrupt_paths": corrupt_paths,
                "duplicate_content_groups": duplicate_groups,
            },
            "mapping": {
                "projection_values": dict(sorted(projection_values.items())),
                "unexpected_projection_values": sorted(
                    value for value in projection_values if value not in {"Frontal", "Lateral"}
                ),
                "views_per_uid": {str(key): value for key, value in sorted(views_per_uid.items())},
                "unmapped_image_paths": sorted(unmapped_image_paths),
                "projection_rows_missing_images": projection_rows_missing_images,
                "report_uids_without_projections": sorted(report_uids_without_projections),
                "projection_uids_without_reports": projection_uids_without_reports,
                "duplicate_projection_filenames": duplicate_projection_filenames,
            },
            "integrity": {
                "report_csv_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "projection_csv_sha256": hashlib.sha256(projection_bytes).hexdigest(),
                "image_inventory_sha256": source.inventory_hash(image_paths),
            },
            "reference_comparison": comparison,
        }
        return IuXrayValidation(tuple(valid_studies), tuple(audit), summary)
