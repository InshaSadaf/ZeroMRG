"""Build and independently validate the portable IU-Xray-500 package."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .archive_io import DatasetSource, open_dataset_source, unique_member_by_basename
from .schemas import canonical_json_bytes, sha256_bytes, write_json
from .split import SplitIntegrityError, file_sha256
from .validate_iu_xray import (
    PROJECTION_BASENAME,
    PROJECTION_COLUMNS,
    REPORT_BASENAME,
    REPORT_COLUMNS,
    validate_iu_xray,
)

PACKAGE_VERSION = "iu_xray_500_package_v1"
_COPY_BLOCK_SIZE = 1024 * 1024


def _read_rows(source: DatasetSource, basename: str) -> tuple[list[str], list[dict[str, str]]]:
    member = unique_member_by_basename(source, basename)
    reader = csv.DictReader(io.StringIO(source.read_bytes(member).decode("utf-8-sig"), newline=""))
    fields = list(reader.fieldnames or ())
    return fields, [{field: row.get(field) or "" for field in fields} for row in reader]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _copy_and_hash(source: DatasetSource, member: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(member) as input_handle, temporary.open("wb") as output_handle:
        for block in iter(lambda: input_handle.read(_COPY_BLOCK_SIZE), b""):
            digest.update(block)
            output_handle.write(block)
    temporary.replace(destination)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_sized_summary(summary: dict[str, Any], destination: Path, bytes_without_summary: int) -> dict[str, Any]:
    """Record an exact directory size despite the summary containing that size."""

    sized = dict(summary)
    sized["package_directory_size_bytes"] = int(bytes_without_summary)
    for _ in range(10):
        sized["artifact_hash"] = sha256_bytes(
            canonical_json_bytes({key: value for key, value in sized.items() if key != "artifact_hash"})
        )
        payload_size = len(
            (json.dumps(sized, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        )
        exact_size = bytes_without_summary + payload_size
        if sized["package_directory_size_bytes"] == exact_size:
            write_json(sized, destination)
            return sized
        sized["package_directory_size_bytes"] = exact_size
    raise SplitIntegrityError("Could not stabilize package directory size metadata")


def _deterministic_zip(package_dir: Path, destination: Path) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted((item for item in package_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(package_dir.parent).as_posix()):
            relative = path.relative_to(package_dir.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as input_handle, archive.open(info, "w", force_zip64=True) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=_COPY_BLOCK_SIZE)
    temporary.replace(destination)
    return file_sha256(destination)


def validate_iu_xray_subset_package(
    package_dir: str | Path,
    selected_uids: Sequence[str],
    expected_image_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate package semantics and exact copied image bytes independently."""

    package_dir = Path(package_dir)
    result = validate_iu_xray(str(package_dir), expected={"primary_usable_samples": len(selected_uids)})
    actual_uids = {str(sample.study_id) for sample in result.valid_studies}
    expected_uids = {str(uid) for uid in selected_uids}
    if actual_uids != expected_uids:
        raise SplitIntegrityError("Packaged UID set does not exactly match subset selection")
    counts = result.summary["counts"]
    forbidden = {
        "projection_rows_missing_images": counts["projection_rows_missing_images"],
        "images_missing_projection_metadata": counts["images_missing_projection_metadata"],
        "corrupt_images": counts["corrupt_images"],
        "report_uids_without_projections": counts["report_uids_without_projections"],
        "projection_uids_without_reports": counts["projection_uids_without_reports"],
    }
    if any(forbidden.values()):
        raise SplitIntegrityError(f"Packaged subset failed integrity checks: {forbidden}")
    image_root = package_dir / "images" / "images_normalized"
    actual_names = {path.name for path in image_root.iterdir() if path.is_file()}
    if actual_names != set(expected_image_hashes):
        raise SplitIntegrityError("Packaged image inventory differs from selected projection filenames")
    for filename, expected_hash in expected_image_hashes.items():
        actual_hash = file_sha256(image_root / filename)
        if actual_hash != expected_hash:
            raise SplitIntegrityError(f"Packaged image bytes changed for {filename}")
    return result.summary


def build_iu_xray_subset_package(
    *,
    source_path: str | Path,
    subset_artifact_path: str | Path,
    preparation_summary_path: str | Path,
    package_dir: str | Path,
    create_zip: bool = True,
) -> dict[str, Any]:
    """Filter original metadata and stream selected image bytes into a portable package."""

    subset_path = Path(subset_artifact_path)
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    selected = [str(uid) for uid in subset["selected_uids"]]
    selected_set = set(selected)
    if len(selected) != int(subset["selected_uid_count"]) or len(selected) != len(selected_set):
        raise SplitIntegrityError("Subset artifact contains an invalid UID list")
    preparation = json.loads(Path(preparation_summary_path).read_text(encoding="utf-8"))
    package_dir = Path(package_dir)
    zip_path = package_dir.with_suffix(".zip")
    if package_dir.exists():
        summary_path = package_dir / "subset_summary.json"
        if not summary_path.is_file():
            raise SplitIntegrityError(f"Refusing to overwrite incomplete package directory: {package_dir}")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("subset_artifact_hash") != subset["artifact_hash"]:
            raise SplitIntegrityError(f"Existing package was built from another subset: {package_dir}")
        summary_changed = False
        if existing.get("package_directory_size_bytes") != _tree_size(package_dir):
            bytes_without_summary = _tree_size(package_dir) - summary_path.stat().st_size
            existing = _write_sized_summary(existing, summary_path, bytes_without_summary)
            summary_changed = True
        validation = validate_iu_xray_subset_package(
            package_dir, selected, existing["image_sha256"]
        )
        zip_hash = None
        if create_zip:
            if not zip_path.is_file() or summary_changed:
                zip_hash = _deterministic_zip(package_dir, zip_path)
            else:
                zip_hash = file_sha256(zip_path)
            sidecar = zip_path.with_suffix(".zip.sha256")
            expected_line = f"{zip_hash}  {zip_path.name}\n"
            if (
                sidecar.exists()
                and not summary_changed
                and sidecar.read_text(encoding="ascii") != expected_line
            ):
                raise SplitIntegrityError(f"ZIP checksum sidecar does not match archive: {sidecar}")
            if not sidecar.exists() or summary_changed:
                sidecar.write_text(expected_line, encoding="ascii")
        return {
            **existing,
            "package_directory_size_bytes": _tree_size(package_dir),
            "zip_path": str(zip_path) if create_zip else None,
            "zip_sha256": zip_hash,
            "reused": True,
            "validation": validation,
        }

    staging = package_dir.with_name(package_dir.name + ".building")
    if staging.exists():
        raise SplitIntegrityError(f"Staging directory already exists; inspect it before retrying: {staging}")
    staging.mkdir(parents=True)
    try:
        with open_dataset_source(source_path, [REPORT_BASENAME, PROJECTION_BASENAME]) as source:
            report_fields, report_rows = _read_rows(source, REPORT_BASENAME)
            projection_fields, projection_rows = _read_rows(source, PROJECTION_BASENAME)
            if not set(REPORT_COLUMNS).issubset(report_fields) or not set(PROJECTION_COLUMNS).issubset(projection_fields):
                raise SplitIntegrityError("Source IU-Xray tables do not have the expected schema")
            reports = [row for row in report_rows if row["uid"] in selected_set]
            projections = [row for row in projection_rows if row["uid"] in selected_set]
            if {row["uid"] for row in reports} != selected_set:
                raise SplitIntegrityError("Selected UID is missing from source report metadata")
            if {row["uid"] for row in projections} != selected_set:
                raise SplitIntegrityError("Selected UID is missing from source projection metadata")
            _write_csv(staging / REPORT_BASENAME, report_fields, reports)
            _write_csv(staging / PROJECTION_BASENAME, projection_fields, projections)

            members_by_basename: dict[str, list[str]] = {}
            for member in source.members:
                members_by_basename.setdefault(PurePosixPath(member.path).name, []).append(member.path)
            image_hashes: dict[str, str] = {}
            for filename in sorted({row["filename"] for row in projections}):
                matches = members_by_basename.get(filename, [])
                if len(matches) != 1:
                    raise SplitIntegrityError(
                        f"Expected exactly one source image named {filename}; found {len(matches)}"
                    )
                image_hashes[filename] = _copy_and_hash(
                    source, matches[0], staging / "images" / "images_normalized" / filename
                )

        shutil.copyfile(subset_path, staging / "subset_500_ids.json")
        readme = (
            "# IU-Xray-500\n\n"
            "This is the deterministic **RESOURCE-CONSTRAINED** IU-Xray execution subset. "
            "It is not the full paper dataset and must not be reported as a full-IU result.\n\n"
            "One report/study is identified by `uid`. All projection rows and every image view "
            "for each selected UID are retained. The CSV field values and PNG bytes come from "
            "the configured original IU-Xray source. `subset_500_ids.json` records selection provenance.\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        view_distribution = dict(
            sorted(
                Counter(
                    str(len(item.image_ids))
                    for item in validate_iu_xray(str(staging)).valid_studies
                ).items()
            )
        )
        summary: dict[str, Any] = {
            "artifact_version": PACKAGE_VERSION,
            "dataset": "iu_xray",
            "execution_profile": {"name": "kaggle_500", "label": "RESOURCE-CONSTRAINED"},
            "subset_artifact_hash": subset["artifact_hash"],
            "subset_artifact_file_sha256": file_sha256(subset_path),
            "preparation_summary_artifact_hash": preparation["artifact_hash"],
            "preparation_dependency_hashes": preparation["dependency_file_hashes"],
            "selected_uid_count": len(selected),
            "physical_image_count": len(image_hashes),
            "views_per_uid": view_distribution,
            "split_counts": preparation["split_counts"],
            "paired_10_count": preparation["paired_10_count"],
            "prompt_count": preparation["prompt_count"],
            "image_sha256": image_hashes,
            "image_inventory_hash": sha256_bytes(canonical_json_bytes({"image_sha256": image_hashes})),
            "report_csv_sha256": file_sha256(staging / REPORT_BASENAME),
            "projection_csv_sha256": file_sha256(staging / PROJECTION_BASENAME),
            "package_directory_size_bytes_before_summary": _tree_size(staging),
            "note": "RESOURCE-CONSTRAINED subset; not a full-IU paper result.",
        }
        summary = _write_sized_summary(
            summary,
            staging / "subset_summary.json",
            int(summary["package_directory_size_bytes_before_summary"]),
        )
        validation = validate_iu_xray_subset_package(staging, selected, image_hashes)
        staging.replace(package_dir)
        zip_hash = None
        if create_zip:
            zip_hash = _deterministic_zip(package_dir, zip_path)
            zip_path.with_suffix(".zip.sha256").write_text(f"{zip_hash}  {zip_path.name}\n", encoding="ascii")
        return {
            **summary,
            "package_directory_size_bytes": _tree_size(package_dir),
            "zip_path": str(zip_path) if create_zip else None,
            "zip_sha256": zip_hash,
            "reused": False,
            "validation": validation,
        }
    except Exception:
        # Keep the staging directory for forensic inspection; never replace source data.
        raise
