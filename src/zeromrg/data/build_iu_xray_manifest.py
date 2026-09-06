"""Build IU-Xray validation artifacts under a configured writable directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .schemas import canonical_json_bytes, sha256_bytes, write_json, write_jsonl
from .validate_iu_xray import validate_iu_xray


@dataclass(frozen=True)
class ManifestArtifacts:
    output_directory: Path
    audit_manifest: Path
    valid_manifest: Path
    summary: Path
    hashes: dict[str, str]
    counts: dict[str, int]


def build_iu_xray_manifest(
    source_path: str | Path,
    processed_root: str | Path,
    expected: Mapping[str, int] | None = None,
) -> ManifestArtifacts:
    validation = validate_iu_xray(str(source_path), expected=expected)
    output = Path(processed_root) / "iu_xray"
    audit_path = output / "audit_manifest.jsonl"
    valid_path = output / "valid_studies.jsonl"
    summary_path = output / "validation_summary.json"

    audit_hash = write_jsonl(validation.audit_records, audit_path)
    valid_hash = write_jsonl(validation.valid_studies, valid_path)
    summary = dict(validation.summary)
    summary["integrity"] = dict(summary["integrity"])
    summary["integrity"].update(
        {"audit_manifest_sha256": audit_hash, "valid_manifest_sha256": valid_hash}
    )
    payload_hash = sha256_bytes(canonical_json_bytes(summary))
    summary["integrity"]["validation_summary_payload_sha256"] = payload_hash
    summary_hash = write_json(summary, summary_path)
    (output / "validation_summary.sha256").write_text(summary_hash + "\n", encoding="ascii")
    return ManifestArtifacts(
        output_directory=output,
        audit_manifest=audit_path,
        valid_manifest=valid_path,
        summary=summary_path,
        hashes={"audit_manifest": audit_hash, "valid_manifest": valid_hash, "validation_summary": summary_hash},
        counts=summary["counts"],
    )

