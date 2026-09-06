"""Deterministic, leakage-checked dataset splits and ID selections."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from zeromrg.utils.seed import derive_seed

from .schemas import canonical_json_bytes, sha256_bytes, write_json

SPLIT_VERSION = "deterministic_split_v1"
SELECTION_VERSION = "deterministic_selection_v1"
SPLIT_NAMES = ("train", "validation", "test")
_TIE_PRIORITY = {"test": 0, "validation": 1, "train": 2}


class SplitIntegrityError(ValueError):
    pass


def membership_id(record: Mapping[str, Any], dataset: str) -> str:
    """Use study UID for IU-Xray and stable sample ID for COV-CTR."""

    value = record.get("study_id") if dataset == "iu_xray" else record.get("sample_id")
    if value is None or str(value) == "":
        raise SplitIntegrityError(f"Missing split membership ID for {dataset}")
    return str(value)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SplitIntegrityError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def load_validated_population(dataset_directory: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Load a primary manifest only after validating its source audit relationship."""

    directory = Path(dataset_directory)
    summary_path = directory / "validation_summary.json"
    if not summary_path.is_file():
        raise SplitIntegrityError(f"Missing validation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = summary.get("dataset")
    valid_name = "valid_samples.jsonl" if dataset == "cov_ctr" else "valid_studies.jsonl"
    valid_path = directory / valid_name
    audit_path = directory / "audit_manifest.jsonl"
    for path in (valid_path, audit_path):
        if not path.is_file():
            raise SplitIntegrityError(f"Missing validated manifest: {path}")

    expected_valid_hash = summary.get("integrity", {}).get("valid_manifest_sha256")
    actual_valid_hash = file_sha256(valid_path)
    if expected_valid_hash != actual_valid_hash:
        raise SplitIntegrityError(
            f"Validated manifest hash mismatch for {valid_path}: expected {expected_valid_hash}, got {actual_valid_hash}"
        )
    expected_audit_hash = summary.get("integrity", {}).get("audit_manifest_sha256")
    actual_audit_hash = file_sha256(audit_path)
    if expected_audit_hash != actual_audit_hash:
        raise SplitIntegrityError(
            f"Audit manifest hash mismatch for {audit_path}: expected {expected_audit_hash}, got {actual_audit_hash}"
        )

    records = load_jsonl(valid_path)
    audit = load_jsonl(audit_path)
    ids = [str(record.get("sample_id", "")) for record in records]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SplitIntegrityError(f"Primary manifest contains missing or duplicate sample IDs: {valid_path}")
    if any(record.get("record_type") != "valid_sample" for record in records):
        raise SplitIntegrityError(f"Non-valid record found in primary manifest: {valid_path}")
    audited_valid_ids = {
        str(record.get("details", {}).get("sample_id"))
        for record in audit
        if record.get("record_type") == "dataset_audit" and record.get("status") == "valid"
    }
    if set(ids) != audited_valid_ids:
        raise SplitIntegrityError(
            "Primary manifest IDs do not exactly equal the validation audit's accepted sample IDs"
        )
    population_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "dataset": dataset,
                "samples": sorted(
                    (
                        {
                            "membership_id": membership_id(record, str(dataset)),
                            "sample_id": record["sample_id"],
                            "image_ids": record["image_ids"],
                        }
                        for record in records
                    ),
                    key=lambda item: item["membership_id"],
                ),
                "valid_manifest_sha256": actual_valid_hash,
            }
        )
    )
    return records, population_hash


def allocate_split_counts(population_size: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder allocation with a documented deterministic tie order."""

    if set(ratios) != set(SPLIT_NAMES):
        raise SplitIntegrityError(f"Split ratios must define exactly {SPLIT_NAMES}")
    numeric = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(value < 0 for value in numeric.values()) or not math.isclose(sum(numeric.values()), 1.0):
        raise SplitIntegrityError(f"Split ratios must be non-negative and sum to one: {numeric}")
    exact = {name: population_size * numeric[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(exact[name]) for name in SPLIT_NAMES}
    remaining = population_size - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), _TIE_PRIORITY[name]),
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def population_hash(records: Sequence[Mapping[str, Any]], dataset: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "dataset": dataset,
                "samples": sorted(
                    (
                        {
                            "membership_id": membership_id(record, dataset),
                            "sample_id": str(record["sample_id"]),
                            "image_ids": list(record["image_ids"]),
                        }
                        for record in records
                    ),
                    key=lambda item: item["membership_id"],
                ),
            }
        )
    )


def validate_split_membership(
    split: Mapping[str, Any], records: Sequence[Mapping[str, Any]], dataset: str
) -> None:
    expected_ids = {membership_id(record, dataset) for record in records}
    lists = {name: [str(value) for value in split[f"{name}_ids"]] for name in SPLIT_NAMES}
    for name, ids in lists.items():
        if len(ids) != len(set(ids)):
            raise SplitIntegrityError(f"Duplicate ID within {dataset} {name} split")
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = set(lists[left]) & set(lists[right])
            if overlap:
                raise SplitIntegrityError(f"Leakage between {left}/{right}: {sorted(overlap)[:10]}")
    actual_ids = set().union(*(set(ids) for ids in lists.values()))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise SplitIntegrityError(f"Split union differs from validated population; missing={missing}, extra={extra}")

    id_to_images = {membership_id(record, dataset): list(record["image_ids"]) for record in records}
    image_owner: dict[str, str] = {}
    for split_name, ids in lists.items():
        for sample_id in ids:
            for image_id in id_to_images[sample_id]:
                previous = image_owner.setdefault(str(image_id), split_name)
                if previous != split_name:
                    raise SplitIntegrityError(
                        f"Image/view leakage for {image_id}: {previous} and {split_name}"
                    )


def _embedded_hash(payload: Mapping[str, Any], hash_field: str) -> str:
    content = {key: value for key, value in payload.items() if key != hash_field}
    return sha256_bytes(canonical_json_bytes(content))


def _read_verified_artifact(path: Path, hash_field: str) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = artifact.get(hash_field)
    actual = _embedded_hash(artifact, hash_field)
    if recorded != actual:
        raise SplitIntegrityError(
            f"Existing artifact has invalid {hash_field}: {path}; expected {recorded}, computed {actual}"
        )
    return artifact


def build_or_reuse_splits(
    records: Sequence[Mapping[str, Any]],
    dataset: str,
    ratios: Mapping[str, float],
    base_seed: int,
    destination: str | Path,
    expected_counts: Mapping[str, int] | None = None,
    validated_population_hash: str | None = None,
    seed_namespace: str | None = None,
    execution_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    destination = Path(destination)
    ids = sorted(membership_id(record, dataset) for record in records)
    if len(ids) != len(set(ids)):
        raise SplitIntegrityError(f"Duplicate split membership IDs in {dataset} population")
    pop_hash = validated_population_hash or population_hash(records, dataset)
    derived = derive_seed(base_seed, seed_namespace or dataset, "split", SPLIT_VERSION)
    counts = allocate_split_counts(len(ids), ratios)
    if expected_counts and counts != {name: int(expected_counts[name]) for name in SPLIT_NAMES}:
        raise SplitIntegrityError(f"Computed split counts {counts} differ from configured expectations {expected_counts}")
    shuffled = list(ids)
    random.Random(derived).shuffle(shuffled)
    memberships = {
        "train_ids": sorted(shuffled[: counts["train"]]),
        "validation_ids": sorted(shuffled[counts["train"] : counts["train"] + counts["validation"]]),
        "test_ids": sorted(shuffled[counts["train"] + counts["validation"] :]),
    }
    train_membership_hash = sha256_bytes(
        canonical_json_bytes({"dataset": dataset, "train_ids": memberships["train_ids"]})
    )
    core = {
        "artifact_version": SPLIT_VERSION,
        "dataset": dataset,
        "base_seed": int(base_seed),
        "derived_split_seed": derived,
        "population_hash": pop_hash,
        "population_count": len(ids),
        "split_ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "counts": counts,
        "allocation": "largest_remainder_ties_test_validation_train",
        "training_membership_hash": train_membership_hash,
        **memberships,
    }
    if seed_namespace is not None:
        core["seed_namespace"] = seed_namespace
    if execution_profile is not None:
        core["execution_profile"] = dict(execution_profile)

    if destination.exists():
        existing = _read_verified_artifact(destination, "split_hash")
        comparable = {key: value for key, value in existing.items() if key not in {"generated_at_utc", "split_hash"}}
        if comparable != core:
            raise SplitIntegrityError(
                f"Existing split is incompatible and will not be silently replaced: {destination}"
            )
        validate_split_membership(existing, records, dataset)
        return existing, True, file_sha256(destination)

    artifact = {**core, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    artifact["split_hash"] = _embedded_hash(artifact, "split_hash")
    file_hash = write_json(artifact, destination)
    validate_split_membership(artifact, records, dataset)
    return artifact, False, file_hash


def build_or_reuse_selection(
    *,
    dataset: str,
    purpose: str,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
    count: int,
    base_seed: int,
    split_hash: str,
    destination: str | Path,
    seed_namespace: str | None = None,
    execution_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    destination = Path(destination)
    if count < 0 or count > len(train_ids):
        raise SplitIntegrityError(f"Cannot select {count} {purpose} IDs from {len(train_ids)} training IDs")
    derived = derive_seed(base_seed, seed_namespace or dataset, purpose, SELECTION_VERSION)
    candidates = sorted(str(value) for value in train_ids)
    chosen = sorted(random.Random(derived).sample(candidates, count))
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(set(train_ids)):
        raise SplitIntegrityError(f"Invalid {purpose} selection")
    leaked = set(chosen) & (set(validation_ids) | set(test_ids))
    if leaked:
        raise SplitIntegrityError(f"{purpose} selection leaked into validation/test: {sorted(leaked)[:10]}")
    core = {
        "artifact_version": SELECTION_VERSION,
        "dataset": dataset,
        "purpose": purpose,
        "base_seed": int(base_seed),
        "derived_seed": derived,
        "source_split_hash": split_hash,
        "count": count,
        "ids": chosen,
    }
    if seed_namespace is not None:
        core["seed_namespace"] = seed_namespace
    if execution_profile is not None:
        core["execution_profile"] = dict(execution_profile)
    if destination.exists():
        existing = _read_verified_artifact(destination, "artifact_hash")
        comparable = {key: value for key, value in existing.items() if key not in {"generated_at_utc", "artifact_hash"}}
        if comparable != core:
            raise SplitIntegrityError(
                f"Existing selection is incompatible and will not be silently replaced: {destination}"
            )
        return existing, True, file_sha256(destination)
    artifact = {**core, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    artifact["artifact_hash"] = _embedded_hash(artifact, "artifact_hash")
    file_hash = write_json(artifact, destination)
    return artifact, False, file_hash
