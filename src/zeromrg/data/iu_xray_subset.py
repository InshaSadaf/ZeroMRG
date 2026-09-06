"""Deterministic, profile-isolated preparation for the IU-Xray-500 subset."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from zeromrg.utils.seed import derive_seed

from .prepare_text import _manifest_records, _report_statistics
from .report_text import NORMALIZATION_VERSION
from .schemas import canonical_json_bytes, sha256_bytes, write_json, write_jsonl
from .split import (
    SPLIT_NAMES,
    SplitIntegrityError,
    build_or_reuse_selection,
    build_or_reuse_splits,
    file_sha256,
    load_validated_population,
    membership_id,
    population_hash,
)
from .vocabulary import TOKENIZER_VERSION, build_vocabulary_artifact, write_or_reuse_vocabulary

SUBSET_VERSION = "iu_xray_uid_subset_v1"
PROFILE_NAME = "kaggle_500"
PROFILE_LABEL = "RESOURCE-CONSTRAINED"


@dataclass(frozen=True)
class IuXraySubsetArtifacts:
    subset_ids: Path
    directory: Path
    split: Path
    paired_10: Path
    prompt_ids: Path
    normalized_manifest: Path
    vocabulary: Path
    report_statistics: Path
    preparation_summary: Path
    selected_records: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    hashes: dict[str, str]
    reused: dict[str, bool]


def _artifact_hash(value: Mapping[str, Any], field: str = "artifact_hash") -> str:
    return sha256_bytes(canonical_json_bytes({key: item for key, item in value.items() if key != field}))


def _uid_order(uid: str) -> tuple[bool, int | str]:
    return (not uid.isdigit(), int(uid) if uid.isdigit() else uid)


def _view_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(len(record["image_ids"])) for record in records).items()))


def build_or_reuse_iu_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    source_population_hash: str,
    source_manifest_sha256: str,
    population_size: int,
    base_seed: int,
    destination: str | Path,
    profile_name: str = PROFILE_NAME,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], bool, str]:
    """Select UIDs with a local RNG while retaining every view of each UID."""

    destination = Path(destination)
    by_uid: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = dict(raw_record)
        uid = membership_id(record, "iu_xray")
        if uid in by_uid:
            raise SplitIntegrityError(f"Duplicate IU-Xray UID in source population: {uid}")
        by_uid[uid] = record
    if population_size <= 0 or population_size > len(by_uid):
        raise SplitIntegrityError(
            f"Cannot select {population_size} UIDs from validated population of {len(by_uid)}"
        )

    derived_seed = derive_seed(base_seed, "iu_xray", profile_name, "subset", SUBSET_VERSION)
    selected_ids = sorted(
        random.Random(derived_seed).sample(sorted(by_uid, key=_uid_order), population_size),
        key=_uid_order,
    )
    selected = tuple(by_uid[uid] for uid in selected_ids)
    core: dict[str, Any] = {
        "artifact_version": SUBSET_VERSION,
        "dataset": "iu_xray",
        "execution_profile": {"name": profile_name, "label": PROFILE_LABEL},
        "selection_level": "uid",
        "selection_algorithm": "python_random_sample_sorted_uid_population",
        "ordered_id_policy": "numeric_uid_ascending_then_lexical",
        "base_seed": int(base_seed),
        "derived_seed": derived_seed,
        "source_population_count": len(by_uid),
        "source_population_hash": source_population_hash,
        "source_valid_manifest_sha256": source_manifest_sha256,
        "selected_uid_count": len(selected_ids),
        "selected_uids": selected_ids,
        "physical_image_count": sum(len(record["image_ids"]) for record in selected),
        "views_per_uid": _view_distribution(selected),
        "all_views_retained": True,
    }
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("artifact_hash") != _artifact_hash(existing):
            raise SplitIntegrityError(f"Existing subset artifact has an invalid hash: {destination}")
        comparable = {key: value for key, value in existing.items() if key not in {"generated_at_utc", "artifact_hash"}}
        if comparable != core:
            raise SplitIntegrityError(
                f"Existing subset selection is incompatible and will not be replaced: {destination}"
            )
        return existing, selected, True, file_sha256(destination)

    artifact = {**core, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    artifact["artifact_hash"] = _artifact_hash(artifact)
    artifact_file_hash = write_json(artifact, destination)
    return artifact, selected, False, artifact_file_hash


def prepare_iu_xray_kaggle_500(
    processed_root: str | Path, config: Mapping[str, Any]
) -> IuXraySubsetArtifacts:
    """Create subset-only splits, normalized text, vocabulary, and selections."""

    if config["execution_profile"]["name"] != PROFILE_NAME:
        raise SplitIntegrityError("IU-Xray subset preparation requires the kaggle_500 profile")
    if config["execution_profile"]["label"] != PROFILE_LABEL:
        raise SplitIntegrityError("IU-Xray subset profile must be labelled RESOURCE-CONSTRAINED")
    if config["text_preparation"]["normalization_version"] != NORMALIZATION_VERSION:
        raise SplitIntegrityError("Configured normalization version does not match implementation")
    if config["text_preparation"]["tokenizer_version"] != TOKENIZER_VERSION:
        raise SplitIntegrityError("Configured tokenizer version does not match implementation")

    iu_root = Path(processed_root) / "iu_xray"
    records, source_population_hash = load_validated_population(iu_root)
    expected_source = int(config["subset"]["source_population"])
    if len(records) != expected_source:
        raise SplitIntegrityError(
            f"Validated IU-Xray population is {len(records)}; profile requires {expected_source}"
        )
    source_manifest = iu_root / "valid_studies.jsonl"
    source_manifest_hash = file_sha256(source_manifest)
    subset_path = iu_root / "subset_500_ids.json"
    subset, selected, subset_reused, subset_file_hash = build_or_reuse_iu_subset(
        records,
        source_population_hash=source_population_hash,
        source_manifest_sha256=source_manifest_hash,
        population_size=int(config["subset"]["population_size"]),
        base_seed=int(config["subset"]["base_seed"]),
        destination=subset_path,
    )

    profile_dir = Path(processed_root) / str(config["execution_profile"]["output_namespace"])
    profile = {"name": PROFILE_NAME, "label": PROFILE_LABEL}
    subset_population_hash = population_hash(selected, "iu_xray")
    ratios = {
        "train": float(config["split"]["train_ratio"]),
        "validation": float(config["split"]["validation_ratio"]),
        "test": float(config["split"]["test_ratio"]),
    }
    split_path = profile_dir / "splits.json"
    split, split_reused, split_file_hash = build_or_reuse_splits(
        records=selected,
        dataset="iu_xray",
        ratios=ratios,
        base_seed=int(config["subset"]["base_seed"]),
        destination=split_path,
        expected_counts={name: int(config["split"][f"{name}_count"]) for name in SPLIT_NAMES},
        validated_population_hash=subset_population_hash,
        seed_namespace="iu_xray:kaggle_500",
        execution_profile=profile,
    )
    normalized, normalized_text, changed = _manifest_records(
        "iu_xray", selected, split, source_manifest_hash
    )
    manifest_path = profile_dir / "studies_with_splits.jsonl"
    manifest_hash = write_jsonl(normalized, manifest_path)

    training_reports = {uid: normalized_text[uid] for uid in split["train_ids"]}
    vocabulary, vocabulary_artifact = build_vocabulary_artifact(
        training_reports,
        training_split_hash=split["training_membership_hash"],
        normalization_version=NORMALIZATION_VERSION,
    )
    vocabulary_artifact["execution_profile"] = profile
    vocabulary_artifact["source_subset_artifact_hash"] = subset["artifact_hash"]
    vocabulary_artifact["artifact_hash"] = _artifact_hash(vocabulary_artifact)
    vocabulary_path = profile_dir / "vocabulary.json"
    vocabulary_artifact, vocabulary_reused, vocabulary_file_hash = write_or_reuse_vocabulary(
        vocabulary_artifact, vocabulary_path
    )

    selection_args = dict(
        dataset="iu_xray",
        train_ids=split["train_ids"],
        validation_ids=split["validation_ids"],
        test_ids=split["test_ids"],
        base_seed=int(config["subset"]["base_seed"]),
        split_hash=split["split_hash"],
        seed_namespace="iu_xray:kaggle_500",
        execution_profile=profile,
    )
    paired_path = profile_dir / "paired_10_ids.json"
    paired, paired_reused, paired_file_hash = build_or_reuse_selection(
        **selection_args,
        purpose="paired_10",
        count=int(config["paired"]["ten_percent_count"]),
        destination=paired_path,
    )
    prompt_path = profile_dir / "prompt_ids.json"
    prompts, prompts_reused, prompt_file_hash = build_or_reuse_selection(
        **selection_args,
        purpose="prompt_ids",
        count=int(config["prompts"]["count"]),
        destination=prompt_path,
    )

    dependency_hashes = {
        "subset_500_ids": subset_file_hash,
        "splits": split_file_hash,
        "normalized_manifest": manifest_hash,
        "vocabulary": vocabulary_file_hash,
        "paired_10_ids": paired_file_hash,
        "prompt_ids": prompt_file_hash,
    }
    statistics = _report_statistics(
        "iu_xray",
        normalized_text,
        split,
        vocabulary,
        str(vocabulary_artifact["artifact_hash"]),
        int(config["text_preparation"]["max_report_tokens"]),
        changed,
        dependency_hashes,
    )
    statistics["execution_profile"] = profile
    statistics["source_subset_artifact_hash"] = subset["artifact_hash"]
    statistics["artifact_hash"] = _artifact_hash(statistics)
    statistics_path = profile_dir / "report_statistics.json"
    statistics_file_hash = write_json(statistics, statistics_path)
    summary: dict[str, Any] = {
        "artifact_version": "iu_xray_kaggle_500_preparation_v1",
        "dataset": "iu_xray",
        "execution_profile": profile,
        "selected_uid_count": len(selected),
        "physical_image_count": int(subset["physical_image_count"]),
        "views_per_uid": subset["views_per_uid"],
        "split_counts": split["counts"],
        "paired_10_count": paired["count"],
        "prompt_count": prompts["count"],
        "vocabulary_training_report_count": vocabulary_artifact["training_report_count"],
        "vocabulary_size": vocabulary_artifact["vocabulary_size"],
        "reports_exceeding_configured_max": statistics["reports_exceeding_configured_max"],
        "dependency_file_hashes": {**dependency_hashes, "report_statistics": statistics_file_hash},
    }
    summary["artifact_hash"] = _artifact_hash(summary)
    summary_path = profile_dir / "preparation_summary.json"
    summary_file_hash = write_json(summary, summary_path)
    return IuXraySubsetArtifacts(
        subset_ids=subset_path,
        directory=profile_dir,
        split=split_path,
        paired_10=paired_path,
        prompt_ids=prompt_path,
        normalized_manifest=manifest_path,
        vocabulary=vocabulary_path,
        report_statistics=statistics_path,
        preparation_summary=summary_path,
        selected_records=selected,
        counts={name: int(split["counts"][name]) for name in SPLIT_NAMES},
        hashes={**dependency_hashes, "report_statistics": statistics_file_hash, "preparation_summary": summary_file_hash},
        reused={
            "subset": subset_reused,
            "splits": split_reused,
            "vocabulary": vocabulary_reused,
            "paired_10_ids": paired_reused,
            "prompt_ids": prompts_reused,
        },
    )
