"""Orchestrate deterministic splits, normalized manifests, and decoder text assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .report_text import (
    NORMALIZATION_VERSION,
    normalize_cov_ctr_target,
    normalize_iu_xray_target,
)
from .schemas import SplitAwareSample, canonical_json_bytes, sha256_bytes, write_json, write_jsonl
from .split import (
    SPLIT_NAMES,
    SplitIntegrityError,
    build_or_reuse_selection,
    build_or_reuse_splits,
    file_sha256,
    load_validated_population,
    membership_id,
)
from .vocabulary import (
    TOKENIZER_VERSION,
    DecoderVocabulary,
    build_vocabulary_artifact,
    tokenize,
    write_or_reuse_vocabulary,
)

PREPARATION_VERSION = "text_preparation_v1"


@dataclass(frozen=True)
class TextPreparationArtifacts:
    dataset: str
    directory: Path
    split: Path
    paired_10: Path
    prompt_ids: Path
    normalized_manifest: Path
    vocabulary: Path
    report_statistics: Path
    counts: dict[str, int]
    paired_10_count: int
    prompt_count: int
    vocabulary_size: int
    max_sequence_length: int
    reports_over_configured_max: int
    hashes: dict[str, str]
    reused: dict[str, bool]


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _normalize(dataset: str, record: Mapping[str, Any]) -> str:
    if dataset == "cov_ctr":
        return normalize_cov_ctr_target(record)
    if dataset == "iu_xray":
        return normalize_iu_xray_target(record)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _raw_target_fields(dataset: str, record: Mapping[str, Any]) -> dict[str, str]:
    fields = record["report_fields"]
    if dataset == "cov_ctr":
        return {"reports_En": str(fields["reports_En"])}
    return {"impression": str(fields["impression"]), "findings": str(fields["findings"])}


def _manifest_records(
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    source_manifest_hash: str,
) -> tuple[list[SplitAwareSample], dict[str, str], int]:
    id_to_split = {
        str(value): name
        for name in SPLIT_NAMES
        for value in split[f"{name}_ids"]
    }
    normalized: list[SplitAwareSample] = []
    normalized_text: dict[str, str] = {}
    changed_count = 0
    for record in sorted(records, key=lambda item: membership_id(item, dataset)):
        member_id = membership_id(record, dataset)
        target = _normalize(dataset, record)
        raw_fields = _raw_target_fields(dataset, record)
        if dataset == "cov_ctr":
            changed_count += target != raw_fields["reports_En"]
        else:
            raw_join = raw_fields["impression"] + " " + raw_fields["findings"]
            changed_count += target != raw_join
        normalized_text[member_id] = target
        normalized.append(
            SplitAwareSample(
                dataset=dataset,
                sample_id=str(record["sample_id"]),
                membership_id=member_id,
                report_id=str(record["report_id"]),
                study_id=None if record.get("study_id") is None else str(record["study_id"]),
                split=id_to_split[member_id],
                source_image_paths=[str(value) for value in record["source_image_paths"]],
                image_ids=[str(value) for value in record["image_ids"]],
                view_metadata=[dict(value) for value in record.get("image_metadata", [])],
                raw_target_fields=raw_fields,
                normalized_target_report=target,
                normalization_version=NORMALIZATION_VERSION,
                source_valid_manifest_sha256=source_manifest_hash,
                validity={"primary_validated": True, "excluded": False, "conflicting": False},
                provenance={
                    "preparation_version": PREPARATION_VERSION,
                    "population_hash": split["population_hash"],
                    "split_hash": split["split_hash"],
                },
            )
        )
    return normalized, normalized_text, changed_count


def _report_statistics(
    dataset: str,
    normalized_text: Mapping[str, str],
    split: Mapping[str, Any],
    vocabulary: DecoderVocabulary,
    vocabulary_hash: str,
    configured_max: int,
    normalization_changed_count: int,
    dependency_hashes: Mapping[str, str],
) -> dict[str, Any]:
    split_lookup = {
        str(value): name
        for name in SPLIT_NAMES
        for value in split[f"{name}_ids"]
    }
    train_ids = [str(value) for value in split["train_ids"]]
    train_char_lengths = [len(normalized_text[value]) for value in train_ids]
    train_whitespace_lengths = [len(normalized_text[value].split()) for value in train_ids]
    train_token_lengths = [len(tokenize(normalized_text[value])) for value in train_ids]
    sequence_lengths = {
        member_id: len(vocabulary.decoder_pair(text)[0])
        for member_id, text in normalized_text.items()
    }
    content_lengths = {member_id: len(tokenize(text)) for member_id, text in normalized_text.items()}
    over_limit = sorted(
        member_id for member_id, length in sequence_lengths.items() if length > configured_max
    )
    longest_by_split: dict[str, dict[str, Any]] = {}
    unknown_by_split = {name: 0 for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        ids = [str(value) for value in split[f"{name}_ids"]]
        maximum = max(sequence_lengths[value] for value in ids)
        longest_by_split[name] = {
            "maximum_including_bos_eos": maximum,
            "sample_ids": sorted(value for value in ids if sequence_lengths[value] == maximum),
        }
        for member_id in ids:
            unknown_by_split[name] += vocabulary.encode(normalized_text[member_id]).count(
                vocabulary.unk_id
            )

    artifact: dict[str, Any] = {
        "artifact_version": "report_statistics_v1",
        "dataset": dataset,
        "normalization_version": NORMALIZATION_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "split_hash": split["split_hash"],
        "vocabulary_artifact_hash": vocabulary_hash,
        "dependency_file_hashes": dict(dependency_hashes),
        "training_reports": len(train_ids),
        "all_reports": len(normalized_text),
        "training_character_lengths": _distribution(train_char_lengths),
        "training_whitespace_token_lengths": _distribution(train_whitespace_lengths),
        "training_decoder_content_token_lengths": _distribution(train_token_lengths),
        "all_decoder_content_token_lengths": _distribution(list(content_lengths.values())),
        "all_decoder_sequence_lengths_including_bos_eos": _distribution(
            list(sequence_lengths.values())
        ),
        "longest_by_split": longest_by_split,
        "configured_max_report_tokens": configured_max,
        "reports_exceeding_configured_max": len(over_limit),
        "percentage_exceeding_configured_max": 100.0 * len(over_limit) / len(normalized_text),
        "affected_sample_ids": over_limit,
        "minimum_safe_max_report_tokens": max(sequence_lengths.values()),
        "max_length_decision": (
            f"retain_{configured_max}" if not over_limit else "configuration_update_required_before_training"
        ),
        "normalization_changed_reports": normalization_changed_count,
        "unknown_token_counts_by_split": unknown_by_split,
        "note": "No report was truncated; sequence lengths include BOS and EOS.",
    }
    artifact["artifact_hash"] = sha256_bytes(canonical_json_bytes(artifact))
    return artifact


def prepare_text_data(
    dataset: str,
    processed_root: str | Path,
    config: Mapping[str, Any],
) -> TextPreparationArtifacts:
    directory = Path(processed_root) / dataset
    records, pop_hash = load_validated_population(directory)
    dataset_config = config["dataset"]
    if config["text_preparation"]["normalization_version"] != NORMALIZATION_VERSION:
        raise SplitIntegrityError(
            "Configured normalization version does not match the implemented normalizer"
        )
    if config["text_preparation"]["tokenizer_version"] != TOKENIZER_VERSION:
        raise SplitIntegrityError(
            "Configured tokenizer version does not match the implemented tokenizer"
        )
    expected_population = int(dataset_config["primary_usable_samples"])
    if len(records) != expected_population:
        raise SplitIntegrityError(
            f"Validated {dataset} population has {len(records)} records; configured primary population is {expected_population}"
        )

    base_seed = int(config["reproducibility"]["seed"])
    split_path = directory / "splits.json"
    split, split_reused, split_file_hash = build_or_reuse_splits(
        records=records,
        dataset=dataset,
        ratios=dataset_config["split"],
        base_seed=base_seed,
        destination=split_path,
        expected_counts=dataset_config["expected_split_counts"],
        validated_population_hash=pop_hash,
    )
    valid_name = "valid_samples.jsonl" if dataset == "cov_ctr" else "valid_studies.jsonl"
    source_manifest_hash = file_sha256(directory / valid_name)
    manifest_records, normalized_text, normalization_changed = _manifest_records(
        dataset, records, split, source_manifest_hash
    )
    normalized_name = "samples_with_splits.jsonl" if dataset == "cov_ctr" else "studies_with_splits.jsonl"
    normalized_path = directory / normalized_name
    normalized_hash = write_jsonl(manifest_records, normalized_path)

    training_reports = {
        member_id: normalized_text[member_id] for member_id in split["train_ids"]
    }
    vocabulary, vocabulary_artifact = build_vocabulary_artifact(
        training_reports,
        training_split_hash=split["training_membership_hash"],
        normalization_version=str(config["text_preparation"]["normalization_version"]),
    )
    vocabulary_path = directory / "vocabulary.json"
    vocabulary_artifact, vocabulary_reused, vocabulary_file_hash = write_or_reuse_vocabulary(
        vocabulary_artifact, vocabulary_path
    )

    paired_path = directory / "paired_10_ids.json"
    paired, paired_reused, paired_file_hash = build_or_reuse_selection(
        dataset=dataset,
        purpose="paired_10",
        train_ids=split["train_ids"],
        validation_ids=split["validation_ids"],
        test_ids=split["test_ids"],
        count=int(dataset_config["paired_10_count"]),
        base_seed=base_seed,
        split_hash=split["split_hash"],
        destination=paired_path,
    )
    prompt_path = directory / "prompt_ids.json"
    prompts, prompt_reused, prompt_file_hash = build_or_reuse_selection(
        dataset=dataset,
        purpose="prompt_ids",
        train_ids=split["train_ids"],
        validation_ids=split["validation_ids"],
        test_ids=split["test_ids"],
        count=int(dataset_config["prompt_count"]),
        base_seed=base_seed,
        split_hash=split["split_hash"],
        destination=prompt_path,
    )

    configured_max = int(config["text_preparation"]["max_report_tokens"])
    statistics = _report_statistics(
        dataset,
        normalized_text,
        split,
        vocabulary,
        str(vocabulary_artifact["artifact_hash"]),
        configured_max,
        normalization_changed,
        {
            "splits": split_file_hash,
            "normalized_manifest": normalized_hash,
            "vocabulary": vocabulary_file_hash,
            "paired_10_ids": paired_file_hash,
            "prompt_ids": prompt_file_hash,
        },
    )
    statistics_path = directory / "report_statistics.json"
    statistics_file_hash = write_json(statistics, statistics_path)
    return TextPreparationArtifacts(
        dataset=dataset,
        directory=directory,
        split=split_path,
        paired_10=paired_path,
        prompt_ids=prompt_path,
        normalized_manifest=normalized_path,
        vocabulary=vocabulary_path,
        report_statistics=statistics_path,
        counts={name: int(split["counts"][name]) for name in SPLIT_NAMES},
        paired_10_count=int(paired["count"]),
        prompt_count=int(prompts["count"]),
        vocabulary_size=int(vocabulary_artifact["vocabulary_size"]),
        max_sequence_length=int(
            statistics["all_decoder_sequence_lengths_including_bos_eos"]["maximum"]
        ),
        reports_over_configured_max=int(statistics["reports_exceeding_configured_max"]),
        hashes={
            "splits": split_file_hash,
            "normalized_manifest": normalized_hash,
            "vocabulary": vocabulary_file_hash,
            "paired_10_ids": paired_file_hash,
            "prompt_ids": prompt_file_hash,
            "report_statistics": statistics_file_hash,
        },
        reused={
            "splits": split_reused,
            "vocabulary": vocabulary_reused,
            "paired_10_ids": paired_reused,
            "prompt_ids": prompt_reused,
        },
    )
