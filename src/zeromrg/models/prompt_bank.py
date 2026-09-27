"""Frozen, train-only report prompt embeddings with compatibility-checked caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from zeromrg.data.schemas import canonical_json_bytes, sha256_bytes, write_json

PROMPT_CACHE_VERSION = "zeromrg_prompt_bank_v1"


class PromptCacheError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hash(value: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes({key: item for key, item in value.items() if key != "artifact_hash"})
    )


def prompt_ids_hash(prompt_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes({"prompt_ids": [str(value) for value in prompt_ids]}))


def validate_prompt_ids(
    prompt_ids: Sequence[str], training_ids: Sequence[str], expected_count: int | None = None
) -> tuple[str, ...]:
    ordered = tuple(str(value) for value in prompt_ids)
    if len(ordered) != len(set(ordered)):
        raise PromptCacheError("Prompt IDs must be unique")
    if expected_count is not None and len(ordered) != int(expected_count):
        raise PromptCacheError(f"Expected {expected_count} prompts, found {len(ordered)}")
    leaked = set(ordered) - {str(value) for value in training_ids}
    if leaked:
        raise PromptCacheError(f"Prompt IDs are not training-only: {sorted(leaked)[:10]}")
    return ordered


def load_prompt_ids_artifact(
    prompt_path: str | Path, split_path: str | Path
) -> tuple[tuple[str, ...], dict[str, Any]]:
    prompt_artifact = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    if prompt_artifact.get("artifact_hash") != _artifact_hash(prompt_artifact):
        raise PromptCacheError(f"Prompt ID artifact has an invalid hash: {prompt_path}")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ids = validate_prompt_ids(
        prompt_artifact["ids"], split["train_ids"], int(prompt_artifact["count"])
    )
    if prompt_artifact.get("source_split_hash") != split.get("split_hash"):
        raise PromptCacheError("Prompt ID artifact was created from a different split")
    return ids, prompt_artifact


class PromptBank(nn.Module):
    """Prompt embeddings are a persistent non-parameter buffer and remain frozen."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        prompt_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Prompt embeddings must be non-empty [Np,d]")
        ids = tuple(str(value) for value in prompt_ids)
        if len(ids) != embeddings.shape[0] or len(ids) != len(set(ids)):
            raise ValueError("Prompt IDs must uniquely align one-to-one with embedding rows")
        normalized = F.normalize(embeddings.detach().float(), p=2, dim=-1)
        self.register_buffer("embeddings", normalized, persistent=True)
        self.prompt_ids = ids
        self.metadata = dict(metadata)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def count(self) -> int:
        return int(self.embeddings.shape[0])

    def forward(self) -> torch.Tensor:
        return self.embeddings

    @classmethod
    def from_reports(
        cls,
        *,
        prompt_ids: Sequence[str],
        training_ids: Sequence[str],
        reports_by_id: Mapping[str, str],
        text_encoder: nn.Module,
        dataset: str,
        profile: str,
        expected_count: int,
        cache_path: str | Path | None = None,
        batch_size: int = 16,
    ) -> "PromptBank":
        ids = validate_prompt_ids(prompt_ids, training_ids, expected_count)
        missing = [value for value in ids if value not in reports_by_id]
        if missing:
            raise PromptCacheError(f"Prompt reports are missing: {missing[:10]}")
        embedding_dim = int(getattr(text_encoder, "embedding_dim"))
        encoder_metadata = (
            text_encoder.cache_metadata() if hasattr(text_encoder, "cache_metadata") else {}
        )
        expected_metadata: dict[str, Any] = {
            "cache_version": PROMPT_CACHE_VERSION,
            "model_name": str(getattr(text_encoder, "model_name", type(text_encoder).__name__)),
            "dataset": dataset,
            "profile": profile,
            "prompt_ids_hash": prompt_ids_hash(ids),
            "number_prompts": len(ids),
            "embedding_dim": embedding_dim,
            "model_version_metadata": encoder_metadata,
        }
        if cache_path is not None:
            tensor_path = Path(cache_path)
            metadata_path = tensor_path.with_suffix(tensor_path.suffix + ".json")
            if tensor_path.exists() != metadata_path.exists():
                raise PromptCacheError("Prompt cache tensor/metadata pair is incomplete")
            if tensor_path.exists():
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if cached_metadata.get("artifact_hash") != _artifact_hash(cached_metadata):
                    raise PromptCacheError("Prompt cache metadata hash is invalid")
                comparable = {
                    key: value
                    for key, value in cached_metadata.items()
                    if key not in {"artifact_hash", "embedding_file_sha256"}
                }
                if comparable != expected_metadata:
                    raise PromptCacheError("Prompt cache metadata is incompatible with this run")
                if cached_metadata["embedding_file_sha256"] != _file_sha256(tensor_path):
                    raise PromptCacheError("Prompt cache embedding file hash is invalid")
                embeddings = torch.load(tensor_path, map_location="cpu", weights_only=True)
                if not isinstance(embeddings, torch.Tensor):
                    raise PromptCacheError("Prompt cache does not contain a tensor")
                bank = cls(embeddings, ids, cached_metadata)
                if bank.embedding_dim != embedding_dim or bank.count != len(ids):
                    raise PromptCacheError("Prompt cache tensor shape is incompatible")
                return bank

        if batch_size <= 0:
            raise ValueError("Prompt encoding batch_size must be positive")
        chunks = []
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            chunks.append(text_encoder([reports_by_id[value] for value in batch_ids]).detach().cpu())
        embeddings = F.normalize(torch.cat(chunks, dim=0).float(), p=2, dim=-1)
        if tuple(embeddings.shape) != (len(ids), embedding_dim):
            raise PromptCacheError(
                f"Encoded prompt shape {tuple(embeddings.shape)} != {(len(ids), embedding_dim)}"
            )
        metadata_payload = dict(expected_metadata)
        if cache_path is not None:
            tensor_path = Path(cache_path)
            metadata_path = tensor_path.with_suffix(tensor_path.suffix + ".json")
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
            torch.save(embeddings, temporary)
            temporary.replace(tensor_path)
            metadata_payload["embedding_file_sha256"] = _file_sha256(tensor_path)
            metadata_payload["artifact_hash"] = _artifact_hash(metadata_payload)
            write_json(metadata_payload, metadata_path)
        return cls(embeddings, ids, metadata_payload)
