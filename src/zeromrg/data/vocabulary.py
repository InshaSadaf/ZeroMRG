"""Train-only decoder vocabulary and deterministic word/punctuation tokenizer."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import canonical_json_bytes, sha256_bytes, write_json
from .split import SplitIntegrityError, file_sha256

TOKENIZER_VERSION = "word_punctuation_v1"
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)
_NO_SPACE_BEFORE = frozenset(".,;:!?%)]}")
_NO_SPACE_AFTER = frozenset("([{\u00a3$#")
_JOIN_BOTH_SIDES = frozenset({"-", "/"})


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("tokenizer input must be a string")
    return _TOKEN_PATTERN.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    """Produce deterministic readable text from word/punctuation tokens."""

    output = ""
    previous = ""
    for token in tokens:
        if not output:
            output = token
        elif (
            token in _NO_SPACE_BEFORE
            or previous in _NO_SPACE_AFTER
            or token in _JOIN_BOTH_SIDES
            or previous in _JOIN_BOTH_SIDES
        ):
            output += token
        else:
            output += " " + token
        previous = token
    return output


def _artifact_hash(artifact: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes({key: value for key, value in artifact.items() if key != "artifact_hash"})
    )


@dataclass(frozen=True)
class DecoderVocabulary:
    id_to_token: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(self.id_to_token[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"Vocabulary must start with special tokens {SPECIAL_TOKENS}")
        if len(self.id_to_token) != len(set(self.id_to_token)):
            raise ValueError("Vocabulary tokens must be unique")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        lookup = self.token_to_id
        return [lookup.get(token, self.unk_id) for token in tokens]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.encode_tokens(tokenize(text))
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Iterable[int], skip_boundary_tokens: bool = True) -> str:
        tokens: list[str] = []
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise ValueError(f"Token ID outside vocabulary: {token_id}")
            token = self.id_to_token[token_id]
            if skip_boundary_tokens and token in {"<pad>", "<bos>", "<eos>"}:
                continue
            tokens.append(token)
        return detokenize(tokens)

    def decoder_pair(self, text: str) -> tuple[list[int], list[int]]:
        content = self.encode(text)
        return [self.bos_id, *content], [*content, self.eos_id]

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "DecoderVocabulary":
        if artifact.get("artifact_hash") != _artifact_hash(artifact):
            raise ValueError("Vocabulary artifact hash is invalid")
        return cls(tuple(str(token) for token in artifact["id_to_token"]))


def build_vocabulary_artifact(
    training_reports: Mapping[str, str],
    training_split_hash: str,
    normalization_version: str,
) -> tuple[DecoderVocabulary, dict[str, Any]]:
    counts: Counter[str] = Counter()
    for report_id in sorted(training_reports):
        counts.update(tokenize(training_reports[report_id]))
    ordinary = sorted(counts, key=lambda token: (-counts[token], token))
    vocabulary = DecoderVocabulary((*SPECIAL_TOKENS, *ordinary))
    token_to_id = vocabulary.token_to_id
    training_ids_hash = sha256_bytes(
        canonical_json_bytes({"training_report_ids": sorted(training_reports)})
    )
    artifact: dict[str, Any] = {
        "artifact_version": "decoder_vocabulary_v1",
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_config": {
            "pattern": _TOKEN_PATTERN.pattern,
            "unicode": True,
            "case_preserving": True,
            "punctuation_separated": True,
            "detokenization": "punctuation_attachment_v1",
        },
        "normalization_version": normalization_version,
        "training_split_hash": training_split_hash,
        "training_report_ids_hash": training_ids_hash,
        "training_report_count": len(training_reports),
        "id_to_token": list(vocabulary.id_to_token),
        "token_to_id": token_to_id,
        "special_token_ids": {
            "pad": vocabulary.pad_id,
            "bos": vocabulary.bos_id,
            "eos": vocabulary.eos_id,
            "unk": vocabulary.unk_id,
        },
        "vocabulary_size": len(vocabulary.id_to_token),
        "token_frequencies": {token: counts[token] for token in ordinary},
    }
    artifact["artifact_hash"] = _artifact_hash(artifact)
    return vocabulary, artifact


def write_or_reuse_vocabulary(
    artifact: Mapping[str, Any], destination: str | Path
) -> tuple[dict[str, Any], bool, str]:
    destination = Path(destination)
    desired = dict(artifact)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("artifact_hash") != _artifact_hash(existing):
            raise SplitIntegrityError(f"Existing vocabulary has an invalid hash: {destination}")
        if existing != desired:
            raise SplitIntegrityError(
                f"Existing vocabulary is incompatible and will not be silently replaced: {destination}"
            )
        return existing, True, file_sha256(destination)
    file_hash = write_json(desired, destination)
    return desired, False, file_hash


def pad_token_sequences(
    sequences: Sequence[Sequence[int]], pad_id: int, length: int | None = None
) -> tuple[list[list[int]], list[list[bool]]]:
    if not sequences:
        return [], []
    target_length = max(len(sequence) for sequence in sequences) if length is None else int(length)
    if target_length < max(len(sequence) for sequence in sequences):
        raise ValueError("Requested padding length would truncate a sequence")
    padded = [list(sequence) + [pad_id] * (target_length - len(sequence)) for sequence in sequences]
    masks = [[index < len(sequence) for index in range(target_length)] for sequence in sequences]
    return padded, masks
