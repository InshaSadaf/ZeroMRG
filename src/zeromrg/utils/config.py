"""Composable YAML configuration with explicit research-evidence labels."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise RuntimeError(
        "PyYAML is required to load ZeroMRG configuration. "
        "Install the dependencies listed in requirements.txt."
    ) from exc


EVIDENCE_LABELS = frozenset(
    {"PAPER-STATED", "IMPLEMENTATION-ASSUMPTION", "RESOURCE-CONSTRAINED"}
)
DEFAULT_OVERRIDE_LABEL = "IMPLEMENTATION-ASSUMPTION"


class ConfigError(ValueError):
    """Raised when a configuration is malformed or incompletely labelled."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ConfigError("Override path cannot be empty")
    cursor = config
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            cursor[part] = {}
        elif not isinstance(current, dict):
            raise ConfigError(
                f"Cannot apply override {dotted_path!r}: {part!r} is not a mapping"
            )
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _leaf_paths(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_paths(item, child)
    else:
        yield prefix


def _label_for(evidence: Mapping[str, str], dotted_path: str) -> str | None:
    candidate = dotted_path
    while candidate:
        if candidate in evidence:
            return evidence[candidate]
        candidate = candidate.rpartition(".")[0]
    return evidence.get("*")


def _validate_evidence(data: Mapping[str, Any], evidence: Mapping[str, str]) -> None:
    invalid = {path: label for path, label in evidence.items() if label not in EVIDENCE_LABELS}
    if invalid:
        raise ConfigError(f"Invalid evidence labels: {invalid}")
    missing = [path for path in _leaf_paths(data) if _label_for(evidence, path) is None]
    if missing:
        raise ConfigError(
            "Every configuration value must have an evidence label. Missing: "
            + ", ".join(missing)
        )


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return parsed


def _load_file(
    path: Path, active: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], dict[str, str], list[Path]]:
    path = path.expanduser().resolve()
    if path in active:
        chain = " -> ".join(str(item) for item in (*active, path))
        raise ConfigError(f"Cyclic configuration composition: {chain}")

    document = dict(_read_yaml(path))
    bases = document.pop("_base_", [])
    evidence = document.pop("_evidence", {})
    if isinstance(bases, (str, Path)):
        bases = [bases]
    if not isinstance(bases, Sequence) or isinstance(bases, (bytes, bytearray)):
        raise ConfigError(f"_base_ must be a path or list of paths in {path}")
    if not isinstance(evidence, Mapping):
        raise ConfigError(f"_evidence must be a mapping in {path}")
    # Require every file to label the values it introduces or replaces. This
    # prevents an overlay from silently inheriting a stale paper/assumption label.
    _validate_evidence(document, {str(key): str(value) for key, value in evidence.items()})

    merged_data: dict[str, Any] = {}
    merged_evidence: dict[str, str] = {}
    sources: list[Path] = []
    for base in bases:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_data, base_evidence, base_sources = _load_file(base_path, (*active, path))
        merged_data = _deep_merge(merged_data, base_data)
        merged_evidence.update(base_evidence)
        for source in base_sources:
            if source not in sources:
                sources.append(source)

    merged_data = _deep_merge(merged_data, document)
    merged_evidence.update({str(key): str(value) for key, value in evidence.items()})
    if path not in sources:
        sources.append(path)
    return merged_data, merged_evidence, sources


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``dotted.path=YAML_VALUE`` from a command line."""

    if "=" not in text:
        raise ConfigError(f"Override must use dotted.path=value syntax: {text!r}")
    path, raw_value = text.split("=", 1)
    path = path.strip()
    if not path:
        raise ConfigError("Override path cannot be empty")
    return path, yaml.safe_load(raw_value)


@dataclass(frozen=True)
class ResolvedConfig(Mapping[str, Any]):
    """Resolved settings plus provenance labels and composition sources."""

    data: dict[str, Any]
    evidence: dict[str, str]
    sources: tuple[str, ...]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def label_for(self, dotted_path: str) -> str:
        label = _label_for(self.evidence, dotted_path)
        if label is None:
            raise ConfigError(f"No evidence label for configuration path: {dotted_path}")
        return label

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        output = copy.deepcopy(self.data)
        if include_metadata:
            output["_evidence"] = copy.deepcopy(self.evidence)
            output["_sources"] = list(self.sources)
        return output

    def digest(self) -> str:
        # Source locations are provenance, but are intentionally excluded from
        # the scientific config hash so moving the same checkout does not alter it.
        payload = json.dumps(
            _jsonable({"data": self.data, "evidence": self.evidence}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(
    config_files: str | Path | Sequence[str | Path],
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    override_labels: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Compose YAML files in order and apply explicit dotted-path overrides.

    Later files override earlier files. Each file can also declare ``_base_``.
    Command-line-style overrides are labelled ``IMPLEMENTATION-ASSUMPTION`` by
    default; callers may supply explicit labels through ``override_labels``.
    """

    if isinstance(config_files, (str, Path)):
        config_files = [config_files]
    if not config_files:
        raise ConfigError("At least one configuration file is required")

    data: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    sources: list[Path] = []
    for filename in config_files:
        file_data, file_evidence, file_sources = _load_file(Path(filename))
        data = _deep_merge(data, file_data)
        evidence.update(file_evidence)
        for source in file_sources:
            if source not in sources:
                sources.append(source)

    parsed_overrides: dict[str, Any] = {}
    if isinstance(overrides, Mapping):
        parsed_overrides.update(overrides)
    elif overrides:
        for item in overrides:
            key, value = parse_override(item)
            parsed_overrides[key] = value

    labels = dict(override_labels or {})
    for dotted_path, value in parsed_overrides.items():
        _set_dotted(data, dotted_path, value)
        for evidence_path in list(evidence):
            if evidence_path == dotted_path or evidence_path.startswith(dotted_path + "."):
                evidence.pop(evidence_path)
        evidence[dotted_path] = labels.get(dotted_path, DEFAULT_OVERRIDE_LABEL)

    _validate_evidence(data, evidence)
    return ResolvedConfig(
        data=copy.deepcopy(data),
        evidence=copy.deepcopy(evidence),
        sources=tuple(str(source) for source in sources),
    )


def resolve_paths(config: Mapping[str, Any], project_root: str | Path | None = None) -> dict[str, Any]:
    """Return the configured path tree with relative values made absolute."""

    path_tree = config.get("paths", {})
    if not isinstance(path_tree, Mapping):
        raise ConfigError("paths must be a mapping")
    root = Path(project_root or Path.cwd()).resolve()

    def resolve(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if value is None or not isinstance(value, str) or not value.strip():
            return value
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (root / path).resolve())

    return resolve(path_tree)


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping for readable diagnostics."""

    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, name))
        else:
            flattened[name] = item
    return flattened
