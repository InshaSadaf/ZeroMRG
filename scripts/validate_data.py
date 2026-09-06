#!/usr/bin/env python
"""Validate configured COV-CTR/IU-Xray sources and build audit manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.data import build_cov_ctr_manifest, build_iu_xray_manifest
from zeromrg.data.archive_io import DatasetSourceError
from zeromrg.utils.config import load_config, resolve_paths
from zeromrg.utils.provenance import detect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cov_ctr", "iu_xray", "all"))
    parser.add_argument(
        "--config", action="append", default=[], help="Additional YAML overlay; repeatable"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration override in dotted.path=YAML_VALUE form; repeatable",
    )
    parser.add_argument("--cov-ctr-path", help="Override the configured COV-CTR source")
    parser.add_argument("--iu-xray-path", help="Override the configured IU-Xray source")
    return parser.parse_args()


def _available_kaggle_inputs() -> list[str]:
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    return [str(path) for path in sorted(root.iterdir(), key=lambda item: item.name.lower())]


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _path_error(dataset: str, configured: object, environment: str) -> ValueError:
    available = _available_kaggle_inputs()
    message = f"{dataset} dataset path is not configured or does not exist: {configured!r}."
    if environment == "KAGGLE":
        listing = "\n".join(f"  {item}" for item in available) or "  (none found)"
        message += f"\nAvailable /kaggle/input entries:\n{listing}"
    return ValueError(message)


def _config_for(dataset: str, environment: str, args: argparse.Namespace):
    files: list[Path] = [PROJECT_ROOT / "configs" / f"{dataset}.yaml"]
    if environment == "KAGGLE":
        files.append(PROJECT_ROOT / "configs" / "kaggle.yaml")
    elif environment == "LOCAL":
        files.append(PROJECT_ROOT / "configs" / "local.yaml")
    files.extend(Path(path) for path in args.config)
    overrides = list(args.override)
    if args.cov_ctr_path:
        overrides.append(f"paths.datasets.cov_ctr={args.cov_ctr_path}")
    if args.iu_xray_path:
        overrides.append(f"paths.datasets.iu_xray={args.iu_xray_path}")
    return load_config(files, overrides=overrides)


def _validate_one(dataset: str, environment: str, args: argparse.Namespace):
    config = _config_for(dataset, environment, args)
    paths = resolve_paths(config, PROJECT_ROOT)
    source_value = paths["datasets"].get(dataset)
    if not source_value or not Path(source_value).exists():
        raise _path_error(dataset, source_value, environment)
    source = Path(source_value)
    if environment == "KAGGLE" and not _within(source, Path("/kaggle/input")):
        raise ValueError(
            f"Refusing Kaggle source outside immutable /kaggle/input: {source}\n"
            + "Available entries:\n"
            + ("\n".join(f"  {item}" for item in _available_kaggle_inputs()) or "  (none found)")
        )

    processed_root = Path(paths["outputs"]["processed"])
    if environment == "KAGGLE" and not _within(processed_root, Path("/kaggle/working")):
        raise ValueError(f"Kaggle processed output must be below /kaggle/working: {processed_root}")
    expected = config["dataset"].get("validation_expectations", {})
    if dataset == "cov_ctr":
        return build_cov_ctr_manifest(source, processed_root, expected=expected)
    return build_iu_xray_manifest(source, processed_root, expected=expected)


def _print_summary(dataset: str, artifacts: object) -> None:
    print(f"\n{dataset} validation complete")
    for key, value in artifacts.counts.items():
        print(f"  {key}: {value}")
    print("  manifests:")
    print(f"    audit: {artifacts.audit_manifest}")
    print(f"    valid: {artifacts.valid_manifest}")
    print(f"    summary: {artifacts.summary}")
    print("  SHA-256:")
    for key, value in artifacts.hashes.items():
        print(f"    {key}: {value}")


def main() -> int:
    args = parse_args()
    environment = detect_environment()
    print(f"Detected environment: {environment}")
    selected = ("cov_ctr", "iu_xray") if args.dataset == "all" else (args.dataset,)
    try:
        for dataset in selected:
            artifacts = _validate_one(dataset, environment, args)
            _print_summary(dataset, artifacts)
    except (ValueError, DatasetSourceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

