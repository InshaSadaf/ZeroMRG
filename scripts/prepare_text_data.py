#!/usr/bin/env python
"""Create deterministic split and decoder-text artifacts from validated manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.data.prepare_text import prepare_text_data
from zeromrg.data.iu_xray_subset import prepare_iu_xray_kaggle_500
from zeromrg.data.archive_io import DatasetSourceError
from zeromrg.data.split import SplitIntegrityError
from zeromrg.utils.config import load_config, resolve_paths
from zeromrg.utils.provenance import detect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cov_ctr", "iu_xray", "all"))
    parser.add_argument(
        "--profile",
        choices=("full", "kaggle_500"),
        default="full",
        help="IU-Xray execution profile; COV-CTR always remains full",
    )
    parser.add_argument(
        "--config", action="append", default=[], help="Additional YAML overlay; repeatable"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration override in dotted.path=YAML_VALUE form; repeatable",
    )
    return parser.parse_args()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_dataset_config(dataset: str, environment: str, args: argparse.Namespace):
    config_name = (
        "iu_xray_kaggle_500"
        if dataset == "iu_xray" and args.profile == "kaggle_500"
        else dataset
    )
    files: list[Path] = [PROJECT_ROOT / "configs" / f"{config_name}.yaml"]
    if environment == "KAGGLE":
        files.append(PROJECT_ROOT / "configs" / "kaggle.yaml")
    elif environment == "LOCAL":
        files.append(PROJECT_ROOT / "configs" / "local.yaml")
    files.extend(Path(value) for value in args.config)
    return load_config(files, overrides=args.override)


def _print_result(result: object) -> None:
    print(f"\n{result.dataset} text preparation complete")
    print(
        "  split counts: "
        f"train={result.counts['train']}, validation={result.counts['validation']}, test={result.counts['test']}"
    )
    print(f"  paired-10 IDs: {result.paired_10_count}")
    print(f"  prompt IDs: {result.prompt_count}")
    print(f"  decoder vocabulary size: {result.vocabulary_size}")
    print(f"  maximum sequence length (including BOS/EOS): {result.max_sequence_length}")
    print(f"  reports exceeding configured maximum: {result.reports_over_configured_max}")
    print("  reused compatible artifacts:")
    for name, reused in result.reused.items():
        print(f"    {name}: {reused}")
    print("  artifact SHA-256:")
    for name, value in result.hashes.items():
        print(f"    {name}: {value}")


def main() -> int:
    args = parse_args()
    environment = detect_environment()
    print(f"Detected environment: {environment}")
    selected = ("cov_ctr", "iu_xray") if args.dataset == "all" else (args.dataset,)
    try:
        for dataset in selected:
            config = _load_dataset_config(dataset, environment, args)
            paths = resolve_paths(config, PROJECT_ROOT)
            processed_root = Path(paths["outputs"]["processed"])
            if environment == "KAGGLE" and not _within(processed_root, Path("/kaggle/working")):
                raise SplitIntegrityError(
                    f"Kaggle processed artifacts must remain below /kaggle/working: {processed_root}"
                )
            profile_name = config.get("execution_profile", {}).get("name")
            if dataset == "iu_xray" and profile_name == "kaggle_500":
                source_path = paths["datasets"].get("iu_xray")
                if not source_path:
                    raise SplitIntegrityError(
                        "IU-Xray-500 preparation requires paths.datasets.iu_xray to point "
                        "to the attached reduced package"
                    )
                result = prepare_iu_xray_kaggle_500(
                    processed_root, config, source_path=source_path
                )
            else:
                result = prepare_text_data(dataset, processed_root, config)
            _print_result(result)
    except (OSError, ValueError, DatasetSourceError, SplitIntegrityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
