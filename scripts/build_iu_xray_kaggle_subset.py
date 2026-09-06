#!/usr/bin/env python
"""Build deterministic IU-Xray-500 preparation artifacts and portable package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.data.archive_io import DatasetSourceError
from zeromrg.data.iu_xray_package import build_iu_xray_subset_package
from zeromrg.data.iu_xray_subset import prepare_iu_xray_kaggle_500
from zeromrg.data.split import SplitIntegrityError
from zeromrg.utils.config import load_config, resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help="Additional YAML overlay; repeatable")
    parser.add_argument("--override", action="append", default=[], help="dotted.path=YAML_VALUE; repeatable")
    parser.add_argument("--source", help="Override paths.datasets.iu_xray")
    parser.add_argument("--output-root", help="Override the package artifact root")
    parser.add_argument("--no-zip", action="store_true", help="Build the directory but omit the deterministic ZIP")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = [PROJECT_ROOT / "configs" / "iu_xray_kaggle_500.yaml", PROJECT_ROOT / "configs" / "local.yaml"]
    configs.extend(Path(value) for value in args.config)
    overrides = list(args.override)
    if args.source:
        overrides.append(f"paths.datasets.iu_xray={args.source}")
    try:
        config = load_config(configs, overrides=overrides)
        paths = resolve_paths(config, PROJECT_ROOT)
        source = Path(paths["datasets"]["iu_xray"])
        if not source.exists():
            raise ValueError(f"Configured IU-Xray source does not exist: {source}")
        prepared = prepare_iu_xray_kaggle_500(paths["outputs"]["processed"], config)
        artifact_root = Path(args.output_root).expanduser().resolve() if args.output_root else Path(paths["outputs"]["artifacts"])
        package = build_iu_xray_subset_package(
            source_path=source,
            subset_artifact_path=prepared.subset_ids,
            preparation_summary_path=prepared.preparation_summary,
            package_dir=artifact_root / "IU-Xray-500",
            create_zip=not args.no_zip,
        )
        print("IU-Xray-500 RESOURCE-CONSTRAINED profile ready")
        print(f"  subset IDs: {prepared.subset_ids}")
        print(f"  profile artifacts: {prepared.directory}")
        print(f"  split counts: {prepared.counts}")
        print(f"  physical images: {package['physical_image_count']}")
        print(f"  package: {artifact_root / 'IU-Xray-500'}")
        if package.get("zip_path"):
            print(f"  zip: {package['zip_path']}")
            print(f"  zip SHA-256: {package['zip_sha256']}")
        print(f"  reused: preparation={prepared.reused}, package={package['reused']}")
    except (OSError, ValueError, KeyError, DatasetSourceError, SplitIntegrityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
