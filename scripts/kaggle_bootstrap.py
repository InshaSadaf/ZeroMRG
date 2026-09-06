#!/usr/bin/env python
"""Validate a Kaggle workspace without preprocessing, downloads, or training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.utils.config import flatten_mapping, load_config, resolve_paths
from zeromrg.utils.provenance import dependency_status, detect_environment, get_device_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--cov-ctr-path", help="Attached COV-CTR path under /kaggle/input")
    parser.add_argument("--iu-xray-path", help="Attached IU-Xray path under /kaggle/input")
    return parser.parse_args()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    config_files = [
        PROJECT_ROOT / "configs" / "base.yaml",
        *(Path(p) for p in args.config),
        PROJECT_ROOT / "configs" / "kaggle.yaml",
    ]
    overrides = list(args.override)
    if args.cov_ctr_path:
        overrides.append(f"paths.datasets.cov_ctr={args.cov_ctr_path}")
    if args.iu_xray_path:
        overrides.append(f"paths.datasets.iu_xray={args.iu_xray_path}")
    config = load_config(config_files, overrides=overrides)
    paths = resolve_paths(config, PROJECT_ROOT)

    environment = detect_environment()
    print(f"Detected environment: {environment}")
    if environment != "KAGGLE":
        print("WARNING: this bootstrap is intended to run inside a Kaggle notebook.")

    expected = [PROJECT_ROOT / "src", PROJECT_ROOT / "configs", PROJECT_ROOT / "scripts"]
    missing_project_dirs = [str(path) for path in expected if not path.is_dir()]
    print("Project directories: " + ("OK" if not missing_project_dirs else f"MISSING {missing_project_dirs}"))

    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        entries = sorted(input_root.iterdir(), key=lambda item: item.name.lower())
        print("Attached Kaggle inputs:")
        if entries:
            for entry in entries:
                kind = "directory" if entry.is_dir() else "file"
                print(f"  {entry} [{kind}]")
        else:
            print("  none")
    else:
        print("Attached Kaggle inputs: /kaggle/input is unavailable")

    invalid_inputs: list[str] = []
    print("Configured dataset paths:")
    for name, value in paths.get("datasets", {}).items():
        if not value:
            print(f"  {name}: NOT CONFIGURED")
            continue
        candidate = Path(value)
        valid = candidate.exists() and _under(candidate, input_root)
        print(f"  {name}: {candidate} ({'OK' if valid else 'INVALID'})")
        if not valid:
            invalid_inputs.append(name)

    working_root = Path("/kaggle/working")
    created_outputs: list[Path] = []
    invalid_outputs: list[str] = []
    print("Writable output paths:")
    for name, value in paths.get("outputs", {}).items():
        candidate = Path(value)
        if not _under(candidate, working_root):
            print(f"  {name}: REFUSED (must be below /kaggle/working): {candidate}")
            invalid_outputs.append(name)
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".zeromrg_write_check"
            test_file.touch(exist_ok=True)
            test_file.unlink()
            created_outputs.append(candidate)
            print(f"  {name}: {candidate} (writable)")
        except OSError as exc:
            print(f"  {name}: ERROR ({exc})")
            invalid_outputs.append(name)

    device_cfg = config["device"]
    device = get_device_info(device_cfg["preferred"], device_cfg["primary_index"])
    print(f"Selected device: {device.selected} ({device.gpu_count} visible GPU(s))")
    missing_dependencies = [name for name, ok in dependency_status().items() if not ok]
    print("Missing dependencies: " + (", ".join(missing_dependencies) if missing_dependencies else "none"))
    print(f"Configuration hash: {config.digest()}")

    if missing_project_dirs or invalid_inputs or invalid_outputs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
