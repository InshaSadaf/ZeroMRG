#!/usr/bin/env python
"""Print a read-only ZeroMRG environment and path diagnostic."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.utils.config import flatten_mapping, load_config, resolve_paths
from zeromrg.utils.provenance import (
    dependency_status,
    detect_environment,
    get_device_info,
    system_resources,
)


def _gib(value: Any) -> str:
    return "unknown" if value is None else f"{int(value) / (1024 ** 3):.2f} GiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML file to compose; may be supplied more than once",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration override in dotted.path=YAML_VALUE form",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = detect_environment()
    configs = [Path(item) for item in args.config]
    if not configs:
        configs = [
            PROJECT_ROOT / "configs" / "base.yaml",
            PROJECT_ROOT / "configs" / ("kaggle.yaml" if environment == "KAGGLE" else "local.yaml"),
        ]
    config = load_config(configs, overrides=args.override)
    paths = resolve_paths(config, PROJECT_ROOT)
    device_cfg = config["device"]
    device = get_device_info(device_cfg["preferred"], device_cfg["primary_index"])
    outputs = paths.get("outputs", {})
    disk_anchor = next((value for value in outputs.values() if value), PROJECT_ROOT)
    resources = system_resources(disk_anchor)

    print("ZeroMRG environment check")
    print(f"Environment: {environment}")
    print(f"Python: {platform.python_version()}")
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "NOT INSTALLED"
    print(f"PyTorch: {torch_version}")
    print(f"CUDA available: {device.cuda_available}")
    print(f"CUDA version: {device.cuda_version or 'unavailable'}")
    print(f"GPU count: {device.gpu_count}")
    print(f"Selected device: {device.selected}")
    if device.gpus:
        for gpu in device.gpus:
            print(f"GPU {gpu['index']}: {gpu['name']} ({_gib(gpu['vram_bytes'])})")
    else:
        print("GPU names/VRAM: none detected")
    print(f"CPU: {resources['cpu']}")
    print(f"Logical CPUs: {resources['logical_cpu_count']}")
    print(f"System RAM: {_gib(resources['ram_bytes'])}")
    print(f"Disk checked at: {resources['disk_path']}")
    print(f"Disk free: {_gib(resources['disk_free_bytes'])}")
    print("Resolved paths:")
    for name, value in sorted(flatten_mapping(paths).items()):
        status = "NOT CONFIGURED" if value is None or value == "" else value
        print(f"  {name}: {status}")
    print(f"Configuration hash: {config.digest()}")
    print("Configuration sources:")
    for source in config.sources:
        print(f"  {source}")
    missing = [name for name, available in dependency_status().items() if not available]
    print("Missing dependencies: " + (", ".join(missing) if missing else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
