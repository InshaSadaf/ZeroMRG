"""Execution-environment detection and reproducibility provenance."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import ResolvedConfig

KAGGLE_INDICATORS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "KAGGLE_CONTAINER_NAME",
    "KAGGLE_DATA_PROXY_TOKEN",
    "KAGGLE_USER_SECRETS_TOKEN",
)


def detect_environment(
    environ: Mapping[str, str] | None = None,
    path_exists: Callable[[str], bool] | None = None,
) -> str:
    """Classify execution as KAGGLE, LOCAL, or UNKNOWN.

    ``ZEROMRG_ENVIRONMENT`` is an explicit, testable override. Kaggle detection
    primarily uses environment indicators, with its paired mount points as a
    secondary signal. An ordinary non-CI host is classified as LOCAL.
    """

    env = os.environ if environ is None else environ
    exists = os.path.exists if path_exists is None else path_exists
    explicit = env.get("ZEROMRG_ENVIRONMENT", "").strip().upper()
    if explicit:
        if explicit not in {"KAGGLE", "LOCAL", "UNKNOWN"}:
            return "UNKNOWN"
        return explicit
    if any(env.get(name) for name in KAGGLE_INDICATORS):
        return "KAGGLE"
    if exists("/kaggle/input") and exists("/kaggle/working"):
        return "KAGGLE"
    if env.get("CI") or env.get("GITHUB_ACTIONS") or env.get("COLAB_RELEASE_TAG"):
        return "UNKNOWN"
    return "LOCAL"


@dataclass(frozen=True)
class DeviceInfo:
    selected: str
    cuda_available: bool
    cuda_version: str | None
    gpu_count: int
    gpus: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["gpus"] = list(self.gpus)
        return result


def get_device_info(preferred: str = "auto", primary_index: int = 0) -> DeviceInfo:
    """Inspect accelerators and select one device; distributed use is excluded."""

    preferred = str(preferred).lower()
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError("preferred device must be one of: auto, cpu, cuda")
    try:
        import torch
    except ImportError:
        if preferred == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed")
        return DeviceInfo("cpu", False, None, 0, ())

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpus: list[dict[str, Any]] = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_bytes": int(properties.total_memory),
            }
        )

    if preferred == "cpu" or not cuda_available:
        if preferred == "cuda":
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        selected = "cpu"
    else:
        primary_index = int(primary_index)
        if primary_index < 0 or primary_index >= gpu_count:
            raise ValueError(
                f"Primary CUDA index {primary_index} is invalid for {gpu_count} visible GPU(s)"
            )
        selected = f"cuda:{primary_index}"
    return DeviceInfo(selected, cuda_available, cuda_version, gpu_count, tuple(gpus))


def get_git_commit(repository: str | Path | None = None) -> str | None:
    """Return the current commit, or ``None`` outside an available Git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repository) if repository else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def dependency_status() -> dict[str, bool]:
    """Report import availability without importing heavyweight dependencies."""

    modules = {
        "numpy": "numpy",
        "PyYAML": "yaml",
        "Pillow": "PIL",
        "pandas": "pandas",
        "psutil": "psutil",
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
        "multilingual-clip": "multilingual_clip",
        "openai-clip": "clip",
        "tqdm": "tqdm",
        "pycocoevalcap": "pycocoevalcap",
        "pytest": "pytest",
    }
    status = {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}
    status["Java (METEOR)"] = shutil.which("java") is not None
    return status


def system_resources(path: str | Path | None = None) -> dict[str, Any]:
    """Collect CPU, RAM, and disk facts using optional psutil where available."""

    resource: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": None,
    }
    try:
        import psutil

        resource["ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        pass

    candidate = Path(path or Path.cwd()).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        disk = shutil.disk_usage(candidate)
        resource["disk_path"] = str(candidate.resolve())
        resource["disk_total_bytes"] = int(disk.total)
        resource["disk_free_bytes"] = int(disk.free)
    except OSError:
        resource["disk_path"] = str(candidate)
        resource["disk_total_bytes"] = None
        resource["disk_free_bytes"] = None
    return resource


def create_provenance(
    config: ResolvedConfig,
    dataset_name: str | None = None,
    execution_environment: str | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe provenance record without requiring Git or a GPU."""

    device_cfg = config.data.get("device", {})
    device = get_device_info(
        preferred=device_cfg.get("preferred", "auto"),
        primary_index=device_cfg.get("primary_index", 0),
    )
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    configured_dataset = config.data.get("dataset", {}).get("name")
    seed = config.data.get("reproducibility", {}).get("seed")
    primary_gpu = (
        device.gpus[int(device.selected.split(":", 1)[1])]
        if device.selected.startswith("cuda:")
        else None
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch_version,
        "cuda_available": device.cuda_available,
        "cuda_version": device.cuda_version,
        "selected_device": device.selected,
        "gpu_count": device.gpu_count,
        "gpus": list(device.gpus),
        "gpu_name": primary_gpu["name"] if primary_gpu else None,
        "gpu_vram_bytes": primary_gpu["vram_bytes"] if primary_gpu else None,
        "operating_system": platform.platform(),
        "dataset_name": dataset_name if dataset_name is not None else configured_dataset,
        "execution_environment": execution_environment or detect_environment(),
        "random_seed": seed,
        "git_commit": get_git_commit(repository),
        "config_hash": config.digest(),
        "resolved_configuration": config.to_dict(),
    }


def save_provenance(record: Mapping[str, Any], destination: str | Path) -> Path:
    """Atomically save a provenance record as UTF-8 JSON."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
