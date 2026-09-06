"""Deterministic random-seed management with optional PyTorch support."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    seed: int
    numpy_seeded: bool
    torch_seeded: bool
    cuda_seeded: bool
    deterministic_algorithms: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed for a named operation or dataset."""

    material = "::".join([str(int(base_seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, PyTorch, and every visible CUDA device if available."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        pass

    torch_seeded = False
    cuda_seeded = False
    deterministic_enabled = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        if deterministic:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
                deterministic_enabled = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    return SeedState(
        seed=seed,
        numpy_seeded=numpy_seeded,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        deterministic_algorithms=deterministic_enabled,
    )

