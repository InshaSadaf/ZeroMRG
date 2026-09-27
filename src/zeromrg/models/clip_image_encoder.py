"""OpenAI CLIP ViT-L/14 visual encoder paired with the paper's M-CLIP."""

from __future__ import annotations

from contextlib import nullcontext
from importlib import metadata
from typing import Any

import torch
from torch import nn

PAPER_VISUAL_CHECKPOINT = "ViT-L/14"
PAPER_VISUAL_DIM = 768
PAPER_INPUT_RESOLUTION = 224
CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


def _installed_version(distribution: str) -> str:
    """Return reproducibility metadata without making model loading brittle."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


class CLIPImageEncoder(nn.Module):
    """Wrap only the official CLIP visual tower, including its 768-D projection."""

    def __init__(
        self,
        visual: nn.Module,
        preprocess: Any = None,
        *,
        model_name: str = PAPER_VISUAL_CHECKPOINT,
        embedding_dim: int = PAPER_VISUAL_DIM,
        input_resolution: int = PAPER_INPUT_RESOLUTION,
        frozen: bool = True,
        version_metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.preprocess = preprocess
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)
        self.input_resolution = int(input_resolution)
        self.version_metadata = dict(version_metadata or {})
        self.frozen = False
        self.set_frozen(frozen)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = PAPER_VISUAL_CHECKPOINT,
        *,
        device: str | torch.device | None = None,
        frozen: bool = True,
        download_root: str | None = None,
    ) -> "CLIPImageEncoder":
        if model_name != PAPER_VISUAL_CHECKPOINT:
            raise ValueError(
                f"Paper companion visual encoder must be {PAPER_VISUAL_CHECKPOINT}, got {model_name}"
            )
        try:
            import clip
        except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
            raise RuntimeError("Loading ViT-L/14 requires the official OpenAI CLIP package") from exc
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, preprocess = clip.load(
            model_name, device=target, jit=False, download_root=download_root
        )
        visual = clip_model.visual
        resolution = int(getattr(visual, "input_resolution", PAPER_INPUT_RESOLUTION))
        output_dim = int(getattr(visual, "output_dim", PAPER_VISUAL_DIM))
        del clip_model
        if resolution != PAPER_INPUT_RESOLUTION or output_dim != PAPER_VISUAL_DIM:
            raise RuntimeError(
                f"Unexpected {model_name} metadata: resolution={resolution}, output_dim={output_dim}; "
                f"expected {PAPER_INPUT_RESOLUTION} and {PAPER_VISUAL_DIM}"
            )
        versions = {"openai_clip": _installed_version("clip")}
        return cls(
            visual,
            preprocess,
            model_name=model_name,
            embedding_dim=output_dim,
            input_resolution=resolution,
            frozen=frozen,
            version_metadata=versions,
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.visual.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.visual.parameters()).dtype
        except StopIteration:
            return torch.float32

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        for parameter in self.visual.parameters():
            parameter.requires_grad_(not self.frozen)
        if self.frozen:
            self.visual.eval()

    def train(self, mode: bool = True) -> "CLIPImageEncoder":
        super().train(mode)
        if self.frozen:
            self.visual.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected preprocessed images [B,3,H,W], got {tuple(images.shape)}")
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            features = self.visual(images.to(device=self.device, dtype=self.dtype)).float()
        if features.ndim != 2 or features.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"CLIP visual encoder returned {tuple(features.shape)}; expected [B,{self.embedding_dim}]"
            )
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    def preprocessing_metadata(self) -> dict[str, Any]:
        return {
            "resize": self.input_resolution,
            "center_crop": self.input_resolution,
            "color_mode": "RGB",
            "interpolation": "bicubic",
            "mean": list(CLIP_PIXEL_MEAN),
            "std": list(CLIP_PIXEL_STD),
        }

    def cache_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "input_resolution": self.input_resolution,
            "versions": self.version_metadata,
            "preprocessing": self.preprocessing_metadata(),
        }
