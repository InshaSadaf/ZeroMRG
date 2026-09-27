"""Official M-CLIP text-encoder wrapper used by ZeroMRG."""

from __future__ import annotations

from contextlib import nullcontext
from importlib import metadata
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

PAPER_MCLIP_CHECKPOINT = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
PAPER_EMBEDDING_DIM = 768


def _installed_version(distribution: str) -> str:
    """Return reproducibility metadata without making model loading brittle."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


class MCLIPTextEncoder(nn.Module):
    """Batch text encoder that reproduces the official M-CLIP forward pass.

    The official implementation mean-pools XLM-R token states by attention mask
    and applies ``LinearTransformation`` into the companion CLIP space.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        model_name: str = PAPER_MCLIP_CHECKPOINT,
        embedding_dim: int = PAPER_EMBEDDING_DIM,
        frozen: bool = True,
        max_tokens: int = 512,
        truncate: bool = False,
        version_metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)
        self.max_tokens = int(max_tokens)
        self.truncate = bool(truncate)
        self.frozen = False
        self.version_metadata = dict(version_metadata or {})
        self.set_frozen(frozen)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = PAPER_MCLIP_CHECKPOINT,
        *,
        device: str | torch.device | None = None,
        frozen: bool = True,
        cache_dir: str | None = None,
        max_tokens: int = 512,
        truncate: bool = False,
    ) -> "MCLIPTextEncoder":
        if model_name != PAPER_MCLIP_CHECKPOINT:
            raise ValueError(
                f"Paper-faithful text encoding requires {PAPER_MCLIP_CHECKPOINT}, got {model_name}"
            )
        try:
            from multilingual_clip import pt_multilingual_clip
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional heavyweight packages
            raise RuntimeError(
                "Loading the paper M-CLIP encoder requires multilingual-clip and transformers"
            ) from exc
        model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        configured_dim = int(getattr(model.config, "numDims", PAPER_EMBEDDING_DIM))
        if configured_dim != PAPER_EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected M-CLIP output dimension {configured_dim}; expected {PAPER_EMBEDDING_DIM}"
            )
        versions = {
            "multilingual_clip": _installed_version("multilingual-clip"),
            "transformers": _installed_version("transformers"),
        }
        wrapper = cls(
            model,
            tokenizer,
            model_name=model_name,
            embedding_dim=configured_dim,
            frozen=frozen,
            max_tokens=max_tokens,
            truncate=truncate,
            version_metadata=versions,
        )
        if device is not None:
            wrapper.to(device)
        return wrapper

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        for parameter in self.model.parameters():
            parameter.requires_grad_(not self.frozen)
        if self.frozen:
            self.model.eval()

    def train(self, mode: bool = True) -> "MCLIPTextEncoder":
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def _official_features(self, reports: Sequence[str]) -> torch.Tensor:
        if hasattr(self.model, "transformer") and hasattr(self.model, "LinearTransformation"):
            tokenizer_arguments: dict[str, Any] = {
                "padding": True,
                "truncation": self.truncate,
                "return_tensors": "pt",
            }
            if self.truncate:
                tokenizer_arguments["max_length"] = self.max_tokens
            tokenized = self.tokenizer(list(reports), **tokenizer_arguments)
            if not self.truncate and int(tokenized["input_ids"].shape[1]) > self.max_tokens:
                raise ValueError(
                    f"M-CLIP input has {tokenized['input_ids'].shape[1]} tokens, exceeding "
                    f"configured maximum {self.max_tokens}; reports are never silently truncated"
                )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            hidden = self.model.transformer(**tokenized)[0]
            attention = tokenized["attention_mask"]
            pooled = (hidden * attention.unsqueeze(2)).sum(dim=1) / attention.sum(dim=1)[:, None]
            return self.model.LinearTransformation(pooled)
        return self.model(list(reports), self.tokenizer)

    def encode(self, reports: Sequence[str]) -> torch.Tensor:
        if not reports:
            raise ValueError("At least one report is required")
        if any(not isinstance(report, str) for report in reports):
            raise TypeError("Every report must be a string")
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            embeddings = self._official_features(reports)
            if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
                raise RuntimeError(
                    f"M-CLIP returned shape {tuple(embeddings.shape)}; expected [B,{self.embedding_dim}]"
                )
            return F.normalize(embeddings.float(), p=2, dim=-1)

    def forward(self, reports: Sequence[str]) -> torch.Tensor:
        return self.encode(reports)

    def cache_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_tokens": self.max_tokens,
            "truncate": self.truncate,
            "versions": self.version_metadata,
        }
