"""Composition of paper-faithful ZeroMRG encoder and decoder paths."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .decoder import TransformerReportDecoder
from .prompt_bank import PromptBank
from .prompt_enrichment import (
    PromptEnrichmentOutput,
    enrich_and_pool_views,
    enrich_with_prompts,
)


class ZeroMRGAutoEncoder(nn.Module):
    """One decoder shared by text auto-encoding and zero-shot image inference."""

    def __init__(
        self,
        *,
        text_encoder: nn.Module,
        decoder: TransformerReportDecoder,
        image_encoder: nn.Module | None = None,
        prompt_bank: PromptBank | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.image_encoder = image_encoder
        self.prompt_bank = prompt_bank
        text_dim = int(getattr(text_encoder, "embedding_dim", decoder.d_model))
        if text_dim != decoder.d_model:
            raise ValueError(f"Text embedding dimension {text_dim} != decoder d_model {decoder.d_model}")
        if image_encoder is not None:
            image_dim = int(getattr(image_encoder, "embedding_dim", decoder.d_model))
            if image_dim != text_dim:
                raise ValueError(f"Image embedding dimension {image_dim} != text dimension {text_dim}")
        if prompt_bank is not None and prompt_bank.embedding_dim != text_dim:
            raise ValueError(
                f"Prompt embedding dimension {prompt_bank.embedding_dim} != text dimension {text_dim}"
            )

    def forward_text(
        self, reports: Sequence[str], decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        condition = self.text_encoder(reports)
        return self.decoder(condition, decoder_input_ids)

    def encode_image(
        self, images: torch.Tensor, view_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.image_encoder is None:
            raise RuntimeError("ZeroMRG has no image encoder")
        if images.ndim == 4:
            return self.image_encoder(images)
        if images.ndim != 5:
            raise ValueError("Images must be [B,C,H,W] or [B,V,C,H,W]")
        if view_mask is None or tuple(view_mask.shape) != tuple(images.shape[:2]):
            raise ValueError("Multi-view images require view_mask [B,V]")
        batch, views = images.shape[:2]
        mask = view_mask.to(device=images.device, dtype=torch.bool)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every study requires at least one valid image view")
        flat_images = images.reshape(batch * views, *images.shape[2:])
        flat_mask = mask.reshape(-1)
        valid_features = self.image_encoder(flat_images[flat_mask])
        features = torch.zeros(
            batch * views,
            valid_features.shape[-1],
            dtype=valid_features.dtype,
            device=valid_features.device,
        )
        features[flat_mask.to(valid_features.device)] = valid_features
        return features.reshape(batch, views, -1)

    def enrich_with_prompts(self, image_embeddings: torch.Tensor) -> PromptEnrichmentOutput:
        if self.prompt_bank is None:
            raise RuntimeError("ZeroMRG has no prompt bank")
        return enrich_with_prompts(image_embeddings, self.prompt_bank.embeddings)

    def image_condition(
        self, images: torch.Tensor, view_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, PromptEnrichmentOutput]:
        features = self.encode_image(images, view_mask)
        if features.ndim == 2:
            output = self.enrich_with_prompts(features)
            return output.phi, output
        if self.prompt_bank is None or view_mask is None:
            raise RuntimeError("Multi-view conditioning requires a prompt bank and view mask")
        return enrich_and_pool_views(features, self.prompt_bank.embeddings, view_mask)

    def generate_from_embedding(
        self, condition: torch.Tensor, *, max_length: int | None = None
    ) -> torch.Tensor:
        return self.decoder.generate(condition, max_length=max_length)

    def generate_from_images(
        self,
        images: torch.Tensor,
        *,
        view_mask: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        condition, _ = self.image_condition(images, view_mask)
        return self.generate_from_embedding(condition, max_length=max_length)
