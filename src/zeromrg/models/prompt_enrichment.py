"""Paper equations (5) and (6), plus documented IU multi-view pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PromptEnrichmentOutput:
    phi: torch.Tensor
    scores: torch.Tensor
    attention: torch.Tensor
    prompt_based: torch.Tensor


def enrich_with_prompts(
    image_embeddings: torch.Tensor, prompt_embeddings: torch.Tensor
) -> PromptEnrichmentOutput:
    """Apply ZeroMRG prompt attention exactly as printed in Eqs. (5)-(6)."""

    if image_embeddings.ndim != 2:
        raise ValueError(f"image_embeddings must be [B,d], got {tuple(image_embeddings.shape)}")
    if prompt_embeddings.ndim != 2:
        raise ValueError(f"prompt_embeddings must be [Np,d], got {tuple(prompt_embeddings.shape)}")
    if image_embeddings.shape[-1] != prompt_embeddings.shape[-1]:
        raise ValueError(
            f"Image/text embedding mismatch: {image_embeddings.shape[-1]} != {prompt_embeddings.shape[-1]}"
        )
    if prompt_embeddings.shape[0] == 0:
        raise ValueError("Prompt bank cannot be empty")
    if image_embeddings.device != prompt_embeddings.device:
        raise ValueError("Image and prompt embeddings must be on the same device")
    prompts = prompt_embeddings.to(dtype=image_embeddings.dtype)
    dimension = int(image_embeddings.shape[-1])
    scores = image_embeddings @ prompts.transpose(0, 1) / math.sqrt(dimension)
    attention = torch.softmax(scores, dim=-1)
    prompt_based = attention @ prompts
    phi = F.normalize(image_embeddings + prompt_based, p=2, dim=-1)
    return PromptEnrichmentOutput(phi, scores, attention, prompt_based)


def masked_mean_views(view_embeddings: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    """IU assumption: masked view mean followed by L2 normalization."""

    if view_embeddings.ndim != 3:
        raise ValueError(f"view_embeddings must be [B,V,d], got {tuple(view_embeddings.shape)}")
    if view_mask.ndim != 2 or tuple(view_mask.shape) != tuple(view_embeddings.shape[:2]):
        raise ValueError(
            f"view_mask must be [B,V]={tuple(view_embeddings.shape[:2])}, got {tuple(view_mask.shape)}"
        )
    mask = view_mask.to(device=view_embeddings.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every study must contain at least one valid view")
    masked = view_embeddings * mask.unsqueeze(-1).to(view_embeddings.dtype)
    pooled = masked.sum(dim=1) / counts.unsqueeze(-1).to(view_embeddings.dtype)
    return F.normalize(pooled, p=2, dim=-1)


def enrich_and_pool_views(
    image_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    view_mask: torch.Tensor,
) -> tuple[torch.Tensor, PromptEnrichmentOutput]:
    """Enrich every view, ignore padding, and produce one condition per study."""

    if image_embeddings.ndim != 3:
        raise ValueError("Multi-view image embeddings must be [B,V,d]")
    batch, views, dimension = image_embeddings.shape
    flat = image_embeddings.reshape(batch * views, dimension)
    enriched = enrich_with_prompts(flat, prompt_embeddings)
    phi_views = enriched.phi.reshape(batch, views, dimension)
    mask = view_mask.to(device=phi_views.device, dtype=torch.bool)
    phi_views = phi_views * mask.unsqueeze(-1).to(phi_views.dtype)
    pooled = masked_mean_views(phi_views, mask)
    shaped = PromptEnrichmentOutput(
        phi=phi_views,
        scores=enriched.scores.reshape(batch, views, -1),
        attention=enriched.attention.reshape(batch, views, -1),
        prompt_based=enriched.prompt_based.reshape(batch, views, dimension),
    )
    return pooled, shaped
