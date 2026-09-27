"""Configurable transformer report decoder conditioned on one embedding token."""

from __future__ import annotations

import math

import torch
from torch import nn


def teacher_forcing_shift(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[BOS, content..., EOS]`` into decoder inputs and next-token targets."""

    if token_ids.ndim != 2 or token_ids.shape[1] < 2:
        raise ValueError("token_ids must be [B,L] with L >= 2")
    return token_ids[:, :-1], token_ids[:, 1:]


class TransformerReportDecoder(nn.Module):
    """Autoregressive decoder; exact layer details are implementation assumptions."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_sequence_length: int = 512,
        norm_first: bool = False,
        final_layer_norm: bool = True,
        tie_token_and_output_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"num_heads={num_heads} must divide d_model={d_model}")
        if min(vocab_size, max_sequence_length, num_layers) <= 0:
            raise ValueError("vocab_size, max_sequence_length, and num_layers must be positive")
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.d_model = int(d_model)
        self.max_sequence_length = int(max_sequence_length)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        final_norm = nn.LayerNorm(d_model) if final_layer_norm else None
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers, norm=final_norm)
        self.output_projection = nn.Linear(d_model, vocab_size)
        if tie_token_and_output_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    @staticmethod
    def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Causal-mask length must be positive")
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, condition: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.d_model:
            raise ValueError(f"condition must be [B,{self.d_model}], got {tuple(condition.shape)}")
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != condition.shape[0]:
            raise ValueError("decoder_input_ids must be [B,T] and share condition batch size")
        sequence_length = int(decoder_input_ids.shape[1])
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Decoder input length {sequence_length} exceeds maximum {self.max_sequence_length}"
            )
        if decoder_input_ids.device != condition.device:
            raise ValueError("Condition and decoder input IDs must be on the same device")
        positions = torch.arange(sequence_length, device=decoder_input_ids.device).unsqueeze(0)
        target = self.token_embedding(decoder_input_ids) * math.sqrt(self.d_model)
        target = target + self.position_embedding(positions)
        memory = condition.unsqueeze(1).to(dtype=target.dtype)
        decoded = self.transformer(
            tgt=target,
            memory=memory,
            tgt_mask=self.causal_mask(sequence_length, decoder_input_ids.device),
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
        )
        return self.output_projection(decoded)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive generation, stopping each sequence on EOS."""

        limit = self.max_sequence_length if max_length is None else int(max_length)
        if limit < 2 or limit > self.max_sequence_length:
            raise ValueError(f"max_length must be in [2,{self.max_sequence_length}]")
        generated = torch.full(
            (condition.shape[0], 1), self.bos_id, dtype=torch.long, device=condition.device
        )
        finished = torch.zeros(condition.shape[0], dtype=torch.bool, device=condition.device)
        for _ in range(limit - 1):
            next_ids = self(condition, generated)[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.pad_id), next_ids)
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished = finished | next_ids.eq(self.eos_id)
            if bool(finished.all()):
                break
        return generated
