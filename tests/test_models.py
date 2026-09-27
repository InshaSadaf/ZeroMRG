from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - local lightweight environment
    torch = None
    class _UnavailableModule:
        pass

    class _UnavailableNN:
        Module = _UnavailableModule

    nn = _UnavailableNN()

if torch is not None:
    from zeromrg.models import (
        CLIPImageEncoder,
        PromptBank,
        PromptCacheError,
        TransformerReportDecoder,
        ZeroMRGAutoEncoder,
        enrich_with_prompts,
        masked_mean_views,
        teacher_forcing_shift,
    )
    from zeromrg.models.mclip_encoder import MCLIPTextEncoder


@unittest.skipIf(torch is None, "PyTorch is not installed in this lightweight test environment")
class ModelTests(unittest.TestCase):
    class MockTextBackend(nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))
            self.dimension = dimension

        def forward(self, reports, tokenizer):
            rows = [torch.arange(1, self.dimension + 1, dtype=torch.float32) * self.scale for _ in reports]
            return torch.stack(rows)

    class MockTextEncoder(nn.Module):
        model_name = "mock-mclip"

        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.embedding_dim = dimension
            self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, reports):
            rows = []
            for report in reports:
                base = torch.arange(1, self.embedding_dim + 1, dtype=torch.float32)
                rows.append(base + float(len(report) % 3))
            return F.normalize(torch.stack(rows), dim=-1)

        def cache_metadata(self):
            return {"model_name": self.model_name, "version": "test"}

    class MockImageEncoder(nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.embedding_dim = dimension
            self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, images):
            means = images.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            basis = torch.arange(1, self.embedding_dim + 1, device=images.device).float().unsqueeze(0)
            return means + basis

    class MockVisualBackend(nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.arange(1, dimension + 1).float())

        def forward(self, images):
            means = images.float().mean(dim=(1, 2, 3)).unsqueeze(1)
            return means * self.weight.unsqueeze(0)

    def decoder(self, vocab_size: int = 12) -> TransformerReportDecoder:
        return TransformerReportDecoder(
            vocab_size=vocab_size,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            dropout=0.0,
            max_sequence_length=8,
        )

    def test_text_embeddings_are_l2_normalized_and_encoder_is_frozen(self) -> None:
        backend = self.MockTextBackend(8)
        encoder = MCLIPTextEncoder(backend, tokenizer=None, embedding_dim=8, frozen=True)
        output = encoder(["one", "two"])
        self.assertEqual(tuple(output.shape), (2, 8))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertFalse(output.requires_grad)
        encoder.train()
        self.assertFalse(backend.training)

    def test_visual_encoder_is_frozen_without_changing_output_dimension(self) -> None:
        backend = self.MockVisualBackend(8)
        encoder = CLIPImageEncoder(backend, embedding_dim=8, frozen=True)
        output = encoder(torch.ones(2, 3, 4, 4))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertFalse(output.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        encoder.train()
        self.assertFalse(backend.training)

    def test_prompt_attention_shapes_sums_and_enriched_norm(self) -> None:
        images = torch.randn(3, 8)
        prompts = F.normalize(torch.randn(5, 8), dim=-1)
        output = enrich_with_prompts(images, prompts)
        self.assertEqual(tuple(output.scores.shape), (3, 5))
        self.assertEqual(tuple(output.attention.shape), (3, 5))
        self.assertEqual(tuple(output.prompt_based.shape), (3, 8))
        self.assertEqual(tuple(output.phi.shape), (3, 8))
        torch.testing.assert_close(output.attention.sum(dim=-1), torch.ones(3))
        torch.testing.assert_close(output.phi.norm(dim=-1), torch.ones(3))

    def test_masked_view_mean_ignores_padding(self) -> None:
        valid_a = F.normalize(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
        valid_b = F.normalize(torch.tensor([4.0, 3.0, 2.0, 1.0]), dim=0)
        views = torch.stack((valid_a, valid_b, torch.full((4,), 9999.0))).unsqueeze(0)
        mask = torch.tensor([[True, True, False]])
        pooled = masked_mean_views(views, mask)
        expected = F.normalize((valid_a + valid_b) / 2, dim=0).unsqueeze(0)
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(pooled.norm(dim=-1), torch.ones(1))

    def test_teacher_forcing_causal_mask_logits_and_causality(self) -> None:
        decoder = self.decoder()
        decoder.eval()
        full = torch.tensor([[1, 4, 5, 2, 0]])
        inputs, targets = teacher_forcing_shift(full)
        self.assertEqual(inputs.tolist(), [[1, 4, 5, 2]])
        self.assertEqual(targets.tolist(), [[4, 5, 2, 0]])
        mask = decoder.causal_mask(4)
        self.assertFalse(bool(mask[2, 1]))
        self.assertTrue(bool(mask[1, 2]))
        condition = F.normalize(torch.randn(1, 8), dim=-1)
        logits = decoder(condition, inputs)
        self.assertEqual(tuple(logits.shape), (1, 4, 12))
        altered = inputs.clone()
        altered[0, 3] = 9
        altered_logits = decoder(condition, altered)
        torch.testing.assert_close(logits[:, :3], altered_logits[:, :3], atol=1e-6, rtol=1e-6)

    def test_generation_stops_on_eos(self) -> None:
        decoder = self.decoder()
        with torch.no_grad():
            decoder.output_projection.weight.zero_()
            decoder.output_projection.bias.zero_()
            decoder.output_projection.bias[decoder.eos_id] = 10.0
        generated = decoder.generate(torch.randn(2, 8), max_length=8)
        self.assertEqual(generated.tolist(), [[1, 2], [1, 2]])

    def test_prompt_cache_compatibility_and_frozen_buffer(self) -> None:
        encoder = self.MockTextEncoder(8)
        ids = ["a", "b"]
        reports = {"a": "first report", "b": "second report"}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "prompts.pt"
            first = PromptBank.from_reports(
                prompt_ids=ids,
                training_ids=["a", "b", "c"],
                reports_by_id=reports,
                text_encoder=encoder,
                dataset="iu_xray",
                profile="kaggle_500",
                expected_count=2,
                cache_path=cache,
            )
            second = PromptBank.from_reports(
                prompt_ids=ids,
                training_ids=["a", "b", "c"],
                reports_by_id=reports,
                text_encoder=encoder,
                dataset="iu_xray",
                profile="kaggle_500",
                expected_count=2,
                cache_path=cache,
            )
            torch.testing.assert_close(first.embeddings, second.embeddings)
            self.assertEqual(list(first.parameters()), [])
            self.assertFalse(first.embeddings.requires_grad)
            with self.assertRaises(PromptCacheError):
                PromptBank.from_reports(
                    prompt_ids=ids,
                    training_ids=["a", "b"],
                    reports_by_id=reports,
                    text_encoder=encoder,
                    dataset="cov_ctr",
                    profile="full",
                    expected_count=2,
                    cache_path=cache,
                )

    def test_same_decoder_accepts_text_and_visual_conditions(self) -> None:
        text = self.MockTextEncoder(8)
        image = self.MockImageEncoder(8)
        prompt_bank = PromptBank(torch.randn(3, 8), ["p1", "p2", "p3"], {})
        decoder = self.decoder()
        model = ZeroMRGAutoEncoder(
            text_encoder=text,
            image_encoder=image,
            prompt_bank=prompt_bank,
            decoder=decoder,
        )
        decoder_ids = torch.tensor([[1, 4, 5], [1, 6, 0]])
        text_logits = model.forward_text(["one", "two"], decoder_ids)
        visual_condition, enriched = model.image_condition(torch.randn(2, 3, 4, 4))
        visual_logits = decoder(visual_condition, decoder_ids)
        self.assertIs(model.decoder, decoder)
        self.assertEqual(tuple(text_logits.shape), (2, 3, 12))
        self.assertEqual(tuple(visual_logits.shape), (2, 3, 12))
        self.assertEqual(tuple(enriched.attention.shape), (2, 3))

    def test_multiview_model_shapes_and_padded_views_are_not_encoded(self) -> None:
        model = ZeroMRGAutoEncoder(
            text_encoder=self.MockTextEncoder(8),
            image_encoder=self.MockImageEncoder(8),
            prompt_bank=PromptBank(torch.randn(4, 8), ["1", "2", "3", "4"], {}),
            decoder=self.decoder(),
        )
        images = torch.randn(2, 3, 3, 4, 4)
        images[1, 1:] = 100000.0
        mask = torch.tensor([[True, True, True], [True, False, False]])
        encoded = model.encode_image(images, mask)
        self.assertEqual(tuple(encoded.shape), (2, 3, 8))
        self.assertEqual(int(torch.count_nonzero(encoded[1, 1:])), 0)
        condition, enriched = model.image_condition(images, mask)
        self.assertEqual(tuple(enriched.phi.shape), (2, 3, 8))
        self.assertEqual(tuple(condition.shape), (2, 8))
        torch.testing.assert_close(condition.norm(dim=-1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
