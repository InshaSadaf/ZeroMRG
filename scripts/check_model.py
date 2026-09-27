#!/usr/bin/env python
"""Run ZeroMRG forward/generation sanity checks without training."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zeromrg.data.archive_io import open_dataset_source
from zeromrg.data.validate_cov_ctr import ANNOTATION_BASENAME
from zeromrg.data.validate_iu_xray import PROJECTION_BASENAME, REPORT_BASENAME
from zeromrg.data.vocabulary import DecoderVocabulary
from zeromrg.models import (
    CLIPImageEncoder,
    MCLIPTextEncoder,
    PromptBank,
    TransformerReportDecoder,
    ZeroMRGAutoEncoder,
)
from zeromrg.models.prompt_bank import load_prompt_ids_artifact
from zeromrg.utils.config import load_config, resolve_paths
from zeromrg.utils.provenance import detect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cov_ctr", "iu_xray"))
    parser.add_argument("--profile", choices=("full", "kaggle_500"), default="full")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", help="Explicit torch device, otherwise CUDA when available")
    parser.add_argument("--prompt-cache", help="Optional prompt embedding .pt cache path")
    parser.add_argument(
        "--sanity-generation-length",
        type=int,
        default=32,
        help="Diagnostic generation cap; trained generation remains configured at 512",
    )
    return parser.parse_args()


def _config(args: argparse.Namespace):
    dataset_config = (
        "iu_xray_kaggle_500.yaml"
        if args.dataset == "iu_xray" and args.profile == "kaggle_500"
        else f"{args.dataset}.yaml"
    )
    files = [PROJECT_ROOT / "configs" / dataset_config, PROJECT_ROOT / "configs" / "zeromrg_model.yaml"]
    environment = detect_environment()
    files.append(PROJECT_ROOT / "configs" / ("kaggle.yaml" if environment == "KAGGLE" else "local.yaml"))
    return load_config(files, overrides=args.override), environment


def _artifact_directory(processed_root: Path, dataset: str, profile: str) -> Path:
    if dataset == "iu_xray" and profile == "kaggle_500":
        return processed_root / "iu_xray" / "kaggle_500"
    return processed_root / dataset


def _manifest_records(directory: Path, dataset: str) -> list[dict]:
    name = "samples_with_splits.jsonl" if dataset == "cov_ctr" else "studies_with_splits.jsonl"
    return [
        json.loads(line)
        for line in (directory / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _parameter_counts(module) -> tuple[int, int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable, total - trainable


def main() -> int:
    args = parse_args()
    try:
        import torch
        from PIL import Image

        config, environment = _config(args)
        paths = resolve_paths(config, PROJECT_ROOT)
        processed_root = Path(paths["outputs"]["processed"])
        directory = _artifact_directory(processed_root, args.dataset, args.profile)
        records = _manifest_records(directory, args.dataset)
        vocabulary_artifact = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
        vocabulary = DecoderVocabulary.from_artifact(vocabulary_artifact)
        prompt_ids, prompt_artifact = load_prompt_ids_artifact(
            directory / "prompt_ids.json", directory / "splits.json"
        )
        configured_prompt_count = int(config["dataset"]["prompt_count"])
        if int(prompt_artifact["count"]) != configured_prompt_count:
            raise ValueError(
                f"Prompt artifact has {prompt_artifact['count']} IDs; "
                f"the active profile requires {configured_prompt_count}"
            )
        expected_vocab_size = (
            358
            if args.dataset == "cov_ctr"
            else 1013 if args.profile == "kaggle_500" else None
        )
        if expected_vocab_size is not None and len(vocabulary.id_to_token) != expected_vocab_size:
            raise ValueError(
                f"{args.dataset} vocabulary has {len(vocabulary.id_to_token)} tokens; "
                f"expected the prepared artifact with {expected_vocab_size}"
            )
        reports_by_id = {
            str(record["membership_id"]): str(record["normalized_target_report"])
            for record in records
        }
        split = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model_config = config["model"]
        text_encoder = MCLIPTextEncoder.from_pretrained(
            model_config["text_encoder"]["checkpoint"],
            device=device,
            frozen=bool(model_config["text_encoder"]["frozen_autoencoding"]),
            max_tokens=int(model_config["text_encoder"]["max_tokens"]),
            truncate=bool(model_config["text_encoder"]["truncate"]),
        )
        image_encoder = CLIPImageEncoder.from_pretrained(
            model_config["image_encoder"]["checkpoint"],
            device=device,
            frozen=bool(model_config["image_encoder"]["frozen_zero_shot"]),
        )
        decoder_config = model_config["decoder"]
        decoder = TransformerReportDecoder(
            vocab_size=len(vocabulary.id_to_token),
            pad_id=vocabulary.pad_id,
            bos_id=vocabulary.bos_id,
            eos_id=vocabulary.eos_id,
            d_model=int(decoder_config["d_model"]),
            num_layers=int(decoder_config["num_layers"]),
            num_heads=int(decoder_config["num_heads"]),
            ffn_dim=int(decoder_config["ffn_dim"]),
            dropout=float(decoder_config["dropout"]),
            activation=str(decoder_config["activation"]),
            max_sequence_length=int(decoder_config["max_sequence_length"]),
            norm_first=bool(decoder_config["norm_first"]),
            final_layer_norm=bool(decoder_config["final_layer_norm"]),
            tie_token_and_output_embeddings=bool(
                decoder_config["tie_token_and_output_embeddings"]
            ),
        ).to(device)
        cache_path = (
            Path(args.prompt_cache)
            if args.prompt_cache
            else Path(paths["outputs"]["artifacts"])
            / "prompt_cache"
            / args.dataset
            / args.profile
            / "mclip_vit_l14.pt"
        )
        prompt_bank = PromptBank.from_reports(
            prompt_ids=prompt_ids,
            training_ids=split["train_ids"],
            reports_by_id=reports_by_id,
            text_encoder=text_encoder,
            dataset=args.dataset,
            profile=args.profile,
            expected_count=configured_prompt_count,
            cache_path=cache_path,
            batch_size=6,
        ).to(device)
        model = ZeroMRGAutoEncoder(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            prompt_bank=prompt_bank,
            decoder=decoder,
        ).to(device)
        model.eval()

        sample = next(record for record in records if record["split"] == "train")
        report = sample["normalized_target_report"]
        text_condition = text_encoder([report])
        full_ids = vocabulary.encode(report, add_bos=True, add_eos=True)
        decoder_inputs = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        logits = model.forward_text([report], decoder_inputs)

        source_path = paths["datasets"].get(args.dataset)
        if not source_path:
            raise ValueError(f"Configure paths.datasets.{args.dataset} for image sanity checks")
        required = [ANNOTATION_BASENAME] if args.dataset == "cov_ctr" else [REPORT_BASENAME, PROJECTION_BASENAME]
        image_record = sample
        if args.dataset == "iu_xray":
            image_record = next(record for record in records if len(record["source_image_paths"]) > 1)
        with open_dataset_source(source_path, required) as source:
            tensors = [
                image_encoder.preprocess(Image.open(io.BytesIO(source.read_bytes(path))))
                for path in image_record["source_image_paths"]
            ]
        if len(tensors) == 1:
            images = tensors[0].unsqueeze(0).to(device)
            raw_image = model.encode_image(images)
            image_condition, enriched = model.image_condition(images)
            view_features_shape = None
            view_mask = None
        else:
            images = torch.stack(tensors).unsqueeze(0).to(device)
            view_mask = torch.ones((1, len(tensors)), dtype=torch.bool, device=device)
            raw_image = model.encode_image(images, view_mask)
            image_condition, enriched = model.image_condition(images, view_mask)
            view_features_shape = tuple(raw_image.shape)
        generated = model.generate_from_embedding(
            image_condition,
            max_length=min(args.sanity_generation_length, decoder.max_sequence_length),
        )
        total, trainable, frozen = _parameter_counts(model)

        print(f"Detected environment: {environment}")
        print(f"Device: {device}")
        print(f"Text encoder: {text_encoder.model_name}")
        print(f"Image encoder: {image_encoder.model_name}")
        print(f"Image preprocessing: {image_encoder.preprocessing_metadata()}")
        print(f"Text embedding shape: {tuple(text_condition.shape)}")
        print(f"Image embedding shape: {tuple(raw_image.shape)}")
        print(f"Prompt bank shape: {tuple(prompt_bank.embeddings.shape)}")
        print(f"Text embedding norm: {text_condition.norm(dim=-1).tolist()}")
        print(f"Image embedding norm: {raw_image.norm(dim=-1).tolist()}")
        print(f"Enriched embedding norm: {image_condition.norm(dim=-1).tolist()}")
        print(f"Decoder input shape: {tuple(decoder_inputs.shape)}")
        print(f"Logits shape: {tuple(logits.shape)}")
        print(f"Generated token shape: {tuple(generated.shape)}")
        if view_features_shape is not None:
            print(f"IU multi-view encoded features: {view_features_shape}")
            print(f"IU view_mask: {view_mask.tolist()}")
            print(f"IU aggregated condition: {tuple(image_condition.shape)}")
        print(f"Parameters: total={total}, trainable={trainable}, frozen={frozen}")
        if device.type == "cuda":
            print(f"Peak allocated GPU memory bytes: {torch.cuda.max_memory_allocated(device)}")
            print(f"Peak reserved GPU memory bytes: {torch.cuda.max_memory_reserved(device)}")
        else:
            print("GPU memory: unavailable (CPU run)")
        print("SANITY STATUS: PASS")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"SANITY STATUS: FAIL - {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
