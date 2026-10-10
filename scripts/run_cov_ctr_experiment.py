#!/usr/bin/env python
"""Run COV-CTR ZeroMRG experiments and keep all outputs in one result folder.

``--profile full`` uses the paper companion encoders and official metrics.
``--profile smoke`` is a deterministic CPU diagnostic and is never paper-comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F

from zeromrg.data.archive_io import open_dataset_source
from zeromrg.data.validate_cov_ctr import ANNOTATION_BASENAME
from zeromrg.data.vocabulary import DecoderVocabulary
from zeromrg.evaluation import diagnostic_caption_metrics, official_caption_metrics, write_experiment_graphs
from zeromrg.models import CLIPImageEncoder, MCLIPTextEncoder, TransformerReportDecoder, enrich_with_prompts
from zeromrg.utils.seed import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, help="Run directory; defaults inside results/cov_ctr")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "COV-CTR with English reports.zip")
    parser.add_argument("--epochs", type=int, help="Override stage epoch count")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience")
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--device", help="Torch device; full defaults to CUDA when available")
    parser.add_argument("--smoke-train", type=int, default=48)
    parser.add_argument("--smoke-validation", type=int, default=8)
    parser.add_argument("--smoke-test", type=int, default=12)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_records() -> list[dict]:
    path = PROJECT_ROOT / "data" / "processed" / "cov_ctr" / "samples_with_splits.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def batches(records: Sequence[dict], size: int, *, shuffle: bool, seed: int) -> Iterable[list[dict]]:
    indices = list(range(len(records)))
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(records), generator=generator).tolist()
    for start in range(0, len(indices), size):
        yield [records[index] for index in indices[start : start + size]]


def token_batch(records: Sequence[dict], vocabulary: DecoderVocabulary, device: torch.device) -> torch.Tensor:
    sequences = [vocabulary.encode(row["normalized_target_report"], add_bos=True, add_eos=True) for row in records]
    width = max(map(len, sequences))
    tensor = torch.full((len(records), width), vocabulary.pad_id, dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        tensor[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return tensor


def loss_for(decoder: TransformerReportDecoder, condition: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logits = decoder(condition, ids[:, :-1])
    targets = ids[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=decoder.pad_id)


def decode_generated(vocabulary: DecoderVocabulary, rows: torch.Tensor) -> list[str]:
    results = []
    for row in rows.tolist():
        content = []
        for token_id in row:
            if token_id == vocabulary.eos_id:
                break
            if token_id not in (vocabulary.bos_id, vocabulary.pad_id):
                content.append(token_id)
        results.append(vocabulary.decode(content))
    return results


def predict(
    decoder: TransformerReportDecoder,
    records: Sequence[dict],
    condition_fn: Callable[[Sequence[dict], bool], torch.Tensor],
    vocabulary: DecoderVocabulary,
    device: torch.device,
    batch_size: int,
    generation_length: int,
) -> list[dict]:
    decoder.eval()
    output: list[dict] = []
    with torch.no_grad():
        for group in batches(records, batch_size, shuffle=False, seed=0):
            condition = condition_fn(group, False)
            generated = decoder.generate(condition, max_length=generation_length)
            texts = decode_generated(vocabulary, generated.cpu())
            for record, generated_text in zip(group, texts):
                output.append({"sample_id": record["membership_id"], "image_id": record["image_ids"][0], "reference": record["normalized_target_report"], "generated": generated_text})
    return output


def metric_value(metrics: dict, name: str) -> float:
    value = metrics.get(name, 0.0)
    return float(value) if isinstance(value, (float, int)) else 0.0


def train_stage(
    *,
    name: str,
    decoder: TransformerReportDecoder,
    train_records: Sequence[dict],
    validation_records: Sequence[dict],
    train_condition: Callable[[Sequence[dict], bool], torch.Tensor],
    validation_condition: Callable[[Sequence[dict], bool], torch.Tensor],
    extra_parameters: Sequence[torch.nn.Parameter],
    extra_module: torch.nn.Module | None,
    vocabulary: DecoderVocabulary,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    generation_length: int,
    metric_fn: Callable,
    checkpoint: Path,
) -> list[dict]:
    parameters = list(decoder.parameters()) + list(extra_parameters)
    optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.99)
    best, bad_epochs = -1.0, 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        decoder.train()
        losses = []
        for group in batches(train_records, batch_size, shuffle=True, seed=42 + epoch):
            optimizer.zero_grad(set_to_none=True)
            condition = train_condition(group, True)
            ids = token_batch(group, vocabulary, device)
            loss = loss_for(decoder, condition, ids)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss in {name} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_predictions = predict(decoder, validation_records, validation_condition, vocabulary, device, batch_size, generation_length)
        validation_metrics = metric_fn([row["reference"] for row in validation_predictions], [row["generated"] for row in validation_predictions])
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation_bleu4": metric_value(validation_metrics, "BLEU-4")}
        history.append(row)
        print(f"{name} epoch {epoch:03d}: loss={row['train_loss']:.4f}, val BLEU-4={row['validation_bleu4']:.4f}")
        if row["validation_bleu4"] > best:
            best, bad_epochs = row["validation_bleu4"], 0
            payload = {"decoder": decoder.state_dict(), "epoch": epoch, "validation_bleu4": best}
            if extra_module is not None:
                payload["extra_module"] = extra_module.state_dict()
            torch.save(payload, checkpoint)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    decoder.load_state_dict(saved["decoder"])
    if extra_module is not None:
        extra_module.load_state_dict(saved["extra_module"])
    return history


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_id", "reference", "generated"))
        writer.writeheader()
        writer.writerows(rows)


def smoke_conditions(records: Sequence[dict], dimension: int, *, image: bool, source=None) -> torch.Tensor:
    vectors = []
    for record in records:
        if image:
            material = source.read_bytes(record["source_image_paths"][0])
            digest = hashlib.sha512(material).digest()
            raw = [digest[index % len(digest)] / 127.5 - 1.0 for index in range(dimension)]
        else:
            raw = [0.0] * dimension
            for token in record["normalized_target_report"].split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                raw[int.from_bytes(digest[:4], "big") % dimension] += 1.0 if digest[4] % 2 else -1.0
        vectors.append(raw)
    return F.normalize(torch.tensor(vectors, dtype=torch.float32), dim=-1)


def run_smoke(args: argparse.Namespace, output: Path, records: list[dict], vocabulary: DecoderVocabulary) -> tuple[dict, dict]:
    device = torch.device(args.device or "cpu")
    train = [row for row in records if row["split"] == "train"][: args.smoke_train]
    validation = [row for row in records if row["split"] == "validation"][: args.smoke_validation]
    test = [row for row in records if row["split"] == "test"][: args.smoke_test]
    dimension, epochs, patience, batch_size, generation_length = 32, args.epochs or 2, args.patience or 2, args.batch_size or 8, 40
    decoder = TransformerReportDecoder(vocab_size=len(vocabulary.id_to_token), pad_id=vocabulary.pad_id, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id, d_model=dimension, num_layers=1, num_heads=4, ffn_dim=64, dropout=0.1, max_sequence_length=128).to(device)
    histories: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    prompt_records = [row for row in train[: min(16, len(train))]]
    prompt_vectors = smoke_conditions(prompt_records, dimension, image=False).to(device)
    text_fn = lambda group, grad: smoke_conditions(group, dimension, image=False).to(device)
    with open_dataset_source(args.dataset, [ANNOTATION_BASENAME]) as source:
        def image_fn(group, grad):
            raw = smoke_conditions(group, dimension, image=True, source=source).to(device)
            return enrich_with_prompts(raw, prompt_vectors).phi

        histories["text_autoencoder"] = train_stage(name="text_autoencoder", decoder=decoder, train_records=train, validation_records=validation, train_condition=text_fn, validation_condition=text_fn, extra_parameters=[], extra_module=None, vocabulary=vocabulary, device=device, batch_size=batch_size, epochs=epochs, patience=patience, generation_length=generation_length, metric_fn=diagnostic_caption_metrics, checkpoint=output / "checkpoint_text_autoencoder.pt")
        text_state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
        zero_predictions = predict(decoder, test, image_fn, vocabulary, device, batch_size, generation_length)
        metrics["0_percent_paired"] = diagnostic_caption_metrics([row["reference"] for row in zero_predictions], [row["generated"] for row in zero_predictions])
        write_predictions(output / "predictions_0_percent.csv", zero_predictions)
        paired10_ids = set(read_json(PROJECT_ROOT / "data" / "processed" / "cov_ctr" / "paired_10_ids.json")["ids"])
        paired10 = [row for row in train if row["membership_id"] in paired10_ids]
        if not paired10:
            paired10 = train[: max(1, round(0.1 * len(train)))]
        for label, subset in (("10_percent_paired", paired10), ("100_percent_paired", train)):
            decoder.load_state_dict(text_state)
            histories[label] = train_stage(name=label, decoder=decoder, train_records=subset, validation_records=validation, train_condition=image_fn, validation_condition=image_fn, extra_parameters=[], extra_module=None, vocabulary=vocabulary, device=device, batch_size=batch_size, epochs=epochs, patience=patience, generation_length=generation_length, metric_fn=diagnostic_caption_metrics, checkpoint=output / f"checkpoint_{label}.pt")
            predictions = predict(decoder, test, image_fn, vocabulary, device, batch_size, generation_length)
            metrics[label] = diagnostic_caption_metrics([row["reference"] for row in predictions], [row["generated"] for row in predictions])
            write_predictions(output / f"predictions_{label}.csv", predictions)
    details = {"profile": "smoke", "scientific_status": "DIAGNOSTIC_ONLY_NOT_PAPER_COMPARABLE", "encoder_backend": "deterministic_hash_features", "split_subset_counts": {"train": len(train), "validation": len(validation), "test": len(test)}, "decoder": {"d_model": dimension, "layers": 1, "heads": 4}, "epochs_per_stage": epochs, "batch_size": batch_size}
    return {"histories": histories, "metrics": metrics}, details


def load_images(source, records: Sequence[dict], image_encoder: CLIPImageEncoder, device: torch.device) -> torch.Tensor:
    try:
        import io
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Full COV-CTR execution requires Pillow") from exc
    images = []
    for record in records:
        with Image.open(io.BytesIO(source.read_bytes(record["source_image_paths"][0]))) as image:
            images.append(image_encoder.preprocess(image.convert("RGB")))
    return torch.stack(images).to(device)


def run_full(args: argparse.Namespace, output: Path, records: list[dict], vocabulary: DecoderVocabulary) -> tuple[dict, dict]:
    if not torch.cuda.is_available() and not args.device:
        raise RuntimeError("The full ViT-L/14 experiment requires a GPU; pass --device explicitly only if intentional")
    device = torch.device(args.device or "cuda")
    epochs, patience, batch_size = args.epochs or 200, args.patience or 50, args.batch_size or 6
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    text_encoder = MCLIPTextEncoder.from_pretrained(device=device, frozen=True)
    image_encoder = CLIPImageEncoder.from_pretrained(device=device, frozen=True)
    decoder = TransformerReportDecoder(vocab_size=len(vocabulary.id_to_token), pad_id=vocabulary.pad_id, bos_id=vocabulary.bos_id, eos_id=vocabulary.eos_id, d_model=768, num_layers=6, num_heads=8, ffn_dim=3072, dropout=0.1, max_sequence_length=512).to(device)
    text_cache = {}
    with torch.no_grad():
        for group in batches(train + validation, 6, shuffle=False, seed=0):
            encoded = text_encoder([row["normalized_target_report"] for row in group]).cpu()
            text_cache.update((row["membership_id"], vector) for row, vector in zip(group, encoded))
    text_fn = lambda group, grad: torch.stack([text_cache[row["membership_id"]] for row in group]).to(device)
    prompt_ids = set(read_json(PROJECT_ROOT / "data" / "processed" / "cov_ctr" / "prompt_ids.json")["ids"])
    prompts = [row for row in train if row["membership_id"] in prompt_ids]
    with torch.no_grad():
        prompt_vectors = torch.cat([text_encoder([row["normalized_target_report"] for row in group]).cpu() for group in batches(prompts, 6, shuffle=False, seed=0)]).to(device)
    histories: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    histories["text_autoencoder"] = train_stage(name="text_autoencoder", decoder=decoder, train_records=train, validation_records=validation, train_condition=text_fn, validation_condition=text_fn, extra_parameters=[], extra_module=None, vocabulary=vocabulary, device=device, batch_size=batch_size, epochs=epochs, patience=patience, generation_length=128, metric_fn=official_caption_metrics, checkpoint=output / "checkpoint_text_autoencoder.pt")
    torch.save(decoder.state_dict(), output / "decoder_text_start.pt")
    torch.save(image_encoder.state_dict(), output / "vision_start.pt")
    with open_dataset_source(args.dataset, [ANNOTATION_BASENAME]) as source:
        def image_fn(group, grad):
            image_encoder.train(grad)
            images = load_images(source, group, image_encoder, device)
            features = image_encoder(images)
            return enrich_with_prompts(features, prompt_vectors).phi
        predictions = predict(decoder, test, image_fn, vocabulary, device, batch_size, 128)
        metrics["0_percent_paired"] = official_caption_metrics([row["reference"] for row in predictions], [row["generated"] for row in predictions])
        write_predictions(output / "predictions_0_percent.csv", predictions)
        paired_ids = set(read_json(PROJECT_ROOT / "data" / "processed" / "cov_ctr" / "paired_10_ids.json")["ids"])
        for label, subset in (("10_percent_paired", [row for row in train if row["membership_id"] in paired_ids]), ("100_percent_paired", train)):
            decoder.load_state_dict(torch.load(output / "decoder_text_start.pt", map_location=device, weights_only=True))
            image_encoder.load_state_dict(torch.load(output / "vision_start.pt", map_location=device, weights_only=True))
            image_encoder.set_frozen(False)
            histories[label] = train_stage(name=label, decoder=decoder, train_records=subset, validation_records=validation, train_condition=image_fn, validation_condition=image_fn, extra_parameters=list(image_encoder.parameters()), extra_module=image_encoder, vocabulary=vocabulary, device=device, batch_size=batch_size, epochs=epochs, patience=patience, generation_length=128, metric_fn=official_caption_metrics, checkpoint=output / f"checkpoint_{label}.pt")
            predictions = predict(decoder, test, image_fn, vocabulary, device, batch_size, 128)
            metrics[label] = official_caption_metrics([row["reference"] for row in predictions], [row["generated"] for row in predictions])
            write_predictions(output / f"predictions_{label}.csv", predictions)
    details = {"profile": "full", "scientific_status": "PAPER_REPRODUCTION_RUN", "encoder_backend": "M-CLIP_XLM-Roberta-Large-ViT-L-14_and_OpenAI_CLIP_ViT-L-14", "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)}, "epochs_max": epochs, "early_stopping_patience": patience, "batch_size": batch_size}
    return {"histories": histories, "metrics": metrics}, details


def main() -> int:
    args = arguments()
    seed_everything(42, deterministic=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or PROJECT_ROOT / "results" / "cov_ctr" / f"{args.profile}_{timestamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty result directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = read_records()
    vocabulary_artifact = read_json(PROJECT_ROOT / "data" / "processed" / "cov_ctr" / "vocabulary.json")
    vocabulary = DecoderVocabulary.from_artifact(vocabulary_artifact)
    started = datetime.now(timezone.utc)
    try:
        result, details = (run_full if args.profile == "full" else run_smoke)(args, output, records, vocabulary)
        write_json(output / "training_history.json", result["histories"])
        write_json(output / "metrics.json", result["metrics"])
        graph_files = write_experiment_graphs(output, result["histories"], result["metrics"])
        details.update({"dataset": "cov_ctr", "seed": 42, "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "torch": torch.__version__, "result_directory": str(output), "graphs": [path.name for path in graph_files], "status": "COMPLETE"})
        write_json(output / "run_summary.json", details)
        (output / "README.txt").write_text("COV-CTR ZeroMRG result bundle\n\nRead run_summary.json before interpreting metrics. Smoke outputs are diagnostic only and are not research results.\n", encoding="utf-8")
        print(f"RESULT DIRECTORY: {output}")
        print("STATUS: COMPLETE")
        return 0
    except Exception as exc:
        write_json(output / "run_summary.json", {"dataset": "cov_ctr", "profile": args.profile, "scientific_status": "FAILED_NO_RESULTS", "status": "FAILED", "error": str(exc), "started_at_utc": started.isoformat(), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
