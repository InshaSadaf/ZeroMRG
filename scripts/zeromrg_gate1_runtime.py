"""Checkpoint-producing Kaggle decoder and fail-closed recovery utilities.

This profile preserves the recorded COV-CTR experiment, not modular defaults.
No training, model download, or inference is performed on import.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = dict(vocab_size=368, max_tokens=160, d_model=768,
                    num_layers=6, num_heads=8, ffn_dim=3072, dropout=0.1,
                    activation="gelu", norm_first=False, final_layer_norm=False,
                    scale_token_embeddings=False, pad_id=0, bos_id=1, eos_id=2)
TRAINING = dict(seed=42, batch_size=6, learning_rate=0.001, momentum=0.99,
                gradient_clip=5.0, patience=50, max_epochs=200,
                precision="float32", scheduler=None,
                selection_metric="validation_text_reconstruction_BLEU4_PTB_x100")
MCLIP_NAME = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
DEPENDENCIES = ("train_manifest.csv", "validation_manifest.csv", "test_manifest.csv",
                "prompt_manifest.csv", "validated_manifest.csv", "validation_audit.json",
                "decoder_vocabulary.json", "mclip_text_features.pt", "mclip_prompt_bank.pt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    require(Path(path).is_file(), f"Selected checkpoint does not exist: {path}")
    # No unsafe unpickling fallback. Legacy project artifacts load with weights_only=True.
    return torch.load(path, map_location="cpu", weights_only=True)


class CheckpointReportDecoder(nn.Module):
    def __init__(self, architecture=None):
        super().__init__()
        self.architecture = dict(ARCHITECTURE if architecture is None else architecture)
        a = self.architecture
        require(not a["scale_token_embeddings"] and not a["final_layer_norm"],
                "This class implements the recorded Kaggle decoder only")
        self.token_embedding = nn.Embedding(a["vocab_size"], a["d_model"], padding_idx=a["pad_id"])
        self.position_embedding = nn.Embedding(a["max_tokens"], a["d_model"])
        layer = nn.TransformerDecoderLayer(
            d_model=a["d_model"], nhead=a["num_heads"], dim_feedforward=a["ffn_dim"],
            dropout=a["dropout"], activation=a["activation"], batch_first=True,
            norm_first=a["norm_first"])
        self.transformer = nn.TransformerDecoder(layer, num_layers=a["num_layers"])
        self.output_projection = nn.Linear(a["d_model"], a["vocab_size"])

    def forward(self, condition, input_ids):
        a = self.architecture
        require(input_ids.ndim == 2 and 0 < input_ids.shape[1] <= a["max_tokens"],
                "Decoder IDs must be [B,T] within the saved positional limit")
        require(tuple(condition.shape) == (input_ids.shape[0], a["d_model"]),
                "Condition dimensions incompatible with checkpoint decoder")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        target = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1],
                                    dtype=torch.bool, device=input_ids.device), diagonal=1)
        hidden = self.transformer(target, condition.unsqueeze(1).to(target.dtype),
                                  tgt_mask=mask, tgt_key_padding_mask=input_ids.eq(a["pad_id"]))
        return self.output_projection(hidden)

    @torch.no_grad()
    def generate(self, condition, max_length=None):
        self.eval()
        a = self.architecture
        limit = a["max_tokens"] if max_length is None else int(max_length)
        require(2 <= limit <= a["max_tokens"], "Invalid generation length")
        ids = torch.full((len(condition), 1), a["bos_id"], dtype=torch.long, device=condition.device)
        finished = torch.zeros(len(condition), dtype=torch.bool, device=condition.device)
        for _ in range(limit - 1):
            next_id = self(condition, ids)[:, -1].argmax(-1)
            next_id = torch.where(finished, torch.full_like(next_id, a["pad_id"]), next_id)
            ids = torch.cat((ids, next_id[:, None]), dim=1)
            finished |= next_id.eq(a["eos_id"])
            if bool(finished.all()):
                break
        return ids


def validate_model_state(checkpoint, model):
    require(isinstance(checkpoint, dict), "Checkpoint must be a state dictionary payload")
    state = checkpoint.get("decoder")
    require(isinstance(state, dict), "Required decoder state dictionary absent")
    expected = model.state_dict()
    require(set(state) == set(expected),
            f"Decoder key mismatch: missing={sorted(set(expected)-set(state))}; unexpected={sorted(set(state)-set(expected))}")
    for key, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and tensor.shape == expected[key].shape,
                f"Decoder shape mismatch: {key}")
        require(tensor.dtype == expected[key].dtype, f"Decoder dtype mismatch: {key}")
        require(bool(torch.isfinite(tensor).all()), f"Non-finite decoder weights: {key}")
    if "architecture" in checkpoint:
        require(checkpoint["architecture"] == model.architecture, "Saved architecture metadata differs")
    for key, expected_value in (("vocabulary_size", model.architecture["vocab_size"]),
                                ("max_tokens", model.architecture["max_tokens"])):
        if key in checkpoint:
            require(checkpoint[key] == expected_value, f"Saved {key} differs")


def inspect_training_state(checkpoint, model, best_checkpoint=None):
    require(isinstance(checkpoint, dict), "Training checkpoint must be a dictionary payload")
    required = {"epoch", "decoder", "optimizer", "history", "best_validation_bleu4", "epochs_without_improvement"}
    require(required <= checkpoint.keys(), f"Required training state absent: {sorted(required-checkpoint.keys())}")
    epoch = checkpoint["epoch"]
    require(type(epoch) is int and epoch > 0, "Saved epoch must be a positive integer")
    require(checkpoint.get("scheduler") is None or checkpoint.get("scheduler") == {},
            "Unknown scheduler state; supply its verified implementation before resuming")
    if "training_config" in checkpoint:
        require(checkpoint["training_config"] == TRAINING, "Training configuration differs from recorded run")
    history = checkpoint["history"]
    require(isinstance(history, list) and len(history) == epoch, "History length differs from saved epoch")
    require(all(isinstance(row, dict) and {"epoch", "training_loss", "validation_bleu4"} <= row.keys()
                for row in history), "History record fields absent or invalid")
    require([row["epoch"] for row in history] == list(range(1, epoch + 1)), "History is not contiguous")
    for row in history:
        require(all(math.isfinite(float(row[key])) for key in ("training_loss", "validation_bleu4")),
                "Non-finite history value")
    selected = max(history, key=lambda row: row["validation_bleu4"])
    require(math.isclose(float(selected["validation_bleu4"]), float(checkpoint["best_validation_bleu4"]),
                         rel_tol=0, abs_tol=1e-10), "History and saved best metric disagree")
    require(type(checkpoint["epochs_without_improvement"]) is int and
            checkpoint["epochs_without_improvement"] == epoch - selected["epoch"], "Patience state disagrees with history")
    validate_model_state(checkpoint, model)
    if best_checkpoint is not None:
        validate_selected_best(checkpoint, model, best_checkpoint)
    opt = checkpoint["optimizer"]
    require(isinstance(opt, dict) and {"state", "param_groups"} <= opt.keys(), "Invalid optimizer state")
    require(len(opt["param_groups"]) == 1, "Expected one decoder SGD parameter group")
    group = opt["param_groups"][0]
    for key, expected in dict(lr=0.001, momentum=0.99, dampening=0, weight_decay=0,
                              nesterov=False, maximize=False).items():
        require(group.get(key, False if key == "maximize" else None) == expected,
                f"SGD setting mismatch: {key}")
    parameters = list(model.named_parameters())
    saved_ids = group["params"]
    require(len(saved_ids) == len(parameters) and len(set(saved_ids)) == len(saved_ids),
            "Optimizer parameter count or mapping incompatible")
    require(set(opt["state"]) == set(saved_ids), "Missing or unexpected optimizer parameter states")
    nonzero = 0
    for pid, (name, parameter) in zip(saved_ids, parameters):
        require(set(opt["state"][pid]) == {"momentum_buffer"}, f"Unknown/missing SGD state: {name}")
        buffer = opt["state"][pid]["momentum_buffer"]
        require(isinstance(buffer, torch.Tensor) and buffer.shape == parameter.shape and
                buffer.dtype == parameter.dtype, f"Momentum shape/dtype incompatible: {name}")
        require(bool(torch.isfinite(buffer).all()), f"Non-finite momentum: {name}")
        nonzero += int(bool(buffer.ne(0).any()))
    require(nonzero > 0, "All SGD momentum is zero")
    if "parameter_names" in checkpoint:
        require(checkpoint["parameter_names"] == [name for name, _ in parameters], "Optimizer parameter order differs")
    return dict(epoch=epoch, next_epoch=epoch + 1, best_epoch=selected["epoch"],
                best_validation_bleu4=float(selected["validation_bleu4"]),
                epochs_without_improvement=checkpoint["epochs_without_improvement"],
                momentum_buffers=len(saved_ids), nonzero_momentum_buffers=nonzero,
                parameter_count=sum(p.numel() for _, p in parameters),
                legacy_rng_missing="rng_state" not in checkpoint,
                scheduler="none", compatibility="PASS")


def validate_selected_best(checkpoint, model, best_checkpoint):
    require(isinstance(best_checkpoint, dict), "Selected best must be a dictionary payload")
    selected = max(checkpoint["history"], key=lambda row: row["validation_bleu4"])
    require(best_checkpoint.get("epoch") == selected["epoch"] and
            math.isclose(float(best_checkpoint.get("validation_bleu4", float("nan"))),
                         float(selected["validation_bleu4"]), rel_tol=0, abs_tol=1e-10),
            f"Best checkpoint disagrees with history: need epoch {selected['epoch']}, "
            f"found epoch {best_checkpoint.get('epoch')}. Recover matching weights; do not substitute latest weights.")
    validate_model_state(best_checkpoint, model)


def restore_weights_optimizer(checkpoint, model, device="cpu"):
    """Restore latest state for diagnostics; not authorization to continue training."""
    result = inspect_training_state(checkpoint, model)
    model.load_state_dict(checkpoint["decoder"], strict=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for parameter, state in optimizer.state.items():
        state["momentum_buffer"] = state["momentum_buffer"].to(parameter.device)
    for pid, parameter in zip(checkpoint["optimizer"]["param_groups"][0]["params"], model.parameters()):
        require(torch.equal(optimizer.state[parameter]["momentum_buffer"].cpu(),
                            checkpoint["optimizer"]["state"][pid]["momentum_buffer"].cpu()),
                "Momentum changed during restoration")
    model.eval()
    return optimizer, result


def restore_training(checkpoint, model, best_checkpoint, device="cpu"):
    result = inspect_training_state(checkpoint, model, best_checkpoint)
    optimizer, _ = restore_weights_optimizer(checkpoint, model, device)
    return optimizer, result


def read_manifest(directory, name):
    with (Path(directory) / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_artifacts(directory, expected_hashes=None):
    directory = Path(directory)
    hashes = {name: sha256(directory / name) for name in DEPENDENCIES}
    if expected_hashes is not None:
        require(hashes == expected_hashes, "Feature/vocabulary/split dependency hashes differ")
    records = {name: read_manifest(directory, name + "_manifest.csv")
               for name in ("train", "validation", "test", "prompt", "validated")}
    ids = {}
    for name, count in dict(train=571, validation=71, test=72, prompt=500, validated=714).items():
        rows = records[name]
        ids[name] = [row["image_id"] for row in rows]
        require(len(rows) == count and len(set(ids[name])) == count, f"{name} count/ID uniqueness failed")
    train, val, test = (set(ids[name]) for name in ("train", "validation", "test"))
    require(not (train & val or train & test or val & test), "Split leakage")
    require(train | val | test == set(ids["validated"]), "Split population differs")
    require(set(ids["prompt"]) <= train, "Prompt bank contains non-training IDs")
    by_id = {r["image_id"]: r for r in records["validated"]}
    for name in ("train", "validation", "test", "prompt"):
        require(all(row["report"] == by_id[row["image_id"]]["report"] for row in records[name]),
                f"{name} reports differ from validated manifest")
    audit = json.loads((directory / "validation_audit.json").read_text(encoding="utf-8"))
    require(audit["csv_rows"] == 726 and audit["physical_images"] == 746, "Raw audit population differs")
    vocabulary = json.loads((directory / "decoder_vocabulary.json").read_text(encoding="utf-8"))
    require(vocabulary["max_tokens"] == 160 and len(vocabulary["vocabulary"]) == 368,
            "Vocabulary or positional capacity differs")
    require(vocabulary["special_token_ids"] == dict(pad=0, bos=1, eos=2, unk=3), "Special-token IDs differ")
    expected_vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted({
        token for row in records["train"] for token in re.findall(r"\w+|[^\w\s]", row["report"])})
    require(vocabulary["vocabulary"] == expected_vocab, "Vocabulary is not the saved training-only lexicographic mapping")
    text = load_checkpoint(directory / "mclip_text_features.pt")
    require(text["model"] == MCLIP_NAME and text["normalization"] == "l2", "Text encoder metadata differs")
    for name in ("train", "validation"):
        require(set(text[name]) == set(ids[name]), f"{name} feature membership differs")
        for vector in text[name].values():
            require(vector.shape == (768,) and bool(torch.isfinite(vector).all()) and
                    torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-4), "Invalid normalized text feature")
    prompt = load_checkpoint(directory / "mclip_prompt_bank.pt")
    require(prompt["model"] == MCLIP_NAME and prompt.get("normalization") == "raw", "Prompt encoder/normalization differs")
    require(prompt["prompt_ids"] == ids["prompt"], "Prompt tensor row order/provenance differs")
    vectors = prompt["embeddings"]
    require(vectors.shape == (500, 768) and bool(torch.isfinite(vectors).all()) and
            bool((vectors.norm(dim=-1) > 0).all()), "Invalid prompt bank")
    aligned = torch.stack([text["train"][key] for key in ids["prompt"]])
    # Separate encodings are not bitwise equal. Record the observed difference;
    # membership and frozen-cache hashes above are the provenance assertions.
    normalized_prompts = F.normalize(vectors, dim=-1)
    cosine = (normalized_prompts * aligned).sum(-1)
    cache_comparison = dict(max_absolute_difference=float((normalized_prompts - aligned).abs().max()),
                            minimum_cosine=float(cosine.min()), mean_cosine=float(cosine.mean()),
                            exact_encoder_revision="UNVERIFIED")
    return dict(records=records, vocabulary=vocabulary, text=text, prompt=prompt,
                dependency_hashes=hashes, prompt_text_comparison=cache_comparison)


def encode_report(text, vocabulary):
    lookup = {token: i for i, token in enumerate(vocabulary)}
    ids = [1] + [lookup.get(token, 3) for token in re.findall(r"\w+|[^\w\s]", text)] + [2]
    require(len(ids) <= 160, "Report exceeds saved positional capacity; no silent truncation")
    return ids


def decode_report(token_ids, vocabulary):
    tokens = []
    for raw in token_ids:
        token = int(raw)
        if token == 2:
            break
        require(0 <= token < len(vocabulary), "Generated token outside vocabulary")
        if token not in {0, 1}:
            tokens.append(vocabulary[token])
    text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", " ".join(tokens))
    return re.sub(r"([(\[])\s+", r"\1", text).strip()


def enrich(image, prompts):
    require(image.ndim == prompts.ndim == 2 and image.shape[-1] == prompts.shape[-1] == 768,
            "Image/prompt dimensions must share the recorded 768-D CLIP space")
    require(bool(torch.isfinite(image).all()) and bool(torch.isfinite(prompts).all()), "Non-finite encoder output")
    scores = image @ prompts.T / math.sqrt(768)
    attention = scores.softmax(-1)
    residual = image + attention @ prompts
    require(bool((residual.norm(dim=-1) > 0).all()), "Zero residual embedding")
    phi = F.normalize(residual, p=2, dim=-1)
    require(bool(torch.isfinite(phi).all()), "Non-finite enriched features")
    return phi, attention


def capture_rng():
    return dict(python=random.getstate(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        require(len(state["torch_cuda"]) == torch.cuda.device_count(), "CUDA RNG device count differs")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def atomic_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def training_payload(model, optimizer, epoch, history, best_metric, bad_epochs, hashes, provenance):
    return dict(schema_version="zeromrg_covctr_gate1_v1", epoch=epoch, decoder=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=None, history=history,
                best_validation_bleu4=best_metric, epochs_without_improvement=bad_epochs,
                architecture=model.architecture, parameter_names=[n for n, _ in model.named_parameters()],
                training_config=TRAINING, dependency_hashes=hashes, provenance=provenance,
                rng_state=capture_rng())
