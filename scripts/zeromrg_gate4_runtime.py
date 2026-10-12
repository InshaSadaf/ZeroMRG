"""Frozen COV-CTR Gate4 inference; no loading, downloading or execution on import."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from zeromrg_gate1_runtime import (
    ARCHITECTURE, MCLIP_NAME, CheckpointReportDecoder, decode_report, enrich,
    load_checkpoint, require, sha256, validate_model_state,
)

CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_SOURCE_SHA256 = "9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6"
CLIP_WEIGHT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
EPOCH90_SHA256 = "6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2"
INFERENCE_FILES = (
    "train_manifest.csv", "prompt_manifest.csv", "decoder_vocabulary.json", "mclip_prompt_bank.pt",
)


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    require(all(p.grad is None for p in module.parameters()), "Unexpected existing gradients")
    return module


def assert_frozen(module):
    require(not module.training and all(not p.requires_grad and p.grad is None
            for p in module.parameters()), "Inference module is not frozen/eval")


def load_inference_artifacts(directory, expected_hashes):
    """Read train-only inputs. Never open validation/test references or text cache."""
    directory = Path(directory)
    hashes = {name: sha256(directory / name) for name in INFERENCE_FILES}
    require(hashes == {name: expected_hashes[name] for name in INFERENCE_FILES},
            "Inference artifact hashes differ from verified checkpoint family")
    def ids(name):
        with (directory / name).open(encoding="utf-8-sig", newline="") as handle:
            return [row["image_id"] for row in csv.DictReader(handle)]
    train_ids, prompt_ids = ids("train_manifest.csv"), ids("prompt_manifest.csv")
    require(len(train_ids) == len(set(train_ids)) == 571, "Training IDs differ")
    require(len(prompt_ids) == len(set(prompt_ids)) == 500 and
            set(prompt_ids) <= set(train_ids), "Prompt membership is not training-only")
    bank = load_checkpoint(directory / "mclip_prompt_bank.pt")
    validate_prompt_bank(bank, prompt_ids)
    vocab = json.loads((directory / "decoder_vocabulary.json").read_text(encoding="utf-8"))
    require(vocab["max_tokens"] == 160 and len(vocab["vocabulary"]) == 368 and
            vocab["special_token_ids"] == dict(pad=0, bos=1, eos=2, unk=3), "Vocabulary differs")
    require(vocab["vocabulary"][:4] == ["<pad>", "<bos>", "<eos>", "<unk>"], "Token IDs differ")
    return dict(train_ids=train_ids, prompt_ids=prompt_ids, prompts=bank["embeddings"].detach(),
                vocabulary=vocab["vocabulary"], hashes=hashes)


def validate_prompt_bank(bank, prompt_ids):
    require(bank["model"] == MCLIP_NAME and bank.get("normalization") == "raw",
            "Wrong prompt encoder or normalization")
    require(bank["prompt_ids"] == prompt_ids, "Prompt bank row ordering differs")
    vectors = bank["embeddings"]
    require(tuple(vectors.shape) == (500, 768) and vectors.dtype == torch.float32 and
            not vectors.requires_grad and bool(torch.isfinite(vectors).all()) and
            bool((vectors.norm(dim=-1) > 0).all()), "Invalid fixed raw prompt bank")


def restore_epoch90(path, device="cpu", expected_sha256=EPOCH90_SHA256):
    require(sha256(path) == expected_sha256, "Epoch90 checkpoint hash differs")
    checkpoint = load_checkpoint(path)
    require(checkpoint.get("epoch") == 90, "Diagnostic requires verified physical epoch90")
    with torch.device("meta"):
        decoder = CheckpointReportDecoder()
    validate_model_state(checkpoint, decoder)
    decoder.load_state_dict(checkpoint["decoder"], strict=True, assign=True)
    freeze(decoder)
    decoder.to(device)
    require(decoder.architecture == ARCHITECTURE, "Historical decoder architecture differs")
    return decoder


def load_companion_clip(device, cache_directory):
    """Explicit future Kaggle operation. Official package/source and weight are pinned."""
    import clip
    require(sha256(Path(clip.__file__).with_name("clip.py")) == CLIP_SOURCE_SHA256,
            "Wrong OpenAI CLIP implementation")
    url = clip.clip._MODELS["ViT-L/14"]
    require(url.split("/")[-2] == CLIP_WEIGHT_SHA256, "Wrong companion weight identity")
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False,
                                  download_root=str(cache_directory))
    weight_path = Path(cache_directory) / "ViT-L-14.pt"
    require(sha256(weight_path) == CLIP_WEIGHT_SHA256, "CLIP weight hash differs")
    freeze(model)
    vision = model.visual
    require(int(vision.output_dim) == 768 and int(vision.input_resolution) == 224,
            "Companion CLIP dimensions differ")
    return vision, preprocess, dict(model="OpenAI ViT-L/14", commit=CLIP_COMMIT,
        weight_sha256=CLIP_WEIGHT_SHA256, clip_source_sha256=CLIP_SOURCE_SHA256,
        preprocess=repr(preprocess), historical_cache_revision="UNVERIFIED")


def preprocess_image(image, preprocess):
    tensor = preprocess(image.convert("RGB"))
    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (3, 224, 224) and
            tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
            "CLIP preprocessing must return finite float32 RGB [3,224,224]")
    return tensor.unsqueeze(0)


def resolve_training_image(data_root, image_id, train_ids):
    require(image_id in train_ids and Path(image_id).name == image_id and
            "/" not in image_id and "\\" not in image_id, "Diagnostic image must be a saved training ID")
    candidates = [p for p in Path(data_root).rglob(image_id) if p.is_file()]
    require(len(candidates) == 1, "Training image missing or ambiguous")
    require(candidates[0].resolve().is_relative_to(Path(data_root).resolve()), "Image escapes data root")
    return candidates[0]


class FrozenImageToReport:
    def __init__(self, vision, preprocess, decoder, prompts, vocabulary):
        self.vision, self.preprocess, self.decoder = freeze(vision), preprocess, freeze(decoder)
        require(decoder.architecture == ARCHITECTURE, "Historical decoder required")
        require(tuple(prompts.shape) == (500, 768) and prompts.dtype == torch.float32 and
                not prompts.requires_grad and bool(torch.isfinite(prompts).all()) and
                bool((prompts.norm(dim=-1) > 0).all()), "Invalid raw fixed prompt tensor")
        require(len(vocabulary) == 368, "Historical vocabulary required")
        self.prompts, self.vocabulary = prompts.detach().clone(), list(vocabulary)
        self.used = False

    @torch.inference_mode()
    def generate_one(self, image, max_length=160, progress=None):
        require(not self.used, "Single-image diagnostic already attempted; no automatic retry")
        require(2 <= max_length <= 160, "Generation exceeds historical positions")
        self.used = True
        log = progress or (lambda phase: None)
        assert_frozen(self.vision)
        assert_frozen(self.decoder)
        device = next(self.decoder.parameters()).device
        require(next(self.vision.parameters()).device == device, "Encoder/decoder device mismatch")
        start = time.monotonic()
        log("preprocess")
        pixels = preprocess_image(image, self.preprocess).to(device)
        log("image_encoder")
        dtype = self.vision.conv1.weight.dtype
        visual = self.vision(pixels.to(dtype=dtype)).float()
        require(tuple(visual.shape) == (1, 768) and bool(torch.isfinite(visual).all()),
                "CLIP output must be finite [1,768]")
        log("prompt_enrichment_eq6_eq5")
        prompts = self.prompts.to(device)
        phi, attention = enrich(visual, prompts)
        residual = visual + attention @ prompts
        require(tuple(attention.shape) == (1, 500) and
                torch.allclose(attention.sum(-1), torch.ones(1, device=device), atol=1e-5) and
                torch.allclose(phi.norm(dim=-1), torch.ones(1, device=device), atol=1e-5),
                "Attention/residual normalization failed")
        # Use the existing historical greedy generation, with a hook to verify EVERY
        # autoregressive forward is finite and visibly log bounded token progress.
        forward_count = 0
        def check_logits(module, inputs, output):
            nonlocal forward_count
            require(bool(torch.isfinite(output).all()), "Non-finite autoregressive logits")
            forward_count += 1
            if forward_count == 1 or forward_count % 20 == 0:
                log("greedy_token_" + str(forward_count))
        hook = self.decoder.register_forward_hook(check_logits)
        try:
            tokens = self.decoder.generate(phi, max_length=max_length).cpu()
        finally:
            hook.remove()
        require(tuple(tokens.shape[:1]) == (1,) and 2 <= tokens.shape[1] <= max_length and
                bool(((tokens >= 0) & (tokens < 368)).all()), "Invalid generated IDs")
        assert_frozen(self.vision)
        assert_frozen(self.decoder)
        require(not self.prompts.requires_grad and torch.equal(prompts.cpu(), self.prompts.cpu()),
                "Prompt tensor changed")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ids = tokens[0].tolist()
        return dict(shapes=dict(pixels=list(pixels.shape), image_embedding=list(visual.shape),
            prompts=list(prompts.shape), attention=list(attention.shape), residual=list(residual.shape),
            decoder_condition=list(phi.shape), tokens=list(tokens.shape)),
            norms=dict(image=visual.norm(dim=-1).cpu().tolist(),
                prompt_min=float(prompts.norm(dim=-1).min()), prompt_max=float(prompts.norm(dim=-1).max()),
                residual=residual.norm(dim=-1).cpu().tolist(), enriched=phi.norm(dim=-1).cpu().tolist()),
            output_tokens=ids, report_text=decode_report(ids, self.vocabulary),
            termination="EOS" if 2 in ids else "MAX_LENGTH", forward_count=forward_count,
            device=str(device), elapsed_seconds=time.monotonic()-start,
            gradients_disabled=True, all_modules_frozen=True, optimizer_created=False,
            paired_supervision_used=False, test_references_accessed=False)
