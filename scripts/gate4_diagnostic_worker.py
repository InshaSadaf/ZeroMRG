"""One future Kaggle training-image diagnostic, launched only by the bounded notebook."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

from zeromrg_gate1_runtime import require, sha256
from zeromrg_gate4_runtime import (
    CLIP_COMMIT, EPOCH90_SHA256, INFERENCE_FILES, FrozenImageToReport,
    load_companion_clip, load_inference_artifacts, resolve_training_image, restore_epoch90,
)


def log(phase):
    print(json.dumps(dict(phase=phase, elapsed_seconds=round(time.monotonic()-START, 3))), flush=True)


def version(name):
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None


def run_checked(args, timeout=300):
    log("dependency_command")
    subprocess.run(args, check=True, timeout=timeout)


def diagnostic(run_root, config):
    import torch
    require(config["max_images"] == 1 and config["max_tokens"] == 160 and
            config["training_enabled"] is False and config["test_inference_enabled"] is False, "Diagnostic scope differs")
    require(Path("/kaggle/input").is_dir() and run_root.resolve().is_relative_to(Path("/kaggle/working")),
            "Kaggle-only diagnostic")
    require(torch.cuda.is_available(), "Kaggle GPU required; no local fallback")
    device = torch.device("cuda:0")
    log("provided_environment_checks")
    # These are the observed baseline stack. Changed images stop for review, not upgrade.
    require(torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8", "Torch/CUDA drift; review required")
    require(version("numpy") == "2.1.3" and version("Pillow") == "12.3.0", "NumPy/Pillow drift")
    require(version("torchvision") is not None and version("torchvision").split("+")[0] == "0.26.0",
            "Expected torchvision0.26.0 companion; do not auto-install Torch/torchvision")
    import torchvision
    from PIL import Image
    # /tmp is ephemeral and excluded from Kaggle saved notebook outputs.
    scratch = Path(tempfile.mkdtemp(prefix="zeromrg_gate4_",dir="/tmp"))
    package_root = scratch / "inference_packages"
    package_root.mkdir(exist_ok=False)
    required = dict(ftfy="6.3.1", regex="2025.11.3", tqdm="4.67.1", wcwidth="0.9.2")
    missing = [line for line in config["support_lock"].splitlines()
               if version(line.split("==")[0]) != required[line.split("==")[0]]]
    if missing:
        lock = run_root / "support.lock"
        lock.write_text("\n".join(missing)+"\n", encoding="utf-8")
        run_checked([sys.executable,"-m","pip","install","--target",str(package_root),
            "--no-deps","--require-hashes","--only-binary=:all:","--retries","0","--timeout","45",
            "-r",str(lock)])
    sys.path.insert(0,str(package_root))
    for name, expected in required.items(): require(version(name) == expected, "Support package pin mismatch: "+name)
    if version("clip") is None:
        require(version("setuptools") is not None and version("wheel") is not None,
                "Provided CLIP build tools missing; no implicit build-tool install")
        run_checked([sys.executable,"-m","pip","install","--target",str(package_root),
            "--no-deps","--no-build-isolation","--retries","0","--timeout","45",
            "clip @ git+https://github.com/openai/CLIP.git@"+CLIP_COMMIT])
    # The exact source hash is checked by load_companion_clip, even for provided clip.
    log("verified_checkpoint_attachment")
    candidates = list(Path("/kaggle/input").rglob(config["zip_name"]))
    require(len(candidates) == 1, "Expected exactly one verified baseline ZIP attachment")
    archive_path = candidates[0]
    require(sha256(archive_path) == config["zip_sha256"], "Baseline ZIP differs")
    artifact_root = scratch / "selected_inputs"
    artifact_root.mkdir(exist_ok=False)
    selected = ["recovery_best_epoch_090.pt"] + ["dependencies/"+name for name in INFERENCE_FILES]
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP members")
        for name in selected:
            destination = artifact_root / name
            require(destination.resolve().is_relative_to(artifact_root.resolve()), "Unsafe ZIP member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Only selected training inputs: reading to EOF verifies their ZIP CRC.
            with archive.open(name) as source, destination.open("xb") as output:
                for block in iter(lambda: source.read(1024*1024), b""): output.write(block)
            require(sha256(destination) == config["selected_hashes"][name], "Selected input differs: "+name)
    data_root = Path(config["data_root"])
    require(data_root.is_dir() and data_root.resolve().is_relative_to(Path("/kaggle/input")), "COV data mount differs")
    artifacts = load_inference_artifacts(artifact_root/"dependencies", config["dependency_hashes"])
    require(hashlib.sha256(json.dumps(artifacts["prompt_ids"],separators=(",",":")).encode()).hexdigest() ==
            config["prompt_ids_sha256"], "Prompt ordering fingerprint differs")
    image_id = config["image_id"]
    require(image_id == artifacts["train_ids"][0], "Fixed first saved training image differs")
    image_path = resolve_training_image(data_root,image_id,artifacts["train_ids"])
    image_sha256 = sha256(image_path)
    log("strict_epoch90_restore")
    decoder = restore_epoch90(artifact_root/"recovery_best_epoch_090.pt",device)
    log("pinned_clip_load")
    vision, preprocess, encoder = load_companion_clip(device,scratch/"clip_weights")
    import clip
    direct_url = metadata.distribution("clip").read_text("direct_url.json")
    if direct_url:
        vcs = json.loads(direct_url).get("vcs_info",{})
        require(vcs.get("commit_id") == CLIP_COMMIT, "CLIP VCS provenance differs")
    pipeline = FrozenImageToReport(vision,preprocess,decoder,artifacts["prompts"],artifacts["vocabulary"])
    versions = {name:version(name) for name in
                ("torch","torchvision","numpy","Pillow","clip","ftfy","regex","tqdm","wcwidth")}
    before = dict(decoder=[p._version for p in decoder.parameters()],
                  encoder=[p._version for p in vision.parameters()])
    torch.cuda.synchronize()
    inference_start = time.monotonic()
    with Image.open(image_path) as image:
        result = pipeline.generate_one(image,max_length=160,progress=log)
    require(before == dict(decoder=[p._version for p in decoder.parameters()],
                           encoder=[p._version for p in vision.parameters()]), "Parameter mutation detected")
    result.update(status="PASS_SINGLE_TRAINING_IMAGE_DIAGNOSTIC",stage="zero_shot_image_to_report_diagnostic",
        image_id=image_id,image_sha256=image_sha256,split="checkpoint_train",image_count=1,
        checkpoint=dict(epoch=90,sha256=EPOCH90_SHA256,zip_sha256=config["zip_sha256"],
                        reconstruction_bleu4=54.21140850351637,selection="available_physical_epoch90; historical85_missing"),
        dependencies=versions,python=platform.python_version(),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),encoder=encoder,artifact_hashes=artifacts["hashes"],
        prompt_ids_sha256=config["prompt_ids_sha256"],config=config,
        source_sha256={name:sha256(Path(__file__).parent/name) for name in
                       ("zeromrg_gate1_runtime.py","zeromrg_gate4_runtime.py","gate4_diagnostic_worker.py")},
        parameter_version_counters_unchanged=True,
        total_worker_seconds=time.monotonic()-START,inference_seconds=time.monotonic()-inference_start,
        scientific_limit="Historical encoder revision/cache alignment unverified; one image is not evaluation")
    (run_root/"diagnostic.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    log("diagnostic_saved")
    print(json.dumps({k:result[k] for k in ("status","image_id","shapes","norms","output_tokens","report_text","termination")}),flush=True)


if __name__ == "__main__":
    START = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root",required=True)
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    destination = Path(args.run_root)
    try:
        diagnostic(destination,json.loads(Path(args.config).read_text(encoding="utf-8")))
    except Exception as error:
        # Never retry or change the architecture/environment automatically.
        (destination/"failure.json").write_text(json.dumps(dict(status="FAILED",error_type=type(error).__name__,
            error=str(error),elapsed_seconds=time.monotonic()-START),indent=2)+"\n",encoding="utf-8")
        log("failed_no_retry")
        raise
