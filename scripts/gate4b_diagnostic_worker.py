"""Future Kaggle-only Gate4B worker; no execution or downloads on import."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from importlib import metadata
from contextlib import nullcontext
from zeromrg_gate4b_progress import StageProgress, run_stage, utc_now

PROGRESS = None
if __name__ == "__main__":
    # Export evidence before Torch imports, CUDA initialization, network or mount I/O.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    destination = Path(args.run_root)
    if not destination.is_dir() or not destination.resolve().is_relative_to(Path("/kaggle/working").resolve()):
        raise ValueError("Kaggle working output required")
    PROGRESS = StageProgress(destination)
    PROGRESS.emit("worker_start")

with PROGRESS.stage("runtime_imports") if PROGRESS else nullcontext():
    import gate4_diagnostic_worker as base
    from gate4_diagnostic_worker import version, run_checked
    from zeromrg_gate1_runtime import require, sha256
    from zeromrg_gate4_runtime import (CLIP_COMMIT, EPOCH90_SHA256, INFERENCE_FILES,
        load_companion_clip, load_inference_artifacts, restore_epoch90)
    from zeromrg_gate4b_runtime import diagnose_images, select_nonprompt_ids
    from zeromrg_cov_data_resolver import resolve_cov_dataset, verify_checkpoint_manifests, CovDataError


def log(phase, **details):
    if PROGRESS:
        return PROGRESS.emit(phase, **details)
    return base.log(phase)


def diagnostic(run_root, config):
    import torch
    contract_path = run_root / "dataset-contract.json"
    require(sha256(contract_path) == config["dataset_contract_sha256"], "Dataset identity contract hash differs")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(1 <= config["max_images"] <= 5 and config["max_tokens"] == 160 and
            config["training_enabled"] is False and config["test_inference_enabled"] is False, "Diagnostic scope differs")
    require(Path("/kaggle/input").is_dir() and run_root.resolve().is_relative_to(Path("/kaggle/working")),
            "Kaggle-only diagnostic")
    require(run_stage(PROGRESS,"cuda_availability",torch.cuda.is_available), "Kaggle GPU required; no local fallback")
    device = torch.device("cuda:0")
    log("provided_environment_checks")
    # These are the observed baseline stack. Changed images stop for review, not upgrade.
    require(torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8", "Torch/CUDA drift; review required")
    require(version("numpy") == "2.1.3" and version("Pillow") == "12.3.0", "NumPy/Pillow drift")
    require(version("torchvision") is not None and version("torchvision").split("+")[0] == "0.26.0",
            "Expected torchvision0.26.0 companion; do not auto-install Torch/torchvision")
    with PROGRESS.stage("vision_support_imports") if PROGRESS else nullcontext():
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
        run_stage(PROGRESS,"support_package_setup",run_checked,[sys.executable,"-m","pip","install","--target",str(package_root),
            "--no-deps","--require-hashes","--only-binary=:all:","--retries","0","--timeout","45",
            "-r",str(lock)])
    sys.path.insert(0,str(package_root))
    for name, expected in required.items(): require(version(name) == expected, "Support package pin mismatch: "+name)
    if version("clip") is None:
        require(version("setuptools") is not None and version("wheel") is not None,
                "Provided CLIP build tools missing; no implicit build-tool install")
        run_stage(PROGRESS,"clip_package_setup",run_checked,[sys.executable,"-m","pip","install","--target",str(package_root),
            "--no-deps","--no-build-isolation","--retries","0","--timeout","45",
            "clip @ git+https://github.com/openai/CLIP.git@"+CLIP_COMMIT])
    # The exact source hash is checked by load_companion_clip, even for provided clip.
    log("verified_checkpoint_attachment")
    candidates = list(Path("/kaggle/input").rglob(config["zip_name"]))
    require(len(candidates) == 1, "Expected exactly one verified baseline ZIP attachment")
    archive_path = candidates[0]
    require(run_stage(PROGRESS,"baseline_zip_sha256",sha256,archive_path) == config["zip_sha256"], "Baseline ZIP differs")
    artifact_root = scratch / "selected_inputs"
    artifact_root.mkdir(exist_ok=False)
    selected = ["recovery_best_epoch_090.pt"] + ["dependencies/"+name for name in INFERENCE_FILES]
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP members")
        run_stage(PROGRESS,"checkpoint_manifest_identity",verify_checkpoint_manifests,archive,contract)
        for name in selected:
            log("selected_input_extract_start", member=name)
            destination = artifact_root / name
            require(destination.resolve().is_relative_to(artifact_root.resolve()), "Unsafe ZIP member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Only selected training inputs: reading to EOF verifies their ZIP CRC.
            with archive.open(name) as source, destination.open("xb") as output:
                for block in iter(lambda: source.read(1024*1024), b""): output.write(block)
            require(run_stage(PROGRESS,"selected_input_sha256",sha256,destination) == config["selected_hashes"][name], "Selected input differs: "+name)
            log("selected_input_extract_complete", member=name)
    artifacts = run_stage(PROGRESS,"train_artifacts_load",load_inference_artifacts,artifact_root/"dependencies",config["dependency_hashes"])
    require(hashlib.sha256(json.dumps(artifacts["prompt_ids"],separators=(",",":")).encode()).hexdigest() ==
            config["prompt_ids_sha256"], "Prompt ordering fingerprint differs")
    image_ids = select_nonprompt_ids(artifacts["train_ids"], artifacts["prompt_ids"])
    require(image_ids == config["image_ids"] and len(image_ids) == config["max_images"] and
            config["repeat_image_id"] == image_ids[0] and config["max_generations"] == len(image_ids)+1,
            "Fixed train-minus-prompts selection/repeat differs")
    require(len(artifacts["train_ids"])-len(artifacts["prompt_ids"]) == config["eligible_count"],
            "Eligible membership differs")
    log("cov_content_bound_data_resolution")
    resolved = run_stage(PROGRESS,"dataset_resolution",resolve_cov_dataset,Path("/kaggle/input"), config["data_root"], contract,
                         artifacts["train_ids"], artifacts["prompt_ids"], scratch, progress=log)
    (run_root/"data_resolution.json").write_text(json.dumps(resolved["evidence"],indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(phase="cov_data_resolved", **resolved["evidence"])),flush=True)
    paths = {identifier:resolved["images"][identifier] for identifier in image_ids}
    image_hashes = {identifier:sha256(filename) for identifier,filename in paths.items()}
    require(image_hashes == config["image_sha256"], "Selected image bytes differ from local raw archive")
    log("strict_epoch90_restore")
    decoder = run_stage(PROGRESS,"decoder_restore",restore_epoch90,artifact_root/"recovery_best_epoch_090.pt",device)
    log("pinned_clip_load")
    vision, preprocess, encoder = run_stage(PROGRESS,"clip_initialize",load_companion_clip,device,scratch/"clip_weights")
    direct_url = metadata.distribution("clip").read_text("direct_url.json")
    if direct_url:
        require(json.loads(direct_url).get("vcs_info",{}).get("commit_id") == CLIP_COMMIT,
                "CLIP VCS provenance differs")
    log("controlled_images_and_one_repeat")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.monotonic()
    def save_completed_generation(index, record):
        evidence = dict(status="COMPLETED_DIAGNOSTIC_IMAGE",run_id=config["run_id"],
                        checkpoint_sha256=EPOCH90_SHA256,generation_index=index+1,
                        image_sha256=image_hashes[record["image_id"]],record=record)
        with (run_root/("generation_"+str(index+1).zfill(2)+".json")).open("x",encoding="utf-8") as stream:
            json.dump(evidence,stream,indent=2,allow_nan=False)
        log("generation_saved", generation=index+1, image_id=record["image_id"])
    result = run_stage(PROGRESS,"six_generations",diagnose_images,vision,preprocess,decoder,artifacts["prompts"],artifacts["vocabulary"],
        image_ids,artifacts["train_ids"],artifacts["prompt_ids"],
        lambda identifier: Image.open(paths[identifier]),progress=log,on_record=save_completed_generation)
    torch.cuda.synchronize()
    inference_seconds = time.monotonic()-inference_start
    require(sha256(artifact_root/"recovery_best_epoch_090.pt") == EPOCH90_SHA256,
            "Checkpoint file changed")
    require(sha256(artifact_root/"dependencies/mclip_prompt_bank.pt") ==
            config["dependency_hashes"]["mclip_prompt_bank.pt"], "Prompt file changed")
    for record in result["records"]: record["image_sha256"] = image_hashes[record["image_id"]]
    result.update(run_id=config["run_id"],image_ids=image_ids,eligible_count=config["eligible_count"],
        checkpoint=dict(epoch=90,sha256=EPOCH90_SHA256,zip_sha256=config["zip_sha256"],
                        selection="available_physical_epoch90; historical85_missing"),
        encoder=encoder,dependencies={name:version(name) for name in
            ("torch","torchvision","numpy","Pillow","clip","ftfy","regex","tqdm","wcwidth")},
        python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        artifact_hashes=artifacts["hashes"],prompt_ids_sha256=config["prompt_ids_sha256"],config=config,
        source_sha256={name:sha256(Path(__file__).parent/name) for name in config["source_files"]},
        dataset_resolution=resolved["evidence"],dataset_contract_sha256=config["dataset_contract_sha256"],
        inference_seconds=inference_seconds,total_worker_seconds=time.monotonic()-base.START,
        gpu_memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       scope="after model load; reset before selected images plus repeat"),
        numerical_settings=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        scientific_limit="Training-only small diagnostic, not clinical scoring; decoder trained on these reports; historical encoder/cache revision unverified")
    require(result["generation_count"] <= 6 and result["image_count"] <= 5, "Generation bound exceeded")
    (run_root/"diagnostic.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    log("diagnostic_saved")
    print(json.dumps(dict(status=result["status"],image_ids=image_ids,comparisons=result["comparisons"])),flush=True)


if __name__ == "__main__":
    base.START = PROGRESS.started
    base.log = log  # Observe the unchanged Gate4 dependency helper in this worker only.
    try:
        diagnostic(destination,json.loads(Path(args.config).read_text(encoding="utf-8")))
    except Exception as error:
        if isinstance(error, CovDataError):
            (destination/"data_resolution_failure.json").write_text(
                json.dumps(error.diagnostic,indent=2)+"\n",encoding="utf-8")
        (destination/"failure.json").write_text(json.dumps(dict(status="FAILED",error_type=type(error).__name__,
            phase="worker_failure",state="failed",timestamp_utc=utc_now().isoformat(),
            error=str(error),elapsed_seconds=time.monotonic()-base.START,no_retry=True),indent=2)+"\n",encoding="utf-8")
        log("failed_no_retry",state="failed",error_type=type(error).__name__)
        raise
