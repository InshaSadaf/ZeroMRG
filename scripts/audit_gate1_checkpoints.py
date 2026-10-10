"""Bounded, CPU-only artifact audit. Never train or run the real decoder forward."""
import argparse
import gc
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch

from zeromrg_gate1_runtime import (
    CheckpointReportDecoder, inspect_training_state, restore_weights_optimizer,
    sha256, validate_artifacts, validate_selected_best, validate_model_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "CPU tensor/schema inspection only; no real model forward or training", "archives": []}
    candidates = sorted(args.downloads.glob("zeromrg_covctr_training_backup*.zip"))
    for candidate in candidates:
        entry = {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}
        with zipfile.ZipFile(candidate) as archive:
            names = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
            entry["members"] = [{"name": info.filename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
                                for info in archive.infolist()]
            require_names = {"text_decoder_last.pt", "text_decoder_best.pt", "text_training_history.json"}
            if not require_names <= names.keys():
                entry["status"] = "BLOCKED: required members absent"
            else:
                with io.BytesIO(archive.read(names["text_decoder_last.pt"])) as handle:
                    last = torch.load(handle, map_location="cpu", weights_only=True)
                with io.BytesIO(archive.read(names["text_decoder_best.pt"])) as handle:
                    best = torch.load(handle, map_location="cpu", weights_only=True)
                with torch.device("meta"):
                    model = CheckpointReportDecoder()
                entry["last_keys"] = sorted(last)
                entry["best_keys"] = sorted(best)
                entry["verification"] = inspect_training_state(last, model)
                validate_model_state(best, model)
                entry["best_artifact_epoch"] = best["epoch"]
                entry["best_artifact_metric"] = best["validation_bleu4"]
                try:
                    validate_selected_best(last, model, best)
                    entry["best_selection_integrity"] = "PASS"
                except ValueError as error:
                    entry["best_selection_integrity"] = "BLOCKED: " + str(error)
                external_history = json.loads(archive.read(names["text_training_history.json"]))
                if external_history != last["history"]:
                    raise ValueError(f"External history differs: {candidate}")
                entry["external_history_matches"] = True
                entry["optimizer_group"] = {key: value for key, value in last["optimizer"]["param_groups"][0].items()
                                            if key != "params"}
                entry["crc_integrity"] = "PASS" if archive.testzip() is None else "FAIL"
                entry["status"] = "PASS"
                # No parameter initialization: assign weights into a metadata-only model.
                if last["epoch"] == 90:
                    model.load_state_dict(last["decoder"], strict=True, assign=True)
                    optimizer, restored = restore_weights_optimizer(last, model)
                    entry["actual_restore"] = restored
                    recovery = args.output / "epoch90"
                    recovery.mkdir(exist_ok=True)
                    hashes = {}
                    for basename, member in names.items():
                        destination = recovery / basename
                        digest = hashlib.sha256()
                        if destination.exists():
                            with archive.open(member) as source:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    digest.update(block)
                            if sha256(destination) != digest.hexdigest():
                                raise ValueError(f"Refusing to overwrite conflicting extraction: {destination}")
                        else:
                            with archive.open(member) as source, destination.open("xb") as out:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    digest.update(block)
                                    out.write(block)
                        hashes[basename] = digest.hexdigest()
                    entry["canonical_member_sha256"] = hashes
                    artifacts = validate_artifacts(recovery)
                    entry["dependency_assertions"] = "PASS"
                    entry["dependency_hashes"] = artifacts["dependency_hashes"]
                    entry["prompt_text_comparison"] = artifacts["prompt_text_comparison"]
                    entry["prompt_norm_range"] = [float(artifacts["prompt"]["embeddings"].norm(dim=-1).min()),
                                                  float(artifacts["prompt"]["embeddings"].norm(dim=-1).max())]
                    del artifacts, optimizer
                del last, best, model
                gc.collect()
        report["archives"].append(entry)
        print(candidate.name, entry.get("verification", {}), flush=True)
        (args.output / "checkpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
