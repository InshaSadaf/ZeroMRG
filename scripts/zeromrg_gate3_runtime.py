"""Separate recovery selection with historical stopping; no execution on import."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from zeromrg_gate1_runtime import (
    ARCHITECTURE, TRAINING, atomic_save, capture_rng, inspect_training_state,
    load_checkpoint, require, restore_rng, restore_weights_optimizer, sha256,
    validate_model_state,
)

ORIGINAL_FILES = {
    "text_decoder_last.pt": "ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299",
    "text_decoder_best.pt": "dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568",
    "text_training_history.json": "ccd65f2ec0ce2f569f683233f5a13c48c26bb27075aa9661bc7a272c3a14a015",
}
ORIGINAL_RECORD = dict(epoch=90, best_epoch=85, best_validation_bleu4=56.44423922571333,
                       epochs_without_improvement=5, patience=50,
                       selected_weights_available=False, stale_best_epoch=8,
                       files=ORIGINAL_FILES)
RECOVERY_POLICY = dict(id="A_historical_stop_available_best_v1", baseline_epoch=90,
                       selection="strict_validation_BLEU4_improvement_in_epochs_90_onward",
                       stopping="original_history_plus_continuation_patience_50",
                       max_epoch=200, max_epochs_per_session=5,
                       historical_weights_missing=True,
                       deviation="selection_restricted_to_recoverable_window; legacy_RNG_lost")
RECOVERY_SCHEMA = "zeromrg_gate3_dual_ledger_v1"


def plain_metadata(value):
    """Normalize str subclasses (including TorchVersion) into safe plain types."""
    return json.loads(json.dumps(value, allow_nan=False))


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def capture_recovery_rng():
    import numpy as np
    result = capture_rng()
    name, keys, position, has_gauss, cached = np.random.get_state()
    result["numpy"] = dict(name=name, keys=keys.tolist(), position=int(position),
                            has_gauss=int(has_gauss), cached=float(cached))
    return result


def restore_recovery_rng(state):
    import numpy as np
    require(isinstance(state, dict) and {"python", "torch_cpu", "torch_cuda", "numpy"} <= state.keys(),
            "Required recovery RNG state absent")
    restore_rng(state)
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached"]))


def seed_legacy_recovery():
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def historical_stop_record(original_history, continuation_history):
    history = original_history + continuation_history
    selected = max(history, key=lambda row: row["validation_bleu4"])
    return dict(best_epoch=selected["epoch"], best_validation_bleu4=float(selected["validation_bleu4"]),
                epochs_without_improvement=history[-1]["epoch"] - selected["epoch"], patience=50)


def safe_name(name):
    require(isinstance(name, str) and name and Path(name).name == name and
            name not in {".", ".."} and "/" not in name and "\\" not in name,
            "Artifact reference must be a relative basename")
    return name


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_measurement(row):
    required = {"epoch", "stage", "validation_bleu4", "predictions_file", "predictions_sha256",
                "metric_fingerprint_sha256"}
    require(isinstance(row, dict) and required <= row.keys(), "Validation measurement fields absent")
    require(type(row["epoch"]) is int and row["epoch"] >= 90, "Invalid measured epoch")
    require(math.isfinite(float(row["validation_bleu4"])) and 0 <= float(row["validation_bleu4"]) <= 100,
            "Invalid measured BLEU4")
    safe_name(row["predictions_file"])
    require(valid_digest(row["predictions_sha256"]) and valid_digest(row["metric_fingerprint_sha256"]),
            "Invalid measurement integrity digest")


def new_recovery_metadata(original_history, baseline, recovery_id, hashes, metric_fingerprint, provenance):
    validate_measurement(baseline)
    require(baseline["epoch"] == 90 and baseline["stage"] == "baseline", "Must measure epoch90 baseline first")
    require(len(original_history) == 90, "Original history must contain exactly90 epochs")
    require(baseline["metric_fingerprint_sha256"] == json_sha256(metric_fingerprint), "Baseline metric identity differs")
    return dict(schema_version=RECOVERY_SCHEMA, epoch=90, recovery_id=recovery_id,
                recovery_policy=copy.deepcopy(RECOVERY_POLICY), historical_record=copy.deepcopy(ORIGINAL_RECORD),
                original_history=copy.deepcopy(original_history), continuation_history=[],
                recovery_history=[copy.deepcopy(baseline)],
                original_stop_tracking=historical_stop_record(original_history, []),
                dependency_hashes=copy.deepcopy(hashes), metric_fingerprint=plain_metadata(metric_fingerprint),
                provenance=plain_metadata(provenance))


def append_recovery_epoch(metadata, epoch, training_loss, measurement):
    """Pure metadata update; no optimizer/model operation and no historical mutation."""
    validate_measurement(measurement)
    require("decoder" not in metadata and "optimizer" not in metadata, "Pass metadata only, not model/optimizer tensors")
    require(epoch == metadata["epoch"] + 1 and measurement["epoch"] == epoch and
            measurement["stage"] == "continuation", "Continuation epoch is not the next completed epoch")
    require(epoch <= 200 and metadata["original_stop_tracking"]["epochs_without_improvement"] < 50,
            "Original stopping condition already met")
    require(math.isfinite(float(training_loss)), "Non-finite training loss")
    require(measurement["metric_fingerprint_sha256"] == json_sha256(metadata["metric_fingerprint"]),
            "Metric implementation changed")
    current_best = max(metadata["recovery_history"], key=lambda row: row["validation_bleu4"])
    improved = float(measurement["validation_bleu4"]) > float(current_best["validation_bleu4"])
    result = copy.deepcopy(metadata)
    # This function accepts metadata only, never a payload containing model tensors.
    result["epoch"] = epoch
    result["continuation_history"].append(dict(epoch=epoch, training_loss=float(training_loss),
                                               validation_bleu4=float(measurement["validation_bleu4"])))
    result["recovery_history"].append(copy.deepcopy(measurement))
    result["original_stop_tracking"] = historical_stop_record(result["original_history"], result["continuation_history"])
    return result, improved


def recovery_best_payload(model, measurement, recovery_id, dependency_hashes):
    validate_measurement(measurement)
    return dict(schema_version="zeromrg_gate3_recovery_best_v1", decoder=model.state_dict(),
                epoch=measurement["epoch"], validation_bleu4=float(measurement["validation_bleu4"]),
                validation_predictions_sha256=measurement["predictions_sha256"],
                metric_fingerprint_sha256=measurement["metric_fingerprint_sha256"],
                recovery_id=recovery_id, selection_scope="recovery_epochs_90_onward",
                historical_selected_epoch=85, is_missing_epoch85_model=False,
                architecture=model.architecture, vocabulary_size=model.architecture["vocab_size"],
                max_tokens=model.architecture["max_tokens"], dependency_hashes=dependency_hashes)


def exclusive_save(payload, path):
    require(not Path(path).exists(), "Immutable artifact already exists; refuse overwrite")
    atomic_save(payload, path)


def best_reference(path, measurement):
    return dict(file=safe_name(Path(path).name), sha256=sha256(path), epoch=measurement["epoch"],
                validation_bleu4=float(measurement["validation_bleu4"]),
                predictions_file=safe_name(measurement["predictions_file"]),
                predictions_sha256=measurement["predictions_sha256"])


def latest_recovery_payload(metadata, model, optimizer, reference):
    result = copy.deepcopy(metadata)
    result.update(decoder=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=None,
                  architecture=model.architecture, parameter_names=[n for n, _ in model.named_parameters()],
                  training_config=TRAINING, recovery_best=copy.deepcopy(reference), rng_state=capture_recovery_rng())
    return result


def metadata_from_payload(payload):
    return {key: copy.deepcopy(payload[key]) for key in
            ("schema_version", "epoch", "recovery_id", "recovery_policy", "historical_record", "original_history",
             "continuation_history", "recovery_history", "original_stop_tracking", "dependency_hashes",
             "metric_fingerprint", "provenance")}


def legacy_view(payload):
    stop = payload["original_stop_tracking"]
    return dict(epoch=payload["epoch"], decoder=payload["decoder"], optimizer=payload["optimizer"],
                history=payload["original_history"] + payload["continuation_history"],
                best_validation_bleu4=stop["best_validation_bleu4"],
                epochs_without_improvement=stop["epochs_without_improvement"],
                scheduler=payload["scheduler"], architecture=payload["architecture"],
                parameter_names=payload["parameter_names"], training_config=payload["training_config"],
                rng_state=payload["rng_state"])


def validate_recovery_state(payload, model, original_history, hashes, best=None):
    require(isinstance(payload, dict), "Recovery checkpoint must be a dictionary")
    required = {"schema_version", "epoch", "recovery_id", "recovery_policy", "historical_record", "original_history",
                "continuation_history", "recovery_history", "original_stop_tracking", "dependency_hashes",
                "metric_fingerprint", "provenance", "decoder", "optimizer", "scheduler", "architecture",
                "parameter_names", "training_config", "recovery_best", "rng_state"}
    require(required <= payload.keys(), "Required recovery state absent: " + str(sorted(required - payload.keys())))
    require(payload["schema_version"] == RECOVERY_SCHEMA and payload["recovery_policy"] == RECOVERY_POLICY,
            "Unknown or altered recovery policy")
    require(payload["historical_record"] == ORIGINAL_RECORD and payload["original_history"] == original_history,
            "Original historical evidence changed")
    require(payload["dependency_hashes"] == hashes, "Split/feature/vocabulary dependencies changed")
    require(payload["scheduler"] is None, "This recovery protocol has no scheduler")
    epoch = payload["epoch"]
    require(type(epoch) is int and 90 <= epoch <= 200, "Recovery epoch outside approved protocol")
    continuation = payload["continuation_history"]
    require(isinstance(continuation, list) and all(isinstance(row, dict) and
            {"epoch", "training_loss", "validation_bleu4"} <= row.keys() for row in continuation),
            "Continuation history fields absent")
    require(len(original_history) == 90 and len(continuation) == epoch - 90, "Historical/continuation lengths disagree")
    require([row["epoch"] for row in continuation] == list(range(91, epoch + 1)), "Continuation history is not contiguous")
    tracking = historical_stop_record(original_history, continuation)
    require(payload["original_stop_tracking"] == tracking, "Original stopping ledger changed/reset")
    for index in range(len(continuation)):
        require(historical_stop_record(original_history, continuation[:index])["epochs_without_improvement"] < 50,
                "Continuation advanced beyond original early stopping")
    rows = payload["recovery_history"]
    require(len(rows) == epoch - 89 and [r["epoch"] for r in rows] == list(range(90, epoch + 1)),
            "Recovery selection history is not contiguous")
    for index, row in enumerate(rows):
        validate_measurement(row)
        require(row["stage"] == ("baseline" if index == 0 else "continuation"), "Recovery measurement stage differs")
        require(row["metric_fingerprint_sha256"] == json_sha256(payload["metric_fingerprint"]),
                "Validation metric fingerprints disagree")
        if index:
            require(float(row["validation_bleu4"]) == float(continuation[index - 1]["validation_bleu4"]),
                    "Validation score differs between ledgers")
    selected = max(rows, key=lambda row: row["validation_bleu4"])
    reference = payload["recovery_best"]
    require(isinstance(reference, dict) and {"file", "sha256", "epoch", "validation_bleu4", "predictions_file",
                                          "predictions_sha256"} <= reference.keys(), "Recovery best reference fields absent")
    safe_name(reference["file"])
    require(valid_digest(reference["sha256"]), "Invalid selected-weight checksum")
    require(reference["epoch"] == selected["epoch"] and reference["validation_bleu4"] == selected["validation_bleu4"] and
            reference["predictions_file"] == selected["predictions_file"] and
            reference["predictions_sha256"] == selected["predictions_sha256"], "Recovery best does not match measured validation history")
    require(isinstance(payload["rng_state"], dict) and
            {"python", "torch_cpu", "torch_cuda", "numpy"} <= payload["rng_state"].keys(), "Recovery RNG state absent")
    result = inspect_training_state(legacy_view(payload), model)
    if best is not None:
        require(best.get("schema_version") == "zeromrg_gate3_recovery_best_v1" and
                best.get("recovery_id") == payload["recovery_id"] and best.get("epoch") == selected["epoch"] and
                best.get("validation_bleu4") == selected["validation_bleu4"] and
                best.get("validation_predictions_sha256") == selected["predictions_sha256"] and
                best.get("metric_fingerprint_sha256") == selected["metric_fingerprint_sha256"] and
                best.get("dependency_hashes") == hashes and best.get("selection_scope") == "recovery_epochs_90_onward" and
                best.get("is_missing_epoch85_model") is False and best.get("historical_selected_epoch") == 85,
                "Selected recovery weights have invalid provenance/score")
        validate_model_state(best, model)
    result.update(recovery_best_epoch=selected["epoch"], recovery_best_validation_bleu4=selected["validation_bleu4"],
                  recovery_non_improvement_counter=epoch - selected["epoch"],
                  historical_rng_missing=True,
                  stop=tracking["epochs_without_improvement"] >= 50 or epoch >= 200)
    return result


def validate_recovery_files(payload, model, original_history, hashes, directory, validation_ids):
    directory = Path(directory)
    ref = payload["recovery_best"]
    best_path = directory / safe_name(ref["file"])
    require(sha256(best_path) == ref["sha256"], "Selected recovery weights checksum mismatch")
    best = load_checkpoint(best_path)
    result = validate_recovery_state(payload, model, original_history, hashes, best)
    for row in payload["recovery_history"]:
        path = directory / safe_name(row["predictions_file"])
        require(sha256(path) == row["predictions_sha256"], "Validation predictions checksum mismatch")
        prediction = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(prediction, dict) and {"split", "image_ids", "epoch", "validation_bleu4",
                "metric_fingerprint_sha256", "references", "predictions"} <= prediction.keys(),
                "Validation prediction fields absent")
        require(prediction["split"] == "validation" and prediction["image_ids"] == validation_ids,
                "Validation predictions contain wrong IDs/split; possible leakage")
        require(len(prediction["references"]) == len(validation_ids) == len(prediction["predictions"]) and
                all(isinstance(text, str) for text in prediction["references"] + prediction["predictions"]),
                "Validation references/predictions length or types differ")
        require(prediction["epoch"] == row["epoch"] and prediction["validation_bleu4"] == row["validation_bleu4"] and
                prediction["metric_fingerprint_sha256"] == row["metric_fingerprint_sha256"], "Saved validation measurement differs")
    return result


def restore_recovery(payload, model, original_history, hashes, directory, validation_ids, device="cpu"):
    result = validate_recovery_files(payload, model, original_history, hashes, directory, validation_ids)
    optimizer, _ = restore_weights_optimizer(legacy_view(payload), model, device)
    # RNG restored again immediately before training, after any diagnostics/evaluation.
    restore_recovery_rng(payload["rng_state"])
    return optimizer, result
