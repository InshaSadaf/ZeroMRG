"""Gate4B observations around the unchanged Gate4 inference path; CPU mocks on import."""
import torch
from zeromrg_gate1_runtime import require
from zeromrg_gate4_runtime import FrozenImageToReport, assert_frozen


def select_nonprompt_ids(train_ids, prompt_ids, limit=5):
    require(type(limit) is int and 1 <= limit <= 5, "Select one to five images")
    require(len(train_ids) == len(set(train_ids)) and len(prompt_ids) == len(set(prompt_ids)),
            "Duplicate canonical train/prompt IDs")
    require(set(prompt_ids) <= set(train_ids), "Prompts must be training-only")
    selected = [identifier for identifier in train_ids if identifier not in set(prompt_ids)][:limit]
    require(bool(selected), "No non-prompt training images available")
    return selected


def compare_runs(runs):
    """Compare observed tensors in float64 on CPU; this does not change conditioning."""
    require(2 <= len(runs) <= 6, "Expected selected images plus one repeat")
    unique = runs[:-1]
    require(len({r["image_id"] for r in unique}) == len(unique) and
            runs[-1]["image_id"] == unique[0]["image_id"], "Invalid unique/repeat order")
    matrices = {}
    pairs = []
    for name in ("raw", "enriched"):
        vectors = torch.stack([r[name].reshape(-1).double() for r in unique])
        require(tuple(vectors.shape) == (len(unique), 768) and bool(torch.isfinite(vectors).all()) and
                bool((vectors.norm(dim=-1) > 0).all()), "Invalid captured features")
        unit = vectors / vectors.norm(dim=-1, keepdim=True)
        matrices[name] = (unit @ unit.T).clamp(-1, 1).tolist()
    for i in range(len(unique)):
        for j in range(i+1, len(unique)):
            a, b = unique[i], unique[j]
            pairs.append(dict(image_ids=[a["image_id"], b["image_id"]],
                raw_cosine=matrices["raw"][i][j], enriched_cosine=matrices["enriched"][i][j],
                raw_features_exact_equal=torch.equal(a["raw"], b["raw"]),
                enriched_features_exact_equal=torch.equal(a["enriched"], b["enriched"]),
                raw_max_abs_difference=float((a["raw"]-b["raw"]).abs().max()),
                enriched_max_abs_difference=float((a["enriched"]-b["enriched"]).abs().max()),
                token_sequences_equal=a["result"]["output_tokens"] == b["result"]["output_tokens"],
                report_text_equal=a["result"]["report_text"] == b["result"]["report_text"]))
    first, repeat = unique[0], runs[-1]
    raw_equal = torch.equal(first["raw"], repeat["raw"])
    enriched_equal = torch.equal(first["enriched"], repeat["enriched"])
    token_equal = first["result"]["output_tokens"] == repeat["result"]["output_tokens"]
    report_equal = first["result"]["report_text"] == repeat["result"]["report_text"]
    return dict(matrix_image_ids=[r["image_id"] for r in unique], pairwise_cosine=matrices,
        pairs=pairs, different_raw_features_observed=any(not p["raw_features_exact_equal"] for p in pairs),
        different_enriched_features_observed=any(not p["enriched_features_exact_equal"] for p in pairs),
        different_token_sequences_observed=any(not p["token_sequences_equal"] for p in pairs),
        different_reports_observed=any(not p["report_text_equal"] for p in pairs),
        distinct_report_count=len({r["result"]["report_text"] for r in unique}),
        determinism=dict(image_id=first["image_id"], total_executions_of_this_image=2,
            raw_features_exact_equal=raw_equal, enriched_features_exact_equal=enriched_equal,
            tokens_exact_equal=token_equal, report_exact_equal=report_equal,
            raw_max_abs_difference=float((first["raw"]-repeat["raw"]).abs().max()),
            enriched_max_abs_difference=float((first["enriched"]-repeat["enriched"]).abs().max()),
            all_observed_outputs_exact_equal=raw_equal and enriched_equal and token_equal and report_equal),
        sensitivity_conclusion=("REPORT_CHANGES_OBSERVED" if any(not p["report_text_equal"] for p in pairs)
                                else "NO_REPORT_CHANGES_OBSERVED_IN_THIS_SMALL_SET"),
        clinical_correctness_assessed=False)


@torch.inference_mode()
def diagnose_images(vision, preprocess, decoder, prompts, vocabulary, image_ids,
                    train_ids, prompt_ids, open_image, progress=None, on_record=None):
    require(image_ids == select_nonprompt_ids(train_ids, prompt_ids), "Fixed saved selection differs")
    progress = progress or (lambda phase: None)
    frozen_prompt_copy = prompts.detach().clone()
    require(not prompts.requires_grad, "Prompt bank must be frozen")
    versions = [[p._version for p in module.parameters()] for module in (vision, decoder)]
    runs = []
    for index, image_id in enumerate(image_ids + [image_ids[0]]):
        repeat = index == len(image_ids)
        progress("image_"+str(index+1)+("_repeat_" if repeat else "_")+image_id)
        captured = {}
        def raw_hook(module, inputs, output):
            require("raw" not in captured, "More than one encoder call per image")
            captured["raw"] = output.detach().float().cpu().clone()
        def condition_hook(module, inputs):
            if "enriched" not in captured:
                captured["enriched"] = inputs[0].detach().float().cpu().clone()
        encoder_hook = vision.register_forward_hook(raw_hook)
        decoder_hook = decoder.register_forward_pre_hook(condition_hook)
        try:
            pipeline = FrozenImageToReport(vision, preprocess, decoder, prompts, vocabulary)
            with open_image(image_id) as image:
                result = pipeline.generate_one(image, max_length=160, progress=progress)
        finally:
            encoder_hook.remove()
            decoder_hook.remove()
        for key in ("raw", "enriched"):
            require(key in captured and tuple(captured[key].shape) == (1, 768) and
                    bool(torch.isfinite(captured[key]).all()), "Missing/invalid feature capture: "+key)
        for module in (vision, decoder): assert_frozen(module)
        require(versions == [[p._version for p in module.parameters()] for module in (vision, decoder)],
                "Model parameter mutation detected")
        require(torch.equal(prompts, frozen_prompt_copy), "Original prompt bank mutated")
        runs.append(dict(image_id=image_id, repeat=repeat, raw=captured["raw"],
                         enriched=captured["enriched"], result=result))
        if on_record is not None:
            on_record(index, dict(image_id=image_id,report_id=image_id,split="checkpoint_train",
                in_prompt_bank=False,repeat=repeat,raw_embedding=captured["raw"].tolist(),
                enriched_embedding=captured["enriched"].tolist(),**result))
        progress("generation_completed_"+str(index+1))
    comparisons = compare_runs(runs)
    # Features are small (<=6 x2 x768), retained for independent cosine/difference verification.
    records = [dict(image_id=r["image_id"], report_id=r["image_id"], split="checkpoint_train",
                    in_prompt_bank=False, repeat=r["repeat"], raw_embedding=r["raw"].tolist(),
                    enriched_embedding=r["enriched"].tolist(), **r["result"]) for r in runs]
    return dict(status="COMPLETED_IMAGE_CONDITIONING_DIAGNOSTIC", image_count=len(image_ids),
        generation_count=len(records), split="checkpoint_train", records=records, comparisons=comparisons,
        all_modules_frozen=True, gradients_disabled=True, parameter_version_counters_unchanged=True,
        prompt_bank_unchanged=True, optimizer_created=False, paired_supervision_used=False,
        test_references_accessed=False, test_scoring=False)
