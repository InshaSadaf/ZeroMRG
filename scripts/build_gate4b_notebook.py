"""Prepare one Gate4B notebook locally; never submit or execute it."""
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from build_gate4_notebook import ROOT, VERIFIED, cell, SUPERVISOR
from zeromrg_gate1_runtime import require, sha256
from zeromrg_gate4_runtime import load_inference_artifacts
from zeromrg_gate4b_runtime import select_nonprompt_ids

OUT = ROOT / "notebooks/gate4b/image_conditioning"
FILES = ("zeromrg_gate1_runtime.py", "zeromrg_gate4_runtime.py", "gate4_diagnostic_worker.py",
         "zeromrg_gate4b_runtime.py", "gate4b_diagnostic_worker.py", "zeromrg_cov_data_resolver.py",
         "zeromrg_gate4b_progress.py")
SUPERVISOR_4B = SUPERVISOR.replace("gate4_diagnostic_worker.py", "gate4b_diagnostic_worker.py").replace(
    "ONE training-image diagnostic; output:", "Gate4B <=5 distinct training images plus ONE repeat; output:").replace(
    "result['image_count'] == 1", "result['image_count'] == CONFIG['max_images'] and result['generation_count'] == CONFIG['max_generations']").replace(
    "'zeromrg_gate4_'", "'zeromrg_gate4b_'")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "RUN_ROOT = Path('/kaggle/working')",
    "from datetime import datetime, timezone\n"
    "supervisor_started = time.monotonic()\n"
    "def supervisor_record(phase, state='event', elapsed_seconds=None, **details):\n"
    "    return dict(phase=phase,state=state,timestamp_utc=datetime.now(timezone.utc).isoformat(),\n"
    "                elapsed_seconds=round(time.monotonic()-supervisor_started,6) if elapsed_seconds is None else elapsed_seconds, **details)\n"
    "print(json.dumps(supervisor_record('notebook_start',state='start')),flush=True)\nRUN_ROOT = Path('/kaggle/working')")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "RUN_ROOT.mkdir(exist_ok=False)",
    "RUN_ROOT.mkdir(exist_ok=False)\n"
    "(RUN_ROOT/'supervisor_status.json').write_text(json.dumps(supervisor_record('notebook_export_start',state='start'))+'\\n',encoding='utf-8')")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "print('Gate4B <=5 distinct training images plus ONE repeat; output:', RUN_ROOT, flush=True)",
    "(RUN_ROOT/'dataset-contract.json').write_text(json.dumps(DATASET_CONTRACT,indent=2)+'\\n',encoding='utf-8',newline='\\n')\n"
    "require(hashlib.sha256((RUN_ROOT/'dataset-contract.json').read_bytes()).hexdigest()==CONFIG['dataset_contract_sha256'], 'Exported contract bytes differ')\n"
    "print('Gate4B <=5 distinct training images plus ONE repeat; output:', RUN_ROOT, flush=True)")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "text=True, bufsize=1, start_new_session=True)",
    "text=True, bufsize=1, start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED':'1'})")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "print('Supervisor alive; elapsed',round(now-started,1),'seconds',flush=True)",
    "heartbeat=supervisor_record('supervisor_heartbeat',state='running',elapsed_seconds=round(now-supervisor_started,3),worker_returncode=process.poll())\n"
    "                output.write(json.dumps(heartbeat)+'\\n'); output.flush()\n"
    "                (RUN_ROOT/'supervisor_status.json').write_text(json.dumps(heartbeat)+'\\n',encoding='utf-8')\n"
    "                print(json.dumps(heartbeat),flush=True)")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "os.killpg(process.pid,signal.SIGKILL)\n                process.wait(timeout=10)\n                failure=dict(status='HARD_TIMEOUT',seconds=now-started,no_retry=True)\n                (RUN_ROOT/'supervisor_failure.json').write_text(json.dumps(failure,indent=2)+'\\n')",
    "failure=supervisor_record('worker_timeout',state='failed',elapsed_seconds=now-supervisor_started,status='HARD_TIMEOUT',seconds=now-started,error_type='TimeoutError',no_retry=True)\n"
    "                (RUN_ROOT/'supervisor_failure.json').write_text(json.dumps(failure,indent=2)+'\\n')\n"
    "                print(json.dumps(failure),flush=True)\n"
    "                try:\n"
    "                    os.killpg(process.pid,signal.SIGKILL)\n"
    "                except ProcessLookupError:\n"
    "                    pass\n"
    "                if process.poll() is None:\n"
    "                    process.wait(timeout=10)")
SUPERVISOR_4B = SUPERVISOR_4B.replace(
    "finally:\n    if process.poll() is None:",
    "except BaseException as error:\n"
    "    failure=supervisor_record('supervisor_exception',state='failed',error_type=type(error).__name__,no_retry=True)\n"
    "    (RUN_ROOT/'supervisor_exception.json').write_text(json.dumps(failure,indent=2)+'\\n',encoding='utf-8')\n"
    "    print(json.dumps(failure),flush=True)\n"
    "    raise\n"
    "finally:\n    if process.poll() is None:")


def image_dimensions(content):
    if content[:8] == b"\x89PNG\r\n\x1a\n" and content[12:16] == b"IHDR":
        width,height = struct.unpack(">II",content[16:24])
        require(width > 0 and height > 0, "Invalid PNG dimensions")
        return "PNG",width,height
    require(content[:2] == b"\xff\xd8", "Selected image must be PNG or JPEG")
    offset = 2
    while offset < len(content):
        require(content[offset] == 0xff, "Invalid JPEG marker")
        while offset < len(content) and content[offset] == 0xff: offset += 1
        require(offset < len(content), "Truncated JPEG marker")
        marker = content[offset]
        offset += 1
        if marker in (0xd8,0xd9,0x01) or 0xd0 <= marker <= 0xd7: continue
        require(offset+2 <= len(content), "Truncated JPEG segment")
        length = int.from_bytes(content[offset:offset+2],"big")
        require(length >= 2 and offset+length <= len(content), "Invalid JPEG segment length")
        if marker in (0xc0,0xc1,0xc2):
            require(length >= 8, "Invalid JPEG frame")
            height,width = struct.unpack(">HH",content[offset+3:offset+7])
            require(width > 0 and height > 0, "Invalid JPEG dimensions")
            return "JPEG",width,height
        require(marker != 0xda, "JPEG frame absent before image scan")
        offset += length
    raise ValueError("JPEG frame missing")


def selection(config):
    artifacts = load_inference_artifacts(VERIFIED/"dependencies", config["dependency_hashes"])
    ids = select_nonprompt_ids(artifacts["train_ids"], artifacts["prompt_ids"])
    require(hashlib.sha256(json.dumps(artifacts["prompt_ids"],separators=(",",":")).encode()).hexdigest() ==
            config["prompt_ids_sha256"], "Original prompt order differs")
    image_evidence = []
    # Only ZIP central directory and these five image bytes; no raw report files are opened.
    with zipfile.ZipFile(ROOT/"data/COV-CTR with English reports.zip") as archive:
        for identifier in ids:
            matches = [entry for entry in archive.infolist() if not entry.is_dir() and
                       Path(entry.filename).name == identifier]
            require(len(matches) == 1, "Selected raw image missing or ambiguous: "+identifier)
            content = archive.read(matches[0])  # CRC verified at EOF; no extraction.
            image_format,width,height = image_dimensions(content)
            image_evidence.append(dict(image_id=identifier,report_id=identifier,split="checkpoint_train",
                in_prompt_bank=False,archive_member=matches[0].filename,bytes=len(content),
                image_format=image_format,crc32=matches[0].CRC,sha256=hashlib.sha256(content).hexdigest(),width=width,height=height))
    return dict(train_count=len(artifacts["train_ids"]),prompt_count=len(artifacts["prompt_ids"]),
        eligible_count=len(artifacts["train_ids"])-len(artifacts["prompt_ids"]),
        selection_policy="First up to five train-minus-prompts IDs in saved canonical training order; no report content selection",
        report_identifier_contract="Canonical manifests/bank use image_id as the associated report-row identifier; no separate report_id field exists",
        images=image_evidence,prompt_ids_sha256=config["prompt_ids_sha256"],
        dependency_hashes=artifacts["hashes"],heldout_reports_accessed=False,
        report_text_used_for_selection=False,image_verification="PNG/JPEG signature and dimensions plus ZIP CRC; actual PIL decoding remains Kaggle guard")


def build():
    config = json.loads((ROOT/"notebooks/gate4/single_image/configuration.json").read_text())
    evidence = selection(config)
    # Reuse independently pinned ID/hash-only contract; no raw or held-out reports.
    contract_path = OUT/"dataset-contract.json"
    require(sha256(contract_path)=="0efeac6636bb971f27580cbe89eb792620feffeb13b42900c625b8309b277772",
            "Previously approved dataset identity contract changed")
    contract_text = contract_path.read_text(encoding="utf-8")
    contract = json.loads(contract_text)
    ids = [item["image_id"] for item in evidence["images"]]
    config.pop("image_id")
    config.update(run_id="ZM-010_GATE4B_STAGE_PROGRESS_V3",image_ids=ids,max_images=len(ids),
        repeat_image_id=ids[0],max_generations=len(ids)+1,eligible_count=evidence["eligible_count"],
        image_sha256={item["image_id"]:item["sha256"] for item in evidence["images"]},
        source_files=list(FILES),dataset_contract_sha256=hashlib.sha256(contract_text.encode()).hexdigest(),
        approved_dataset_attachment=contract["attachment"],
        approval_scope="D-021 cancelled-v2 local diagnosis/progress preparation only; no submission or cancellation; new execution approval required")
    sources = {name:(ROOT/"scripts"/name).read_text(encoding="utf-8") for name in FILES}
    hashes = {name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    setup = "import hashlib, json, sys, uuid\nfrom pathlib import Path\n"
    setup += "SOURCES = "+repr(sources)+"\nSOURCE_HASHES = "+repr(hashes)+"\nCONFIG = "+repr(config)+"\nDATASET_CONTRACT = "+repr(contract)
    intro = """# ZeroMRG Gate4B — image-conditioning diagnostic

Prepared only; **not submitted or executed**. Explicit new owner execution approval required. Reuse the successful Gate4 pinned environment, exact frozen CLIP ViT-L/14, epoch90 historical decoder/vocab/greedy max160, and unchanged ordered raw500 prompts/Eq6 attention/Eq5 residual/L2. No extra environment rebuild or Torch upgrade; only missing support packages and exact CLIP are installed as before.

Use the first up to five saved training IDs absent from the prompt bank (report-row identifier=image_id). Repeat the first image once, giving at most six forwards through the full generation pipeline. Train-only, no reference scoring or prompt changes. Reports may have been used in decoder text reconstruction; absence from prompts is not unseen-training generalization.

Capture actual raw visual and actual decoder-condition tensors without changing them. Save pairwise raw/enriched cosine matrices, exact feature differences, token/report comparisons and repeat-image determinism under identical settings. No seed reset or numerical backend change. Differing reports do not establish clinical correctness; identical reports do not alone prove an unused image condition. Features saved for independent verification. GPU memory observed after model loading.

Existing private COV/checkpoint datasets and verified baseline output; input/weight/source/config hashes checked. Exactly one worker, hard1800s, dependency commands300s, heartbeat30s and visible per-image/token phases. No retry. Scratch in /tmp; small JSON/logs/sources only in unique working output. If pins/mounts/selection/runtime fail, preserve failure and stop. Reverify target privacy and resources before authorized execution; no submission during preparation.

Mount recovery V2: only the approved owner/slug or short-slug attachment is searched. Bind the original annotation bytes, all746 physical image hashes, exact714 valid IDs, checkpoint571/71/72 manifests and ordered500 prompts. Accept a unique extracted/nested tree or original ZIP; extract canonical image bytes into /tmp only. Reject ambiguous roots or content/membership mismatch and save actionable JSON. Reference CSV bytes are used only for identity hashing, never for scoring. The prior submitted version1 is immutable evidence; this revised local notebook has not been submitted. Dataset differs from the paper population; do not claim numerical equivalence.

Local stage-progress revision after cancelled version2: preserve the exact frozen algorithm and dataset guards. Persist worker_start before framework imports; record stage start/completion/failure and dataset hashing/extraction progress. Persist each completed generation without extra inference. Write supervisor heartbeats to working files as well as flushed stdout; write timeout evidence before process-group kill. Explicit LF contract export is checked before child launch. These files improve surviving evidence but cannot guarantee Kaggle publishes outputs after cancellation. No v3 has been submitted.
"""
    cells = [cell(intro,"markdown"),cell(setup),cell(SUPERVISOR_4B)]
    for index,item in enumerate(cells): item["id"] = f"gate4b-{index:02d}"
    notebook = dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
        language_info=dict(name="python")),nbformat=4,nbformat_minor=5)
    OUT.mkdir(parents=True,exist_ok=True)
    filename = "ZeroMRG_COV_CTR_Gate4B_Image_Conditioning_Diagnostic.ipynb"
    meta = json.loads((ROOT/"notebooks/gate4/single_image/kernel-metadata.json").read_text())
    meta.update(id="inshasadaf114/zeromrg-covctr-gate4b-image-conditioning",title="ZeroMRG COVCTR Gate4B Image Conditioning",code_file=filename)
    for name,value in ((filename,notebook),("kernel-metadata.json",meta),("configuration.json",config),
                       ("selection.json",evidence)):
        (OUT/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    # Hash exact UTF-8/LF bytes on Windows and Kaggle alike.
    (OUT/"dataset-contract.json").write_text(contract_text,encoding="utf-8",newline="\n")
    print(json.dumps(dict(notebook=str(OUT/filename),selection=evidence),indent=2))


if __name__ == "__main__": build()
