"""Build ONE self-contained private diagnostic notebook. Never submit or execute."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "artifacts/gate3/baseline_retrieval_20261011T055727985114Z/verified_export_20261011T060635057524Z"
OUT = ROOT / "notebooks/gate4/single_image"
FILES = ("zeromrg_gate1_runtime.py","zeromrg_gate4_runtime.py","gate4_diagnostic_worker.py")


def cell(source,kind="code"):
    result=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True))
    if kind == "code": result.update(execution_count=None,outputs=[])
    return result


SUPERVISOR = r'''
import os, queue, signal, subprocess, threading, time
require = lambda condition, message: None if condition else (_ for _ in ()).throw(ValueError(message))
require(not globals().get('_gate4_attempted', False), 'Already attempted; no automatic retry')
require(Path('/kaggle/input').is_dir(), 'Kaggle-only execution; local execution prohibited')
_gate4_attempted = True
RUN_ROOT = Path('/kaggle/working') / ('zeromrg_gate4_' + uuid.uuid4().hex[:12])
RUN_ROOT.mkdir(exist_ok=False)
for name, source in SOURCES.items():
    require(hashlib.sha256(source.encode()).hexdigest() == SOURCE_HASHES[name], 'Embedded source differs')
    (RUN_ROOT/name).write_text(source, encoding='utf-8')
(RUN_ROOT/'configuration.json').write_text(json.dumps(CONFIG,indent=2)+'\n',encoding='utf-8')
print('ONE training-image diagnostic; output:', RUN_ROOT, flush=True)
print('Hard worker limit:', CONFIG['hard_timeout_seconds'], 'seconds; engineering bound, not measured feasibility', flush=True)
started = time.monotonic()
command = [sys.executable, '-u', str(RUN_ROOT/'gate4_diagnostic_worker.py'),
           '--run-root', str(RUN_ROOT), '--config', str(RUN_ROOT/'configuration.json')]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1, start_new_session=True)
messages = queue.Queue()
def read_output():
    for line in process.stdout: messages.put(line)
reader = threading.Thread(target=read_output,daemon=True)
reader.start()
last_heartbeat = started
try:
    with (RUN_ROOT/'execution.log').open('x',encoding='utf-8') as output:
        while process.poll() is None or reader.is_alive() or not messages.empty():
            now = time.monotonic()
            if now-started > CONFIG['hard_timeout_seconds']:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=10)
                failure=dict(status='HARD_TIMEOUT',seconds=now-started,no_retry=True)
                (RUN_ROOT/'supervisor_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise TimeoutError('Gate4 worker exceeded hard bound; no retry')
            try:
                line=messages.get(timeout=1)
                output.write(line); output.flush(); print(line,end='',flush=True)
            except queue.Empty:
                pass
            if now-last_heartbeat >= 30:
                print('Supervisor alive; elapsed',round(now-started,1),'seconds',flush=True)
                last_heartbeat=now
    require(process.returncode == 0, 'Diagnostic failed; inspect execution.log/failure.json; no retry')
    result=json.loads((RUN_ROOT/'diagnostic.json').read_text(encoding='utf-8'))
    require(result['image_count'] == 1 and result['split'] == 'checkpoint_train' and
            not result['test_references_accessed'] and not result['optimizer_created'], 'Diagnostic scope differs')
    print('Prepared diagnostic result:', RUN_ROOT/'diagnostic.json',flush=True)
finally:
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)
'''


def build():
    verification=json.loads((ROOT/"docs/evidence/gate3_baseline_verification.json").read_text())
    manifest=json.loads((VERIFIED/"sha256_manifest.json").read_text())["files"]
    with (VERIFIED/"dependencies/train_manifest.csv").open(encoding="utf-8-sig",newline="") as handle:
        image_id=next(csv.DictReader(handle))["image_id"]
    with (VERIFIED/"dependencies/prompt_manifest.csv").open(encoding="utf-8-sig",newline="") as handle:
        prompt_ids=[row["image_id"] for row in csv.DictReader(handle)]
    names=("train_manifest.csv","prompt_manifest.csv","decoder_vocabulary.json","mclip_prompt_bank.pt")
    selected=("recovery_best_epoch_090.pt",)+tuple("dependencies/"+name for name in names)
    config=dict(run_id="ZM-010_GATE4_SINGLE_TRAIN_IMAGE_V1",image_id=image_id,
        data_root="/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports",
        zip_name=verification["zip"]["file"],zip_sha256=verification["zip"]["sha256"],
        selected_hashes={name:manifest[name]["sha256"] for name in selected},
        dependency_hashes={name:manifest["dependencies/"+name]["sha256"] for name in names},
        prompt_ids_sha256=hashlib.sha256(json.dumps(prompt_ids,separators=(",",":")).encode()).hexdigest(),
        support_lock=(ROOT/"requirements/gate4-clip-support.lock").read_text(),
        hard_timeout_seconds=1800,max_images=1,max_tokens=160,training_enabled=False,
        test_inference_enabled=False,approval_scope="Prepared only; execution requires separate owner approval")
    sources={name:(ROOT/"scripts"/name).read_text(encoding="utf-8") for name in FILES}
    hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    setup="import hashlib, json, sys, uuid\nfrom pathlib import Path\n"
    setup+="SOURCES = "+repr(sources)+"\nSOURCE_HASHES = "+repr(hashes)+"\nCONFIG = "+repr(config)
    intro="""# ZeroMRG Gate4 — one frozen training-image diagnostic

Prepared locally; **not submitted or executed**. Owner approval is required before running this notebook on Kaggle. No training or metrics. Exactly one fixed image from the original checkpoint training CSV; no validation/test references are opened. Epoch90 is the latest verified physical model, with missing historical85 disclosed.

Existing private dataset attachments supply COV-CTR and checkpoint backup. The independently verified epoch90 model/prompt bundle comes from the existing private baseline kernel output, SHA-pinned; no new dataset publication. If mount/runtime/privacy/quota or pins differ, stop for review.

Image → frozen official OpenAI CLIP ViT-L/14 → raw500 prompt Eq6 attention → Eq5 residual/L2 → historical six-layer Transformer → greedy report, EOS or max160 including BOS. The paper states equations/freezing/500 prompts/EOS-or-limit; decoder details and greedy choice preserve the producing implementation. Historical encoder/cache revision is unverified. No M-CLIP re-encoding or text-encoder download is needed for the fixed bank.

Worker hard limit1800 seconds, dependency command limit300 seconds, heartbeat30 seconds; these are engineering safeguards, not measured runtime or GPU-budget predictions. Failure preserves logs/JSON, kills the worker group, and never retries. Source/config/checkpoint/encoder/dependency fingerprints and a small diagnostic JSON are saved in a unique output directory. Fluent text does not establish clinical correctness or reproduction of Table1.
"""
    cells=[cell(intro,"markdown"),cell(setup),cell(SUPERVISOR)]
    for index,item in enumerate(cells): item["id"]=f"gate4-single-{index:02d}"
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
                  language_info=dict(name="python")),nbformat=4,nbformat_minor=5)
    OUT.mkdir(parents=True,exist_ok=True)
    filename="ZeroMRG_COV_CTR_Gate4_Single_Image_Diagnostic.ipynb"
    (OUT/filename).write_text(json.dumps(notebook,indent=1)+"\n",encoding="utf-8")
    meta=dict(id="inshasadaf114/zeromrg-covctr-gate4-single-image-diagnostic",
        title="ZeroMRG COVCTR Gate4 Single Image Diagnostic",code_file=filename,language="python",
        kernel_type="notebook",is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        dataset_sources=["inshasadaf114/dataset","inshasadaf114/zeromrg-checkpoint-backup5-zip"],
        kernel_sources=["inshasadaf114/zeromrg-covctr-gate3-recovery-baseline"],competition_sources=[],model_sources=[])
    (OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    (OUT/"configuration.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(OUT/filename)


if __name__ == "__main__":
    build()
