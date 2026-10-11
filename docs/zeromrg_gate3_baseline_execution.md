# Gate 3 baseline execution evidence

Verified 2026-10-11. **PASS: the single authorized baseline-only version completed, and its exported recovery checkpoint was independently retrieved and verified.** This is validation text reconstruction, not an image-to-report result. No training session is authorized by this result.

## Existing run and source provenance

Actual private notebook: `inshasadaf114/zeromrg-covctr-gate3-recovery-baseline`, version **1**, numeric ID **138127332**. The authenticated project wrapper reports `KernelWorkerStatus.COMPLETE`. Interruption of the local assistant did not interrupt Kaggle. This verification session performed only status/output reads and local CPU inspection; it did not push, resubmit, publish a dataset, or launch another run.

Prepared metadata requested the shorter `zeromrg-covctr-gate3-recovery` ID. Kaggle's original submission warning reported creation of the title-derived `zeromrg-covctr-gate3-recovery-baseline` slug. Retrieved server metadata confirms privacy and exactly these attachments: `inshasadaf114/dataset` and `inshasadaf114/zeromrg-checkpoint-backup5-zip`. All 11 retrieved server cells match the submitted notebook; file hashes differ because Kaggle reserialized JSON. Preserve both files and the warning. Future submission metadata must be explicitly reviewed against the actual target; do not reuse an old push command automatically.

| Source artifact | SHA-256 |
|---|---|
| Submitted notebook | `46dc67a9833d7b698968e3afea10604f9520512e784db563fedc9d7ea7d0872b` |
| Prepared kernel metadata | `3093a64fcf9fd2cce962310d51a9c94368c90147f69bc2edab6a5eec59a46720` |
| Retrieved server notebook | `afafde1f3aed4810a37e44d56a6d4b26f1258436b63caab6e5e9f693e90dba60` |

Original submission/source evidence remains in `artifacts/gate3/baseline_20261011T054542853281Z`. Preparation-document snapshots and their manifest were preserved in its `pre_execution_docs/` before updating these documents. Original Gate 1/2 reports and historical checkpoint evidence remain unchanged. [Execution audit](evidence/gate3_baseline_execution_audit.json) records status, privacy/source checks and the reviewed working tree; its source comparison accounts for Kaggle cell-metadata serialization.

## Measured results and execution boundaries

| Check | Actual result |
|---|---|
| Dataset/dependencies | Raw/dependency mounts verified; 571 train / 71 validation / 72 test; original vocabulary 368, text width 768, training-only prompt bank 500; nine dependency hashes match submitted pins |
| Epoch90 restoration | Canonical `text_decoder_last.pt` loaded; six-layer historical decoder, 57,399,152 parameters, 112 state tensors |
| Optimizer | All 112 finite nonzero SGD momentum buffers restored; LR 0.001, momentum 0.99, dampening/weight decay 0, Nesterov false; scheduler none |
| Validation | Ordered IDs and references match all 71 established validation reports; cached text conditioning, greedy decoding, PTB tokenization and corpus BLEU4 ×100 |
| Actual baseline BLEU4 | **54.21140850351637**; difference from historical epoch90 score **0.0** |
| Recovery winner | Physical epoch90 weights saved as `recovery_best_epoch_090.pt`; selection scope `recovery_epochs_90_onward`; explicitly not the missing epoch85 model |
| History | Original 90 records unchanged, continuation history empty, recovery history one fresh baseline measurement |
| Historical best/stopping | Epoch85 / **56.44423922571333**, counter **5**, patience **50**, all unchanged |
| New full-state export | Decoder, SGD, scheduler none, configuration, independent ledgers, dependencies, metric/runtime provenance, Python/NumPy/Torch CPU and two CUDA RNG states present |
| No updates | Submitted training flag false; final baseline-only branch logged; every saved decoder tensor and momentum buffer bitwise equal to original90 |
| Scope | No image encoding/report inference, held-out scoring, training backward or optimizer update occurred |

The fresh measurement reached the log at about 50.14 seconds, export preparation at about 67.90 seconds, and the final baseline-only message at about 67.92 seconds. The last execution-log entry is at 77.752748028 seconds, including notebook conversion. These are log timings, not GPU billing or training-epoch measurements.

Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, Tesla T4. Before model loading, reported free/total device memory was 15,526,068,224 / 15,636,037,632 bytes (14.460 / 14.562 GiB). Peak memory was not measured. Two CUDA RNG states were exported; the report identifies the selected device as Tesla T4 and does not inventory every device's memory.

## Independent artifact verification

New non-overwriting retrieval directory: `artifacts/gate3/baseline_retrieval_20261011T055727985114Z`. Read-only Kaggle output listing/download retrieved the diagnostic files, raw log, sidecar and ZIP from the actual existing notebook. Streaming retrieval avoided downloading redundant loose checkpoint copies; all required copies are in the ZIP. The signed download URLs and credentials were not stored in evidence.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `recovery_latest.pt` | 459,295,737 | `f641f7ef7312e74b2c79c0b76f1920498999a90328f296f50d75950784a4e404` |
| `recovery_best_epoch_090.pt` | 229,639,967 | `6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2` |
| `recovery_export_epoch_090_54468b70.zip` | 1,382,728,028 | `dec3cd3391e2c107a7c9ed6209997f00554cd47cb0940ca339a74446253680b1` |
| `validation_epoch_090.json` | 62,048 | `71ca8ba5c4c19de34deb805fff764db59ece359873de7c3581fd68e9e4c38d4d` |

ZIP hash/size match the sidecar, CRC passes, all 26 manifest-listed artifacts and both runtime sources match their hashes, and historical originals/reports and nine dependencies match the audited pins. The manifest, scripts and files have an exact archive-membership check. Extraction rejects traversal/symlinks and never overwrites a prior attempt.

CPU safe loading (`weights_only=True`) and architecture/schema inspection pass. All 112 decoder tensors in latest and best equal original90 bit for bit; all 112 latest momentum buffers and their parameter ordering equal original90. Decoder tensor-content SHA-256 is `ec179ec2be9a9792144e5e877bf5d09f179043db6775b374eaf49598934c15a6`. Metadata, selection pointer, best-file SHA, predictions SHA, score, validation membership and original stopping state agree across checkpoint/history/configuration/runtime report.

Verified extraction: `artifacts/gate3/baseline_retrieval_20261011T055727985114Z/verified_export_20261011T060635057524Z`. Use its **`recovery_latest.pt`** for a future approved restart, not the model-only best file. [Machine-readable verification](evidence/gate3_baseline_verification.json) records the exact paths and checks. [Raw runtime report](evidence/gate3_baseline_runtime_report.json) and [execution log](evidence/gate3_baseline_execution_log.json) are preserved. The raw report's `durable_retrieval_verified:false` was written before remote retrieval; it is intentionally unchanged. The independent verification now demonstrates retrieval of this completed version's export, not a separately published private dataset or perpetual platform retention.

Scores were recomputed from saved predictions on Kaggle. Local verification checked their hashes and cross-artifact agreement; it did not rerun Java/PTB locally, where Java is unavailable. This distinction limits claims of independent metric recomputation.

## Failures, warnings and remaining requirements

No Kaggle traceback, installation failure, metric failure, checkpoint failure or filesystem failure was found. Locked isolated `pycocoevalcap==1.2` installed and matched its wheel/source/JAR hashes. OpenJDK **21.0.12** executed the pinned Stanford PTB 3.4.1 fixture and validation/recomputation successfully. Frozen-module debugger messages and notebook-conversion escape warnings were nonfatal.

The first local verification attempt completed artifact/state checks but failed while formatting its result: a relative input path was compared with an absolute repository root. The correction resolves the input path and uses a unique extraction directory on retry. The first extraction was preserved. This was a local reporting error, not a Kaggle/checkpoint/metric failure; the second complete verification passed.

Original epoch85 weights remain unavailable; epoch8 remains a stale historical best. Historical encoder revisions, original RNG state and exact historical metric revisions remain unknown. Newly saved RNG states make future recovery boundaries explicit; they cannot reconstruct the lost original trajectory. CLIP/M-CLIP runtime and image-path fidelity are still unverified. Baseline agreement does not verify encoder cache provenance or establish zero-shot medical generation performance.

**Decision:** Gate 3 baseline **PASS**. A technical epoch91 starting state is verified under policy A's disclosed recovery-window selection, preserving historical early stopping. The complete reproduction remains conditional; missing original85 and encoder provenance are unresolved. Decoder training remains unauthorized. The next proposed action is owner review/approval of a new private durable recovery attachment, followed by a separately approved bounded epoch91+ session after metadata/checkpoint/environment review. No next execution or publication was performed.
