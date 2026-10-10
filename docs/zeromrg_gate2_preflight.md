# ZeroMRG Gate 2 bounded Kaggle preflight

Executed after explicit owner approval of one preflight only. Submission date: 2026-10-10 UTC; results collected 2026-10-11 Asia/Calcutta. Scope: COV-CTR 0%-paired, no decoder training, continuation, final zero-shot inference, test scoring or checkpoint overwriting.

**Gate 2: BLOCKED by missing validation-selected epoch 85 weights. The bounded diagnostic behaved as designed.** Kaggle version 1 ended with `KernelWorkerStatus.ERROR` because the notebook deliberately raises after writing its BLOCKED report. Latest decoder restoration, optimizer restoration and GPU forward checks passed.

## Submission and safety evidence

Read [Gate 1 readiness](zeromrg_gate1_readiness.md) and [checkpoint recovery](zeromrg_checkpoint_recovery.md) before execution. The unchanged preflight SHA-256 was `3d9731b0e36bf0aa0ab591f3dfa370aabeb6512f0670d6201c141f0ca7b54991`. Static cell syntax, embedded source and approved attachment metadata were checked; the notebook contains no training loop, backward call, optimizer step or held-out scoring. It reads checkpoint inputs from immutable mounted datasets and writes only a unique diagnostic output directory. Missing selected-best weights cause a report and exception before CLIP import/loading.

The complete nine-entry owned-kernel listing showed no existing preflight target. The authorized command created a new kernel version1:

```powershell
.\scripts\kaggle.ps1 -KaggleArgs @('kernels','push','-p','notebooks/gate1/preflight','-t','1200')
```

The installed client forwards `is_private`, dataset attachments and the 1200-second session timeout. After submission, independently pulled server metadata verified **private**, GPU enabled, TPU disabled, and exactly `inshasadaf114/dataset` plus `inshasadaf114/zeromrg-checkpoint-backup5-zip`. Retrieved cell source matched the audited submission; the pulled JSON byte hash differs due to server serialization: `dd0946729f6b07c353f589699bc98478fe88ed506fab3f72ca1aab5414e6be18`.

Target: [private preflight](https://www.kaggle.com/code/inshasadaf114/zeromrg-covctr-gate1-preflight). It initially queued for roughly 23 minutes before RUNNING was observed at 18:33:52 UTC. The collected execution log's last elapsed timestamp is approximately 47.03 seconds; billed GPU time was not measured. Terminal ERROR status was confirmed after output retrieval. No second version, continuation kernel, dataset mutation or Git push was submitted.

## Actual retrieved results

| Check | Result and observed evidence |
|---|---|
| Mounted paths | PASS: `/kaggle/input/datasets`; exact checkpoint and COV-CTR raw paths resolved |
| Raw population | PASS: 726 report rows, 746 physical images; duplicate-ID assertion passed |
| Saved population/splits | PASS: 714 usable pairs, 571 train / 71 validation / 72 test; original split reused, no regeneration |
| Dependencies/provenance | PASS: pinned hashes, report/image correspondence, training-only vocabulary 368 and 500 ordered training prompts; 768-D text/prompt dimensions |
| Latest load | PASS: epoch 90; next 91 derived; checkpoint SHA matches canonical local latest |
| Decoder compatibility | PASS: actual historical six-layer post-norm decoder, width 768 / eight heads / FFN 3072, 160 positions, unscaled embeddings / no final norm; 57,399,152 parameters |
| Optimizer/momentum | PASS: complete SGD restoration; 112/112 finite nonzero momentum buffers; restored buffers equal saved values; no optimizer update |
| History/selection state | Preserved: best history epoch 85, BLEU4 `56.44423922571333`, non-improvement counter 5; no metric/patience reset |
| GPU/memory | Tesla T4; 14.45977783203125 GiB free of 14.56219482421875 GiB **before decoder loading**; peak and post-load free memory were not measured |
| Finite forward | PASS: one fixed training report; CE `0.08689973503351212`; finite outputs, no gradients or backward/update. This is a diagnostic loss, not an image-generation or test metric |
| Output writability | PASS: write/read probe; retrieved `write_probe.txt` contains `writable` |
| Disposable save/restore | PASS: isolated full-state serialization/reload, exact weights/momentum, identical restored logits; disposable artifact removed by its own temporary-directory context |
| Best checkpoint | BLOCKED: history requires 85 but saved file identifies 8; no substitution |
| Frozen image/zero-shot path | NOT REACHED: stopped before CLIP import, model download or image generation |

Canonical checkpoint SHA-256 computed on the **actual Kaggle mount** is `ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299`. This now establishes identity with the verified local latest through executed, retrieved diagnostics; it is not a claim of independently downloading the remote459MB latest file.

Disposable diagnostic SHA-256: `df7aa91ad241a380ae37a0a89c02f349450589970f6a31802ef16bb30b5dbe0b`. It is not a new durable recovery checkpoint; the disposable file no longer exists. The original source checkpoints were never written. Canonical local last/best/history hashes were rechecked and remain byte-identical to Gate 1.

## Exception and environment

The report records `Best checkpoint disagrees with history: need epoch 85, found epoch 8. Recover matching weights; do not substitute latest weights.` The intended terminal exception is:

```text
ValueError: PREFLIGHT BLOCKED: validation-selected best weights unavailable. Latest restoration checks passed.
```

Papermill wraps this in `PapermillExecutionError` at input6, yielding Kaggle ERROR. Debugger frozen-module warnings and nbconvert/mistune escape-sequence warnings also appear; they did not prevent the preceding checks. No timeout, out-of-memory or earlier restoration exception appears in the retrieved execution log.

Observed environment: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, NumPy 2.1.3, pandas 2.3.3, Pillow 12.3.0, transformers 5.16.1. `clip`, `multilingual-clip`, and `pycocoevalcap` were **not installed**. They were not needed for the executed cached-text diagnostic. The first is needed for the image path; pycocoevalcap/Java are needed for continuation validation. Java availability was not checked. No dependency installation occurred. Historical PyTorch 2.10 and absent original RNG/encoder revisions still limit exact trajectory reproduction despite this successful restoration.

## Retrievable evidence and interpretation

[Original preflight report](evidence/gate2_preflight_report.json), [original execution log](evidence/gate2_execution_log.json), and [audit summary](evidence/gate2_preflight_audit.json) are retained as small local evidence. Raw submission/status/source/output snapshots are in ignored `artifacts/gate2/preflight_20261010T181029719Z`.

The terminal `kernels files --csv` request returned only a header. Nevertheless, `kernels output` actually retrieved the report, write probe and log. Artifact retrieval, not the empty listing, is the positive evidence here. Report SHA-256 is `607b4d92d49698eaf69f4d8d874ef88ccc63f7c3db130412abcbd225341b0ca2`; log SHA-256 is `a4e30bc0f559f4b5c2adc82a7700e6112b9b666bb0a7fa30401a0726699e26c5`. Successful retrieval of diagnostics does not establish durability of any newly trained checkpoint; no training occurred and no new recovery checkpoint was exported.

This discrepancy also limits negative conclusions from earlier header-only kernel file listings. A future authorized checkpoint search should test actual output retrieval as well as listings before declaring a remote output absent. No other kernel's outputs were retrieved in this Gate 2 task, and no new epoch85 weights have been verified.

Operational latest-state preflight passed and the selected-best guard worked. Overall readiness remains BLOCKED: recover and verify the provenance of genuine epoch 85 weights before default continuation or the selected-best image diagnostic. Preserve historical best/patience. A separately approved incomplete-selection policy remains only a documented option; it was not enabled. No final test metrics or valid selected-best zero-shot result are available.

The authorized preflight is complete. **No further remote execution or training is approved or automatic.** Do not rerun this notebook, install packages remotely, push continuation, or enable training from these results without a new explicit authorization.
