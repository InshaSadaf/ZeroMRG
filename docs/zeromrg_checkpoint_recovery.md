# ZeroMRG Gate 1 checkpoint recovery

> **Gate 2 verification, 2026-10-11 IST:** the approved preflight loaded the actual Kaggle-mounted epoch90 checkpoint and computed SHA-256 `ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299`, matching the local canonical artifact. GPU decoder/112-buffer SGD restoration and disposable save/restore passed. Epoch85 best weights remain missing; the guard correctly rejected epoch8 without resetting history. Earlier statements about unexecuted GPU checks/remote latest identity describe the historical Gate 1 evidence. See [Gate 2 record](zeromrg_gate2_preflight.md); no full remote latest download or training is claimed.

Audit date: 2026-10-10. Workspace: `E:\Insha Project\ZeroMRG`. Starting evidence: [Gate 0](kaggle_gate0_assessment.md); its selected-best assertion is corrected here by inspecting the actual tensors and metadata. Scope is COV-CTR, 0% paired. No training, local real-model forward, remote execution, upload, or deletion occurred in this audit.

## Decision and evidence

**Epoch 90 is the canonical latest recovery checkpoint. The history-selected epoch-85 model is NOT recovered.** Model/SGD restoration passes; complete selected-best recovery is BLOCKED. An epoch number and BLEU score in history do not establish the existence of corresponding weights.

Machine-readable evidence: [archive audit](evidence/gate1_checkpoint_audit.json), [independent remote best audit](evidence/gate1_remote_best_audit.json). The complete extracted backup is retained in ignored `artifacts/reconciliation/gate1_20261010_checkpoint_audit/epoch90/`. The separately downloaded remote best is in `artifacts/reconciliation/gate1_20261010_remote_best/`. Credentials and weights are excluded from commits.

| Artifact | Verified metadata | Bytes | SHA-256 |
|---|---|---:|---|
| Downloads `zeromrg_covctr_training_backup (5).zip` | Canonical epoch-90 backup; ZIP CRC check passed, 15 members | 693642148 | `e3fc6449cfc8098ff43cefa9da704738d3f3898b3c17d7b3058caac5cec762e7` |
| Extracted `text_decoder_last.pt` | epoch 90; full decoder, optimizer, history and selection counters | 459275387 | `ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299` |
| Extracted `text_decoder_best.pt` | **epoch 8**, validation text BLEU-4 `52.33887482243904`, vocabulary 368, max tokens 160 | 229639049 | `dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568` |
| Remote `inshasadaf114/zeromrg-checkpoint-backup5-zip/text_decoder_best.pt` | **epoch 8**, same metadata and byte-identical to extracted best | 229639049 | `dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568` |
| `text_training_history.json` | 90 contiguous epochs; maximum at epoch 85 | See evidence JSON | `ccd65f2ec0ce2f569f683233f5a13c48c26bb27075aa9661bc7a272c3a14a015` |

The independent remote best has exactly the keys `decoder`, `epoch`, `max_tokens`, `validation_bleu4`, `vocabulary_size`; 112 float32 decoder tensors totaling 57,399,152 parameters. It has **no optimizer, scheduler, history or RNG state**. Its weights are architecturally valid, but it fails validation-selected best matching. No epoch-85 model was found in the accessible, bounded search.

## Search coverage and limits

Six local Downloads archives were inspected, including the unnumbered backup and backups `(1)` through `(5)`. Their saved latest epochs are 15, 30, 45, 60, 75, 90. All latest checkpoints pass the model/SGD/history audit. Every archived best has epoch-8 metadata and the same member CRC; history selects epoch 8 for epochs 15/30, epoch 35 for 45/60/75, and epoch 85 for epoch 90. The canonical extracted best and independently retrieved remote best additionally have matching SHA-256 hashes.

The search covered existing repository artifacts/checkpoint filenames, Downloads backup files and notebooks, Downloads/Compressed, bounded Documents and Documents/Codex listings (depth two), and sibling project directory names. Repository CPU-smoke outputs are diagnostic artifacts from a different tiny configuration, not candidates for this trained decoder. Large raw-image trees, virtual environments, Git internals, entire disks, cloud storage, live interactive RAM and unavailable older remote versions were not searched. Desktop was unavailable. This is a bounded negative finding, not proof that no epoch-85 copy exists anywhere.

Read-only authenticated Kaggle discovery found the owned backup5 dataset and refreshed workflow status. Current workflow and preprocessing kernel output file listings returned headers without checkpoint entries. Historical notebook output through epoch 120 is real log evidence, but supplies neither saved weights nor momentum. No persistent model-and-optimizer pair after epoch 90 was verified. The remote **latest** checkpoint was listed but not independently downloaded and hashed; matching name/size is not proof of identity with the local canonical latest. Only the remote **best** was independently downloaded in full and hash-verified.

## Epoch-90 restoration results

Saved top-level keys are `epoch`, `decoder`, `optimizer`, `history`, `best_validation_bleu4`, `epochs_without_improvement`. History covers 1 through 90; saved best is epoch 85 with `56.44423922571333`; saved non-improvement counter is **5**, agreeing with history. Next epoch is **91**, derived from metadata.

The checkpoint-producing architecture is six post-norm Transformer decoder layers, width 768, eight attention heads, feed-forward width 3072, GELU/dropout 0.1, learned 160-position embeddings, vocabulary 368 and one conditioning memory token. Token embeddings are unscaled and there is no final decoder normalization. All 112 state keys, shapes, float32 dtypes and finite tensors match this implementation. The newer modular decoder defaults are incompatible and were not used for recovery.

SGD has one parameter group: learning rate 0.001, momentum 0.99, dampening 0, weight decay 0, Nesterov false, maximize false; saved `foreach`/`fused` are null and `differentiable` false. All **112/112** parameters have correctly shaped, finite, nonzero momentum buffers. Strict CPU model loading and actual optimizer restoration preserved every saved momentum tensor exactly. Original source parameter order matches the recovered decoder; legacy state does not itself carry parameter names, so source reconciliation is necessary. Future checkpoints record names explicitly. There is no scheduler in the producing source or legacy state; this is consistent with the original constant-learning-rate experiment, not a missing scheduler to invent.

No RNG state, exact encoder revision, complete environment snapshot or complete configuration is stored in the legacy checkpoint. Matching model/optimizer state does **not** establish bitwise replay of the interrupted trajectory. A restarted dropout/data-order stream and package/hardware changes can produce different future results.

## Selected-best discrepancy and recovery policy

The workflow constructs `text_decoder_best_slim.pt` in a separate cell, then packages that file under `text_decoder_best.pt`. The archived stale epoch-8 best across six backups is consistent with the slimming cell not being rerun after improvements. This is a plausible explanation, not verified evidence of exactly which interactive cells were executed.

Missing epoch-85 weights do **not** prevent calculating the next SGD step or preserving the historical early-stopping comparison: the latest state includes the metric and counter. They **do** prevent returning the original selected best if subsequent epochs do not improve. They also prevent faithful image-to-report initialization from that selected checkpoint. Exact original stopping/model-selection recovery is therefore unavailable as a complete artifact pair.

The default continuation policy is fail closed: require a best artifact whose saved epoch and validation metric agree with history, then strictly check its architecture. Current inputs stop with `need epoch 85, found epoch 8`. Keep historical best `56.44423922571333` and patience `5/50`; never substitute epoch 90 or epoch 8, reset the metric, or fabricate weights. At unchanged best, 45 more non-improving epochs would stop at epoch 135. The 200-epoch ceiling is an inherited implementation assumption, not evidence that every run must reach 200.

Preferred recovery: obtain a genuine epoch-85 artifact from another authorized backup/version, inspect its schema, finite tensor shapes and provenance, record its SHA-256, and verify its metric/epoch against history before enabling continuation or best-selected inference. Shape/metadata matching alone cannot prove provenance of an arbitrary file.

If recovery remains impossible, a **separately approved recovery experiment** could continue epoch-90 SGD while preserving the historical metric and patience and explicitly tracking `historical_selected_weights_missing`. Only a future strict improvement above the historical maximum would create a recoverable new winner. If stopping occurs without improvement, final selection remains blocked; neither latest nor best available may be relabeled as the historical winner. This exceptional policy is documented only: it is not enabled in the generated notebook and has not been approved. It would not recreate the lost epoch-85 model or prove the original trajectory.

## Dependency integrity

Canonical feature/split/vocabulary SHA-256 values are in the archive audit and embedded in both notebooks. Key dependencies:

| File | SHA-256 |
|---|---|
| `train_manifest.csv` | `8902b8fea1163429aec6940a9744f7c792d1b7f9a00ca300a1fab17448ead973` |
| `validation_manifest.csv` | `94c555cb25a19485616065079139722bfe7122a17639267ba5e03d9e4cfa7c5d` |
| `test_manifest.csv` | `fcbcd1bd12220b0a3a286c4b1a82a8543e34c0a3495c5d0b8c89c81033b3eb66` |
| `prompt_manifest.csv` | `f60c45a1dc9beecba16917d767f1cf6b0f67231141a9226efdb771a27f9281de` |
| `decoder_vocabulary.json` | `c0e169c3558fe0cbf2e36431281aa223f64e2e4fb88f7361e23b33e12bcc3ad5` |
| `mclip_text_features.pt` | `7275ebb5ea1a9ef273eee38b16e45fe6a3f42f154bb844ca2ba890e09853863a` |
| `mclip_prompt_bank.pt` | `3cb770a2a470c70d95952828cf23abc088250571897b960a2a39e9bae6c8e1ce` |

Saved dependencies verify 714 usable pairs, splits 571/71/72, 500 unique training-only prompts, vocabulary 368 and 768-dimensional finite features. Runtime assertions also check raw 726 CSV rows and 746 physical images on Kaggle; those actual mounts have not been executed in this Gate 1 session. Cached normalized prompt/text vectors differ slightly (maximum absolute difference `0.0018524825573`, minimum cosine `0.99988132715`); cause and exact historical encoder revision remain UNVERIFIED. Preserve the caches; do not silently re-encode or normalize the raw prompt bank differently.
