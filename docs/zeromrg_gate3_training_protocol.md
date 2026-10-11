# ZeroMRG Gate 3 training protocol

Scope: COV-CTR 0%-paired, original preprocessing/split/features, Kaggle GPU only. **Prepared, not executed.** [Policy decision](zeromrg_gate3_recovery_decision.md) declares the recovery deviation and preserves the original experiment. The separate [notebook](../notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb) embeds the unchanged Gate 1 decoder/runtime plus new recovery helpers; it needs no repository clone or `src` imports. Adjacent private kernel metadata uses a new slug, never the old workflow/preflight target.

## Initialization and execution boundaries

Default `ENABLE_APPROVED_RECOVERY_TRAINING=False`. Future authorized default execution installs only the locked isolated BLEU package, validates data/state, measures epoch90 validation, saves a genuine recovery baseline, verifies save/reload and prepares an export. It performs no image inference, optimizer step/backward or held-out scoring. Training calls occur only inside a separately guarded function. Changing the flag requires **new explicit remote training authorization** after the baseline artifacts are retrieved and reviewed.

Default checkpoint: `/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip/text_decoder_last.pt`. Select an explicit path; no newest-file guessing. Original dependency/historical source directory stays the exact backup mount even when selecting a later Gate 3 `recovery_latest.pt`. Raw mount: `/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports`. Metadata attaches only `inshasadaf114/dataset` and `inshasadaf114/zeromrg-checkpoint-backup5-zip` for the first baseline run. A future restart needs an additionally approved new private recovery dataset; its metadata and path must be reviewed before that run. Extracted recovery bundle layout has `recovery_latest.pt`, winner files and validation measurements at the root.

Legacy initialization accepts only the canonical SHA-256 epoch90 file and original history/stale8 hashes. Strict model/optimizer/history inspection precedes loading. Restore **all112** SGD momentum buffers, LR0.001, momentum0.99, dampening0, weight decay0, NesterovFalse, maximizeFalse. No scheduler was used; save `scheduler=None` and reject unknown scheduler state. No scratch fallback or newly initialized optimizer replaces a missing state.

The code rejects enabling training directly from the legacy90 file. Training must initialize from a separately retrieved/reviewed complete Gate 3 baseline or later recovery checkpoint; this enforces the baseline-first review sequence rather than combining unreviewed baseline creation and updates in one run.

Original model: six post-norm Transformer decoder layers, width768,8heads, FFN3072 GELU/dropout0.1, learned160 positions, vocab368, unscaled token embeddings and no final decoder norm. Do not substitute modular defaults. Float32, batch6, token-masked teacher-forcing cross entropy, clip gradient norm5.0. These checkpoint-producing implementation choices are not invented paper architecture claims. Existing normalized768-D M-CLIP text caches condition training/validation; raw500x768 training-only prompts remain dependencies. No encoder retraining or cache regeneration occurs.

Assertions reuse the Gate 2-verified implementation:726 raw rows,746 physical images,714 validated pairs,571 train/71 validation/72 test, unique disjoint IDs, training-only vocabulary/prompt bank, exact nine dependency hashes, report/image correspondence, normalized text cache and raw prompt dimensions. Test manifest checks verify integrity/disjointness only. Test data never enters decoder losses, predictions, metric selection or recovery calibration.

## Fresh90 baseline and validation selection

Evaluate physical90 weights in eval/no-grad mode on exactly the ordered71 validation records, using their cached text features. Greedy decode BOS1 to EOS2 or160 total tokens, then unchanged vocabulary postprocessing. Compute corpus BLEU4 with pinned COCO PTB lowercase/punctuation tokenization and closest-reference scoring, multiplied by100. Save ordered IDs, references, predictions, measured score, metric implementation fingerprint and SHA-256. No score is fabricated from old logs/history.

Immediately save `recovery_best_epoch_090.pt` with these **actual90 weights**, measured score and prediction hash; verify the saved tensors equal the original90 decoder. It is labelled a recovery baseline, `historical_selected_epoch=85`, `is_missing_epoch85_model=False`, selection scope90 onward. This does not fill the original85 gap. Record difference from historical90 score54.21140850351637 for diagnosis only. Exact historical metric/package/RNG revisions are unavailable; a material difference requires review before training, not quiet alteration of original values.

After each separately approved training epoch, regenerate all71 validation reports and record the measured score. Select only on strict `score > current recovery-window best`; ties retain the earlier saved winner. Write the immutable new winner before atomically writing latest state with its file/hash pointer. All saved selection metadata must match the independently recorded validation score/prediction hash. Scores are recomputed from saved predictions during baseline/restoration. Save/reload checks compare weights and every momentum buffer. Source checks and synthetic tests do not substitute for executing this new schema on Kaggle.

## Early stopping and budget

Keep original fixed evidence85/56.44423922571333 and5/50. The separate original stopping tracker begins with these values and considers **only new91+ scores** thereafter. A strict original-best improvement resets its counter; otherwise increment. A baseline90 remeasurement neither resets patience nor changes historical best. Stop after counter reaches50 or total epoch200. The independent recovery-selection counter never controls stopping.

If no new score exceeds56.44423922571333, the remaining45 non-improving epochs end at **135**. If original-best improvements occur, training may continue up to total200, at most110 updates after90. The200 ceiling is inherited implementation configuration, not a paper-specified convergence result. Use at most **five new epochs per separately approved session**, fewer if stopping occurs. The first approved run is baseline-only (zero updates). GPU seconds per validation/training epoch remain unmeasured; review baseline runtime and retrieval before allocating larger timeouts. Do not silently extend epochs, patience, learning rate or precision.

## Interruption recovery

New resumable checkpoints include decoder, exact optimizer parameter order and buffers, schedulerNone, original immutable history/record, continuation and recovery histories, both selection/stopping records, best file/hash, configuration, dependency hashes, metric fingerprint, runtime/environment provenance, and Python/NumPy/TorchCPU/allCUDA RNG state. NumPy state is serialized using lists/plain types and Torch version strings are normalized for safe `weights_only=True` loading.

Legacy90 contains no RNG state. Establish an explicitly documented new seed42; historical stochastic trajectory cannot be reconstructed. New training uses deterministic per-epoch sampler generator42+epoch and captures global RNG at completed boundaries. This is a new reproducible recovery convention, not proof that original shuffle/dropout trajectory was identical. Restore RNG again immediately before training after diagnostics/export. No mixed precision or new optimizer schedule is introduced.

Restart only from a complete selected Gate 3 checkpoint. Require policy/schema, original history, split/features/vocabulary, model/optimizer, current best hash/metadata, validation IDs/references/score hashes, RNG and exact recorded metric/environment signatures. Drift stops for review; no automatic reseeding, best reset or patience reset. Selected missing/corrupt files fail. No orphan new-best file is auto-selected if a crash occurred before updating latest. The latest checkpoint is authoritative; a secondary history JSON can be regenerated from it after an interrupted ancillary write.

A crash midway through an epoch discards incomplete updates; resume from the last **retrievable completed** epoch. Local atomic saving protects complete on-disk writes, not against loss of an interrupted Kaggle filesystem. A restart from a downloaded durable checkpoint is interruption-safe at that boundary; `/kaggle/working` alone is not durable. See the [persistence procedure](zeromrg_gate3_persistence.md).

Only one training attempt is allowed per kernel, even if it fails before completing an epoch. A persistent marker survives re-execution of the definition cell. After an exception, restart the kernel/job and explicitly reload a verified completed checkpoint; never retry against partially updated in-memory weights or momentum. A synthetic failure test verifies this guard without any model update.

## Approval sequence

1. Approve one private baseline-only version with locked BLEU installation and timeout1800; training staysFalse.
2. Retrieve status/log, actual baseline/predictions, verified current model/SGD state and ZIP; verify hashes/metadata locally without GPU work. If Java/metric/environment/data checks fail, diagnose and obtain approval for any revised remote execution.
3. Review scientific calibration, original stopping and recovery baseline. Publish a **new private** durable recovery dataset only after explicit approval; verify retrieval before using it.
4. Obtain separate decoder-training authorization for a bounded chunk; select an explicit verified recovery checkpoint and review revised attachment metadata/flag. Retrieve and verify every chunk before another approval.
5. After defensible stopping and a validation-selected retrievable winner, obtain approval for frozen image-path diagnostics and eventually final test evaluation. Historical encoder revisions and future inference dependencies remain unresolved; no final image-generation validity is claimed now.

## Source control review

Current baseline is `e4fe4a07aa2125d95b359a3cae3e8fcb2d87b932`, which already contains the completed Gate 1/2 work. The pre-existing `.gitignore` addition for `.kaggle/` is meaningful and retained. This task adds only Gate 3 docs/small evidence, new notebook/metadata/locks, source builders/auditors/recovery helpers and targeted tests, plus a short progress annotation. Gate 1/2 reports/notebooks are preserved and hash checked. Generated large historical copies, downloaded inspection wheels/resolver details and logs remain under ignored `artifacts/`; credentials and `.pt` files are not part of the proposed commit. No whitespace cleanup, staging, commit, discard, remote notebook/dataset publication or Git push was performed.

Controlled commit plan for owner review: (1) policy/protocol/dependency/persistence docs, progress annotation and small evidence; (2) new Gate 3 runtime, builders/auditors/static validator, CPU tests, notebook/metadata and locks. Review actual staged file names/diff, avoid including user-owned `.gitignore` changes without separate review, and run `git diff --check` on staged content. No commit or push is authorized in this task. Full working-tree review and delivery hashes are recorded in `docs/evidence/gate3_final_review.json`.
