# ZeroMRG COV-CTR checkpoint persistence procedure

Date: 2026-10-10. This procedure applies only to explicitly approved Kaggle execution of the 0%-paired experiment. No remote publication has been performed. `/kaggle/working` is writable session storage; the existence of a file there is not proof of durable recovery after an interrupted interactive session.

## Artifact responsibilities

| Artifact | Purpose and required contents |
|---|---|
| `text_decoder_last.pt` | Latest **resumable** state: completed epoch, full decoder, full SGD/momentum, explicit scheduler `None`, contiguous history, historical best metric and patience counter, architecture, parameter names/order, training configuration, dependency hashes, Python/Torch CPU/CUDA RNG states and provenance |
| `text_decoder_best.pt` | Actual validation-selected decoder, epoch/metric, vocabulary and positional dimensions, architecture/dependency metadata; need not contain the latest optimizer |
| History/configuration | Full per-epoch text-reconstruction losses and validation BLEU4; batch/optimizer/patience/ceiling, split and encoder assumptions, source/runtime/environment provenance |
| Frozen dependencies | Exact vocabulary, train/validation/test/validated/prompt manifests, validation audit, normalized text features and raw prompt bank; immutable files with SHA-256 |
| Evaluation artifacts | Separate stage-labelled image-to-report predictions/references, metric settings/scores, selected checkpoint hash, image/text encoder weight revisions, environment and execution provenance. No final evaluation artifacts exist yet |

Current legacy last state lacks RNG/configuration fields; this limitation is retained in provenance when migrating. The selected epoch85 is missing. Do not package the stale epoch8 file as a valid epoch85 best or reset selection history to make the package look complete.

## Every completed epoch, when training is approved

1. Keep the source dataset and attached backup immutable. Use a unique `/kaggle/working/zeromrg_gate1_<id>` output directory.
2. Evaluate the original validation text-reconstruction metric. On strict improvement only, atomically save the actual current best weights with their matching epoch/metric. Preserve prior best on ties or deterioration.
3. Validate latest state and selected-best metadata as a pair. Atomically save the latest full model/optimizer state **every epoch**, using a same-directory temporary file, flush/fsync and replace. Save contiguous history/configuration. RNG is captured after the epoch's validation; future restore replays saved RNG immediately before training, after diagnostics.
4. The latest checkpoint embeds history, so a crash between its atomic replacement and the external history write is detectable. Current notebooks refuse an external-history mismatch; reconcile the external file against the embedded checkpoint in a documented local recovery step instead of guessing an epoch. Latest and best are separate files, not a transactional pair: an interrupted improvement may similarly require selecting an intact earlier immutable bundle. Never silently combine conflicting versions.
5. Export an immutable epoch-labelled bundle after each five-epoch interval, at the end of the at-most-five-epoch session chunk, or on early stopping. The notebook copies the **current selected-best file**, not a stale separately generated slim copy. Include latest, best, history, configuration, all required immutable dependencies, the embedded runtime and per-file SHA-256 manifest. Produce ZIP and external ZIP SHA-256 sidecar. Check the package's storage size against available Kaggle disk before the next chunk; multiple immutable snapshots consume space. No existing checkpoint is deleted by this code.

New checkpoints capture the stochastic sources used by this decoder-only loop (Python and Torch CPU/CUDA, with the data-loader epoch seed explicit). This is not a promise of bitwise replay across hardware/PyTorch changes. Historical encoder features are reused and no stochastic NumPy processing is used in this loop. Record any future additional RNG dependency before adding it.

## Durability outside the session

The training chunk returns after at most five epochs; do not begin another chunk until the export is externally secured. An export message explicitly says `DURABILITY UNVERIFIED`.

**Practical local route:** download the immutable ZIP plus sidecar to a new non-overwriting backup location. Verify ZIP SHA-256, CRC/member hashes, load metadata with `weights_only=True`, inspect model/SGD/history and require the selected-best epoch/metric match. Log the recovered epoch, best epoch, file sizes/hash and storage location. This proves recovery from that retrieved copy, not availability of an untested remote version.

**Private Kaggle route (requires explicit approval):** prepare a clean dataset staging directory for an epoch-labelled private checkpoint dataset/version. Include only the audited bundle/manifest/configuration; never credentials. Use the installed wrapper's `datasets create` or `datasets version` interface after checking its help and reviewing exact staging metadata. This publishes remote artifacts and is not authorized by Gate 1 preparation. Existing remote datasets must not be overwritten without approval. Optionally save a private committed kernel version with outputs; this is also a remote execution/persistence action needing approval.

After publication, list the exact private dataset/version and independently download its backup into another new directory. Compare the downloaded ZIP and member hashes and re-run model/optimizer/best/history verification. Only then mark that version **retrievable and verified** in the recovery record. Listing success or an upload response alone is insufficient. Attach that exact audited version before another training chunk; verify the mounted hashes again.

Kaggle CLI `kernels push` executes a new version; it is not a harmless source-only upload. The installed CLI help has no `--no-run` option. Use [official kernel documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) and the installed command help for any execution settings; no push, Save Version, dataset publication or training is part of this completed local task.

## Current durable evidence

The local epoch90 archive and extracted dependencies are verified. The independently downloaded remote best file is retrievable, hash-identical and **stale epoch8**. A remotely retrieved epoch90 latest pair, genuine epoch85 best, or any later persistent checkpoint has not been verified during Gate 1. Do not label the proposed export strategy implemented-and-durable until a corresponding future export is actually retrieved and verified.
