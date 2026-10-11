# Gate 3 first controlled recovery training session: epochs91–95

Authorized 2026-10-11: **one private Kaggle session, epochs91–95 inclusive only**, or earlier documented stop. No second session, image-to-report inference, test scoring, dataset mutation or Git push is authorized.

## Pre-execution evidence

All safety/provenance checks passed before submission. The separate notebook is `notebooks/gate3/training_091_095/ZeroMRG_COV_CTR_Recovery_Epochs_091_095_Kaggle.ipynb`; private target is `inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95`. Exact attachments: `inshasadaf114/dataset`, `inshasadaf114/zeromrg-checkpoint-backup5-zip`, and existing private baseline **kernel output** `inshasadaf114/zeromrg-covctr-gate3-recovery-baseline`. No new dataset publication is needed or performed. Kernel outputs are supported by [official Kaggle metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md); hashes pin the exact baseline identity despite the CLI source reference being a slug.

Read-only server checks confirmed baseline private/current version1/COMPLETE and the original source cells unchanged. Its verified export is currently listed. A target pull returned ambiguous HTTP403; this was not treated as absence. The complete authenticated owned-kernel catalog independently confirmed no existing training target. No original remote notebook is overwritten.

Input full latest90 SHA: `f641f7ef7312e74b2c79c0b76f1920498999a90328f296f50d75950784a4e404`; recovery-best90 SHA: `6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2`; export SHA: `dec3cd3391e2c107a7c9ed6209997f00554cd47cb0940ca339a74446253680b1`. ZIP size is1,382,728,028 bytes. Mounted input must match these hashes, CRC and manifest before restoration; missing/changed mounts stop before updates.

Historical checkpoint-producing architecture, SGD0.001/momentum0.99, batch6, clipping5, schedulerNone, original571/71/72 split, vocabulary368, width768 and training-only prompts500 remain unchanged. All112 momentum buffers and full dual-ledger state were inspected on CPU. Original best85/56.44423922571333/counter5/patience50 and recovery baseline90/54.21140850351637 are explicit starting assertions. New RNG state is restored; per-epoch sampler seed42+epoch follows the already declared recovery convention. No missing historical RNG is fabricated.

Start must equal91, requested new epochs must equal5, end must equal95, loop clamps to95, and every iteration asserts91≤epoch≤95. One-attempt guard prevents rerunning against partly updated memory. Four invalid-range fixtures passed without reaching training; all14 existing CPU recovery tests passed. Notebook cells compile; original baseline notebook/metadata and embedded audited runtime/metric/dependency cells are unchanged. Prepared notebook SHA `b331e603e50654334ea61f8463208a6c5d28ad675fa3a920a058ec9baa0e5fbc`; metadata SHA `4768be549eda663d9aa9937db6d4010384e75c314691b25124c1d0d267cf7657`.

Precheck/source snapshot: `artifacts/gate3/training_091_095_20261011T062341734415Z`; active state: `artifacts/gate3/active_training_091_095.json`. Timeout ceiling7200 seconds is a safety bound, not an observed runtime or permission for extra epochs.

## Model selection and persistence

Strict validation-only improvement over the recovery-window best selects and saves physical weights, including improvements below56.44423922571333. That yields a reproducible recoverable decoder without reproducing the unavailable original-best score. Historical85 evidence stays immutable; only new continuation scores update the separate original stopping tracker. Recovery-window patience never controls stopping. Test72 has integrity/disjointness checks only and no predictions or scores.

Every completed epoch atomically saves full resumable latest; genuine new winners have immutable files. Completion records include loss, validation BLEU, learning rate, stopping/selection ledgers, update count, elapsed time and latest/winner/prediction hashes. A final restoration/recomputation check precedes the final export. ZIP includes full state/RNG, original evidence, dependencies/configuration/metric provenance, histories, winners and completion records. Successful saving alone is not persistence: independent download, CRC/hash/schema/model/optimizer checks are required after execution.

## Execution result

Submission/status, actual metrics, final epoch, final patience/winner, hashes, GPU runtime and independent persistence verification are **PENDING**. No result is inferred from prepared code. This record will be completed from actual Kaggle outputs; a failure will be diagnosed without automatic resubmission.
