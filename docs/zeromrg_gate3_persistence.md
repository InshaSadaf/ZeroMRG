> **Governance authority notice, 2026-10-11:** Current project position is maintained only in [project_state.md](project_state.md); roadmap and advancement are in [implementation_master_plan.md](implementation_master_plan.md). This original body is preserved as historical evidence or a detailed technical procedure, not a competing current-state or authorization record. Later scoped approvals/results are indexed in the execution ledger. The later reviewed training session used existing private baseline kernel output; a new private dataset is not an automatic restart prerequisite. Preserve historical evidence and follow the master plan amendments.

# Gate 3 checkpoint persistence and verification

This supplements the unchanged [Gate 1 persistence procedure](zeromrg_checkpoint_persistence.md) for the separately labelled recovery branch. **The completed private baseline version1 export was independently retrieved and verified**, with retained local immutable copies. No new remote checkpoint dataset/version was published; no new restart attachment is approved. Completed-version retrieval is demonstrated, not perpetual platform retention or interrupted-session durability.

| Artifact | Meaning and update rule |
|---|---|
| Read-only `historical_evidence/` | Original90 latest, stale8 best, original90 history and Gate 1/2 reports; fixed hashes, never relabelled or overwritten |
| `recovery_latest.pt` | Full resumable latest in a unique new run directory; atomically saved after every completed training epoch and after baseline initialization |
| `recovery_best_epoch_NNN.pt` | Actual validation-selected winner within measured90+ window, immutable filename; created on baseline or strict window improvement |
| Model/SGD/scheduler/RNG | Latest contains decoder, full SGD state/parameter order, schedulerNone and Python/NumPy/TorchCPU/allCUDA RNG; best is a selection artifact, not a substitute resumable latest |
| Original/continuation/recovery ledgers | Original immutable history and85/56.44423922571333/5-of50 record; independent original stopping tracker and recovery-window measurement history |
| Dependencies | Exact vocabulary, manifests, normalized text caches/raw prompt bank and hashes; never rebuild/swap them |
| Validation artifacts | Ordered71 IDs, references/predictions, measured BLEU4, metric implementation fingerprint and SHA; no held-out outputs |
| Configuration/provenance | Architecture/training/recovery policy, source hashes, environment/Java/metric fingerprint, runtime source and lockfile |

## Atomic local writes and interrupted selection

Save an actual new winner with exclusive creation before atomically replacing latest. Latest includes the current winner file/SHA, epoch, score and prediction SHA. Validate the pointer against strict first-winner selection in measured history and the best file's provenance/model state. Verify every saved measurement's split/IDs/hash, and recompute scores from saved predictions on restoration. No “slim best” copied at startup is reused after improvement.

If interrupted between winner creation and latest update, previous latest retains its valid previous winner; an orphan new winner is ignored. If interrupted during a mutable history JSON update, latest is authoritative; regenerate ancillary metadata from it. Mid-epoch training is not checkpointed: rollback to the last complete, retrievable epoch rather than guessing partial SGD state. Tests cover this pointer/orphan behavior and missing or altered metadata/files using tiny synthetic fixtures.

Original artifacts are never written. Only a `recovery_latest.pt` created within the current unique `/kaggle/working/zeromrg_gate3_<uuid>` run is atomically replaced. Completed winner files and historical copies remain. No checkpoint deletion or in-place mutation of backup datasets/notebooks is part of this procedure.

## Export and durable publication procedure

The notebook exports a unique epoch-labelled ZIP after baseline/restoration and at each approved chunk's end (at most5 training epochs). Latest is saved locally **every epoch**, independently of export cadence. ZIP includes the latest, current winner, any newly selected winners in this session, all recovery validation measurements, immutable originals/reports, nine dependencies, configuration/history/metric lock and both runtime sources. File SHA/byte sizes and runtime hashes are inside; a ZIP SHA sidecar is outside. Check free space with a1GiB margin before export. This can consume roughly1.4GiB or more per ZIP due to full originals/latest/winners; five improvements increase size. Repeated bundles also consume disk. Stop for retrieval when space is insufficient; do not delete checkpoints automatically. Export preparation alone is not durability.

1. Before any run, approve the new private target, exact source and attachment metadata. After its completion, read `kernels status` and retrieve actual outputs with the authenticated wrapper into a **new non-overwriting** local directory:

   ```powershell
   .\scripts\kaggle.ps1 -KaggleArgs @('kernels','status','inshasadaf114/zeromrg-covctr-gate3-recovery-baseline')
   .\scripts\kaggle.ps1 -KaggleArgs @('kernels','output','inshasadaf114/zeromrg-covctr-gate3-recovery-baseline','-p','artifacts/gate3/<new-retrieval-directory>')
   ```

2. Verify ZIP SHA against sidecar, ZIP CRC, internal per-file hashes and runtime hashes. Extract without overwriting existing paths. Load latest with `weights_only=True` on CPU and strict architecture/SGD/ledger/selected-file checks; recompute saved-prediction validation scores where Java/locked metric support is available. Local validation must not execute actual GPU inference or training. Retain raw logs, status and retrieval provenance.
3. If approved, create a **new private recovery dataset**, distinct from `zeromrg-checkpoint-backup5-zip`, using the verified extracted bundle. Mark privacy explicitly in metadata; use owner-approved unique slug. Do not publish over an existing backup. Later versions may append approved new recovery states to this new dataset only after explicit version-publication authorization, retaining local immutable exports and recording exact dataset version.
4. Independently download/list the actual new version and verify its latest/winner/dependency hashes. Only then record it as durable and approved for restart. Listing success or a console message alone is insufficient. A kernel listing that returns just headers does not prove files are absent; Gate 2 successfully retrieved outputs despite such a listing.
5. For the next separately approved chunk, attach the verified private version alongside the original datasets, select its exact latest path, verify metadata/private target, and preserve the original historical dependencies. Retrieve/hash-verify again before another chunk. Stop after original early stopping or epoch200.

No automatic dataset-create/version commands run in the notebook. A future owner-approved wrapper invocation for a new private dataset must be concretely reviewed before execution; it is not authorized by this design document. Publishing remote versions and decoder training are separate approval boundaries.

`/kaggle/working` is session storage and may disappear during an interrupted interactive run. Local every-epoch writes do not guarantee recovery across session loss. Completed version outputs are useful only when actually retrievable; dataset publication is useful only when its version is independently retrievable. Five-epoch chunks bound potentially lost work, but cannot eliminate it. If durable retrieval is not demonstrated, do not begin the next chunk or claim persistence verification.

## Current verification

Original historical files/reports were copied with exclusive creation, hashes verified and accidental-write protection applied: [preservation evidence](evidence/gate3_historical_preservation.json). Gate 2 verified real original-checkpoint disposable save/reload and momentum restoration. Gate 3 actual baseline save/reload, full latest/winner export, independent remote download, ZIP SHA/CRC,26 manifest entries, runtime hashes and CPU model/optimizer/history/RNG checks now **PASS**: [execution evidence](zeromrg_gate3_baseline_execution.md), [verification](evidence/gate3_baseline_verification.json). The downloaded1,382,728,028-byte ZIP has SHA `dec3cd3391e2c107a7c9ed6209997f00554cd47cb0940ca339a74446253680b1`.

Verified files are under `artifacts/gate3/baseline_retrieval_20261011T055727985114Z/verified_export_20261011T060635057524Z`. The first local verification extraction is preserved separately after a reporting-path exception. Select the verified `recovery_latest.pt` for any future approved restart; the best file is model-selection evidence. Retrieval streamed only required diagnostics and the full ZIP through the authenticated project Kaggle SDK, avoiding duplicate loose checkpoints. Listing/download provenance and raw log are retained. The runtime report's pre-retrieval false durability flag remains unchanged; independent evidence establishes actual completed-version retrieval. New private publication/attachment review remains a separate approval step before a training restart.
