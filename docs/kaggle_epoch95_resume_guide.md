# ZeroMRG COV-CTR: superseded epoch-95 guide

Updated 2026-10-10. **Do not use the epoch-95 notebook as the active continuation workflow.** No persistent epoch95 model-and-optimizer checkpoint was verified. The canonical latest checkpoint is epoch90; next epoch91 is derived from saved metadata, not a hard-coded source epoch. Historical logs through120 are not recoverable weights.

The old guide is preserved at `artifacts/reconciliation/gate1_20261010_remote/legacy_epoch95_resume_guide.md`. Original source notebooks remain unchanged for comparison.

Use the new [checkpoint-driven continuation notebook](../notebooks/gate1/continuation/ZeroMRG_COV_CTR_Checkpoint_Continuation_Kaggle.ipynb) and the separate [preflight notebook](../notebooks/gate1/preflight/ZeroMRG_COV_CTR_Preflight_Only_Kaggle.ipynb), each with adjacent metadata. They implement the actual checkpoint-producing decoder, preserve float32 SGD and all momentum buffers, and validate the existing split/vocabulary/cache hashes. Training remains disabled.

**Continuation is currently BLOCKED:** history selects epoch85 with text-reconstruction BLEU4 56.44423922571333, while the retrieved best file is epoch8. The latest checkpoint preserves the historical metric and patience5/50; those must never be reset. Missing epoch85 weights prevent complete selected-best recovery. Historical RNG is absent, so even a state-compatible restart cannot be a bitwise continuation.

Read [recovery](zeromrg_checkpoint_recovery.md), [readiness and exact proposed preflight action](zeromrg_gate1_readiness.md), and [persistence](zeromrg_checkpoint_persistence.md) before any execution. No remote execution is approved. Recover and verify genuine epoch85 first for the default complete continuation; any exceptional incomplete-selection recovery experiment requires its own explicit approval.

Current scope is COV-CTR 0% paired only, with GPU execution on Kaggle. Do not start training, final held-out evaluation, paired variants or IU-Xray from this guide. The 200-epoch ceiling is an implementation assumption; scientific stopping remains the preserved validation-BLEU patience rule. Five-epoch chunks are proposed durability boundaries, not permission to train or a replacement stopping criterion.
