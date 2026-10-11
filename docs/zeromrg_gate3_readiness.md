# Gate 3 readiness after baseline verification

Updated 2026-10-11. **Gate 3 baseline PASS. One private recovery session limited to epochs91–95 is now explicitly authorized and submitted as version1; its results/retrieval remain pending.** [Baseline evidence](zeromrg_gate3_baseline_execution.md), [CPU verification](evidence/gate3_baseline_verification.json) and [controlled-session record](zeromrg_gate3_training_091_095_execution.md) distinguish the two executions. The original Gate 1/2 selection blockers remain historical facts.

| Requirement | Status |
|---|---|
| Existing private version1 source/privacy/attachments | VERIFIED; actual target `inshasadaf114/zeromrg-covctr-gate3-recovery-baseline` |
| Epoch90 decoder, complete SGD momentum and historical state | VERIFIED; saved tensors/buffers equal original90 |
| Fresh 71-report validation baseline | VERIFIED: **54.21140850351637** with pinned PTB/corpus BLEU4 ×100 |
| Genuine available-window baseline checkpoint | VERIFIED: epoch90, not original85; best hash `6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2` |
| Full latest / RNG / save-reload / export integrity | VERIFIED on Kaggle and CPU schema/tensor checks; full export independently downloaded |
| Historical85 / 56.44423922571333 / 5-of50 | VERIFIED unchanged |
| No training, updates, image inference or test scoring | VERIFIED source branch/logs and unchanged weights/momentum; continuation history empty |
| Metric/Java runtime | VERIFIED on this Kaggle image; original versions still unknown |
| Original epoch85 weights / historical encoder revisions | UNVERIFIED / unavailable in examined evidence |
| Reviewed kernel-output attachment | VERIFIED: existing private baseline version1, exact ZIP/latest/best hashes; no new dataset publication |
| Decoder training | One epochs91–95 session authorized/submitted; completion metrics and export pending |
| Final image-to-report evaluation | NOT AUTHORIZED / NOT EXECUTED |

Use only the verified export's `recovery_latest.pt`, epoch90/next91, for a future approved recovery restart. Keep SGD 0.001/0.99, exact architecture/splits, baseline score, original stopping counter5/50 and original best independently. Strict validation improvement selects a real recovery90+ winner; only continuation scores update original stopping. At most five new epochs per approved session, inherited cap200, stop135 if no original-best improvement. Do not silently relabel90 as85, reseed a saved recovery checkpoint, reset patience or change metric/environment.

The historical RNG was absent; the baseline records a new seed42 convention and complete future RNG state. Restore and validate the exact recorded metric/environment before updates. GPU epoch costs are still unmeasured. Actual baseline log body was approximately68 seconds; this is not a training budget estimate.

The controlled session uses a separate private `inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95` target and the reviewed existing baseline output attachment. Its source/metadata and server privacy/attachments match. Original baseline metadata/source and checkpoints remain unchanged. This task stops after the single controlled session's retrieval, CPU verification and documentation; it must not launch another session or publish a dataset. Another continuation requires separate explicit authorization.
