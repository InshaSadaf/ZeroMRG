# ZeroMRG execution ledger and provenance index

Canonical activity/run history and artifact/configuration index, established2026-10-11. [State](project_state.md) is the concise current view; [plan](implementation_master_plan.md) owns step status/acceptance; [decisions](decision_log.md) owns choices. Existing Gate reports/raw manifests retain detailed immutable measurements; entries link them rather than making competing copies. Reconstructed entries are explicitly retrospective, not newly executed activities.

## Recording contract

Append stable EV IDs; do not renumber/delete failures. Each substantive attempt gets date/time, step/run/attempt ID, purpose and authorization source/scope, code commit **and dirty source/patch hashes**, resolved configuration/hash, dependency lock/runtime/hardware, dataset/profile/exact split/prompt/vocabulary/encoder identities, input checkpoint/hash and RNG boundary, requested limits, actual outcome/metrics with stage/backend/scale, actual timing/resource observations or unknown, output paths/hashes, verification performed/limitations, blockers/deviations/decision links and next action. Unknown means unknown, not inherited from another run. Store report-bearing details in approved local artifacts; small evidence must exclude secrets and signed URLs.

Separate activity outcome from acceptance: SUBMITTED/RUNNING are observations, VERIFIED_COMPLETE requires artifact/acceptance checks, FAILED requires actual failure evidence, UNRESOLVED is not FAILED. Training, reconstruction validation, image inference and held-out scoring are separate event types. State is updated after every accepted or failed event.

Future entry template:

~~~text
EV-NNN | UTC timestamp | ZM-NNN | run_id/attempt/parent
Purpose / approved scope and evidence:
Source: commit + dirty identity + notebook/server version/cell hashes:
Config / lock / runtime / hardware / resource limits and observations:
Data/profile/split/prompt/vocab/encoder hashes:
Input state/checkpoint/RNG and lineage:
Attempt / actual outcome / failure or deviation:
Measurements: stage, split, metric backend/scale; unknown fields explicit:
Artifacts: path, bytes/hash, producer and provenance manifest:
Verification: criteria, checks, evidence, limitations; acceptance status:
Decision/blocker/remaining work:
Next eligible action and permission; plan/state updates completed:
~~~

### EV-001

Retrospective2026-10-10 — equivalent Phase0/Gate 0 assessment. Read-only project/CLI/checkpoint inventory; source thenfd2a370, not today'sHEAD. Assessment complete but source/recovery readiness unresolved. [Actual assessment](kaggle_gate0_assessment.md) includes configurations/environment/resources/hashes and dated limits. Later EV-005–009 supersede current-state conclusions; no repeat inventory/experiment prescribed. MapsZM-001 evidence.

### EV-002

Retrospective prepared-data record — files generated2026-09-06 and existing IU500 preparation. COV modular714 split571/71/72, full IU3331 split2332/333/666; subset 500 split350/50/100. Counts/split/manifests/vocab/prompts/configuration identified by AP-001–003. Files are present and summaries inspected; original preparation code/environment/owner approval not fully reconstructed. **These are preparation, not real experiments.** Current inspection confirms checkpoint/modular COV membership differs; no replacement. Maps ZM-003/016/017; qualification uses preserved evidence, not automatic rebuilding.

### EV-003

Retrospective2026-09-27 18:38:15–18:38:17UTC — completed CPU smoke. [Run summary](../results/cov_ctr/completed_cpu_smoke/run_summary.json), predictions/metrics/history in same directory. Python3.10/Torch2.14CPU; deterministic hash features;1-layer32-D decoder;2epochs/stage;48/8/12 subset. DIAGNOSTIC_ONLY_NOT_PAPER_COMPARABLE. Original code version/complete config provenance not established; preserve artifacts. Neither0/10/100 research outcome achieved. No repeat needed for documentation.

### EV-004

Retrospective historical COV text experiment — backups preserve latest15/30/45/60/75/90; logs extend91–120 without matching recovered state. Producing source/architecture, fixed dataset/splits/cache/vocab/prompt hashes, SGD0.001/momentum0.99/batch6 and exact historical scores in [checkpoint recovery](zeromrg_checkpoint_recovery.md) and [source reconciliation](zeromrg_source_reconciliation.md). Latest90 genuine; history selects85/56.44423922571333; actual stored best 8. Original RNG and precise encoder/metric/environment revisions incomplete. Logs are computation evidence, not restoration. AP-004–006; maps006/007. Do not recreate missing85 by relabelling/rerunning.

### EV-005

Retrospective2026-10-10 — Gate 1 local recovery/source audit and readiness preparation.51 lightweight CPU tests; exact 90 model/112SGD momentum restored; remote stale8 independently hash-matched; two new self-contained notebooks/metadata statically checked. [Gate 1](zeromrg_gate1_readiness.md), [reconciliation](zeromrg_source_reconciliation.md), [audit](evidence/gate1_checkpoint_audit.json). Overall default selected-best recovery BLOCKED; no GPU execution in that task. Evidence supports003; no reason to repeat completed tests absent changed code/drift.

### EV-006

Retrospective2026-10-10/11 — single authorized private preflight version 1; metadata/source/status/log/output in [Gate 2 report](zeromrg_gate2_preflight.md) and [audit](evidence/gate2_preflight_audit.json). Mounted checkpoint/data hashes, full restoration, finite GPU text forward and disposable save/reload passed on T4. Deliberate stale8-vs85 guard raised; remoteERROR is the expected blocked selection result, not decoder failure. No updates/image encoder/test. Runtime/body timings and package environment in report. Original dataset/checkpoint unchanged. Feasibility evidence for005, not final0% outcome.

### EV-007

Retrospective2026-10-11 — **verified baseline-only run**. Actual private target inshasadaf114/zeromrg-covctr-gate3-recovery-baseline version 1 ID138127332, both datasets attached. Submitted/server sources differ in JSON serialization but all 11 cells match. [Execution](zeromrg_gate3_baseline_execution.md) and [verification](evidence/gate3_baseline_verification.json) are authoritative for exact source/config/runtime/metric hashes, paths and measurements.

Epoch90 measured text-reconstruction validation BLEU4=54.21140850351637 on71 records, pinnedPTB/corpus×100; historical stopping unchanged. Model/112momentum equal original90; no updates/image/test. Python3.13.15/Torch2.11+cu128/NumPy2.1.3/T4; Java21.0.12. Body about68seconds, not training cost or billing. Real90 winner and full RNG/latest exported and independently downloaded;CRC/26manifest entries/source/model/state checks pass. Local CPU verified hashes/schema/tensors; score recomputed on Kaggle, not local Java. First local verification's reporting-path error preserved; second complete verification passes. AP-007. Baseline authorization exhausted; no resubmission.

### EV-008

Retrospective2026-10-11 — **one submitted controlled91–95 session, outcome UNRESOLVED**. Target inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95 version 1 ID138131184. [Execution record](zeromrg_gate3_training_091_095_execution.md), [audit](evidence/gate3_training_091_095_execution_audit.json), [active snapshot](../artifacts/gate3/active_training_091_095.json), AP-008. Input verified baseline private **kernel output**, no new dataset publication; exact latest/best/export hashes pinned. NotebookSHA b331e603e50654334ea61f8463208a6c5d28ad675fa3a920a058ec9baa0e5fbc; server cell-equivalent sourceSHA c1b92cae947fdadd281f2dab18d8215811fce3f59ac171109d399c8d0566fbb1; requested timeout7200.

Approval in source record: one private session91–95 or earlier documented stop, no second session/image/test/dataset mutation/Gitpush. Monitoring reached local7800-second bound; remote last observed RUNNING/no outputs; monitor limit is not execution failure. Actual epochs/loss/BLEU/stopping/winner/export/GPU/runtime/quota unknown. Input configuration/dual ledgers fixed perGate3protocol; no epoch 95 assumed. Next002 verifies existing version; approval exhausted for submission. source_status COMPLETE in activeJSON describes the **input baseline**, not this training job.

### EV-009

Retrospective2026-10-11 10:11:58–10:12:08UTC — bounded read-only incident inspection (three targeted requests). [Incident](zeromrg_gate3_runtime_investigation.md), [raw evidence](../artifacts/gate3/runtime_incident_20261011T101158310018Z/investigation.json). Same exact training versionRUNNING; source unchanged; zero files/log characters; conflicting server last_run_time; no measured phase/utilization/quota. Six preserved artifact hashes intact. Recommendation INSUFFICIENT EVIDENCE, not a stall diagnosis. No cancellation/retry. [Future one-epoch design](zeromrg_gate3_one_epoch_diagnostic_plan.md) is unapproved/not ready; helper not integrated. Maps002/004/005.

### EV-010

2026-10-11 — Phase1 planning/governance, **documentation-only verified**. StartingHEAD7dd83c10e3e3cf43447bd31b09485208deb1f6b5 plus preserved prior dirty/staged/untracked work. Owner authorized planning records, not implementation. Local files/Git metadata inspected; no fresh remote call, package change, data/model download or experiment. Shell tool could not create its helper process; read-only filesystem/Git inspection used the Node tool. No permission rejection or escalation bypass occurred.

Equivalent assessment accepted; later evidence reconciled; modular/checkpoint split mismatch independently checked on image IDs. Four canonical records/root operating instructions and authority notices written, original bodies preserved. [Structured review](evidence/phase1_governance_review.json) holds input hashes, graph/card/link/unchanged-source checks, inventory/limitations and updated document hashes. Maps ZM-001 COMPLETE; next 002 awaiting review/resumption. No trained model or scientific result created.

### EV-011

2026-10-11 — necessary local source-control commit, explicitly authorized by the owner's instruction “Commit all the neccesary changes.” This authorizes staging/committing the reviewed governance records and existing linked incident evidence/helpers, not implementation execution or Git push. Starting HEAD7dd83c10e3e3cf43447bd31b09485208deb1f6b5.

Scope: canonical plan/state/ledger/decisions/instructions and preserved-history notices; required Gate3 pending-execution audit, incident/diagnostic design and existing recording/inspection/diagnostic helpers/test; .kaggle credential-ignore rule. Supplied ref_docs/, .agents/ skills and skills-lock.json remain local and excluded. No checkpoints, raw/derived datasets, secrets or new remote activity included. Existing code is recorded, not modified or executed against Kaggle.

Verification: two existing CPU-only logging/deadline tests passed; four selected Python files parsed; selected secret-pattern checks clear; documentation links/card/dependency checks reused from EV-010 and rechecked for new references; staged diff whitespace/content/scope reviewed. [Commit review](evidence/governance_commit_review.json) records the selected file manifest and verification. Commit identity is obtained from Git history containing this event; the event cannot embed its own commit hash. This event supersedes earlier handoff statements that no local commit was authorized.

Research state unchanged: last independently verified90; existing91–95 outcome remains unknown; ZM-002 still awaits plan review/resumption instruction. No new implementation, download/install, training, inference, test scoring, cancellation, upload or push. Next action remains the state/card's bounded existing-version verification after its prerequisites and authorization.

## Artifact and configuration provenance index

This index owns artifact **discovery/producer links**. Existing sha256/configuration/manifests own exact bytes, dependencies and measured fields. Register new artifacts here with producerEV/run/step, role/path and manifest link; do not create a competing hash spreadsheet. Logical IDs AP-NNN stable; existing artifacts retain original filenames. Verify actual hashes before using them in a new run.

|ID|Artifact family / role|Producer and canonical provenance|
|---|---|---|
|AP-001|Local modularCOV preparation|EV-002; [splits](../data/processed/cov_ctr/splits.json), vocabulary/prompt/paired IDs/report statistics/audit in same directory; [config](../configs/cov_ctr.yaml). Derived-seed memberships differ from checkpoint family. Not recovery input.|
|AP-002|Full strictIU preparation|EV-002; [validation](../data/processed/iu_xray/validation_summary.json), [split](../data/processed/iu_xray/splits.json) and peer manifests; [config](../configs/iu_xray.yaml).3331UID/2332-333-666. Approval/real-model qualification pending.|
|AP-003|IU500 isolated package/preparation|EV-002; [summary](../data/processed/iu_xray/kaggle_500/preparation_summary.json), subset_500_ids.json, artifacts/IU-Xray-500/ and ZIP; [config](../configs/iu_xray_kaggle_500.yaml).500UID/977images,350-50-100,250prompts. No training/upload approval inferred.|
|AP-004|Original full latest90|EV-004/005/006; [historical preservation manifest](evidence/gate3_historical_preservation.json) records original/mirrored path, bytes/SHA; decoder+fullSGD/history, legacyRNG absent.|
|AP-005|Stale actual best 8|EV-004/005; same preservation manifest plus [remote audit](evidence/gate1_remote_best_audit.json). Model-only; not85 or full resumable state.|
|AP-006|Immutable original history/selection and checkpoint-family dependencies|EV-004/005; [archive audit](evidence/gate1_checkpoint_audit.json), [preservation](evidence/gate3_historical_preservation.json); exact CSV/vocab/text/rawprompt hashes; best 85 evidence separate from physicalweights.|
|AP-007|Verified recovery 90 baseline/latest/winner/export/predictions/config/locks/runtime|EV-007; [verification](evidence/gate3_baseline_verification.json); [verified configuration](../artifacts/gate3/baseline_retrieval_20261011T055727985114Z/verified_export_20261011T060635057524Z/configuration.json) and sha256_manifest.json. Exact producing private kernel/version/source/runtime/Java fingerprint recorded.|
|AP-008|Submitted91–95 source/metadata/status snapshots|EV-008/009; [execution audit](evidence/gate3_training_091_095_execution_audit.json), active snapshot and incident. InputAP-007; actual post90 artifacts not registered until independently verified.|
|AP-009|Metric/inference candidate locks and source audits|EV-007 verifiedPTB/BLEU; [dependency audit](evidence/gate3_dependency_audit.json), [metric fingerprint](evidence/gate3_metric_implementation.json), [resolution](evidence/gate3_inference_resolution.json), [candidate lock](../requirements/gate3-inference-candidate.lock). Resolution/source inspection is not encoder/full-metric runtime certification.|
|AP-010|Phase1 governance/source evidence|EV-010; [review](evidence/phase1_governance_review.json); canonical records. Documentation, not experiment output.|

Future checkpoint/result registration must connect **artifact→producer run/EV→code snapshot/config→data/split/prompt/vocab/encoder→environment/metric→acceptance evidence**. Missing links block use/claim. Large ignored artifact directories need independently verified storage/retrieval; a Git record alone does not back up weights.
