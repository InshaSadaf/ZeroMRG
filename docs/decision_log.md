# ZeroMRG decision and authorisation log

Canonical significant choices/proposals/approval index. Established2026-10-11 from local evidence. [Plan](implementation_master_plan.md) defines work; [state](project_state.md) shows current blockers; [ledger](execution_ledger.md) records actions/artifacts. Retrospective authority is limited to the scope actually recorded; no invented blanket approval.

Statuses: OBSERVED_EXISTING (actual implementation/evidence), RATIFIED_SCOPED (documented owner action covers the named operation only), PROPOSED (recommendation), OPEN (unresolved), SUPERSEDED (retained with successor). Evidence classifications: PAPER-STATED, OBSERVED, INFERRED, IMPLEMENTATION-NECESSARY, MISSING/AMBIGUOUS. Neither an observed configuration nor a paper fact authorizes execution.

Each new material decision records stableD-ID,date,question,evidence/classification,alternatives,rationale,status,owner/authority/scope,affectedsteps/artifacts and re-opening condition. Exact approval text/reference belongs here; link detailed configuration instead of rewriting it. Log evidence-backed changes, including significant interpretation changes and unsuccessful/partial closure.

### D-001 — Objective, scope and documentation authority

2026-10-11; RATIFIED_SCOPED for **planning/governance only**. Current prompt asks end-to-end0/10/100 roadmap; own research evidence scopes COV-CTR/IU-Xray separately, faithful paper method without novelty. COV_CT references in research instructions are stale naming, not permission to add COV-CT. Alternative importing ref_docs architecture/hardware rejected: supplied documents are governance examples only. Four canonical working records plus root instructions minimize duplication; existing method/Gate bodies retained. Affected all cards/AP-010. Owner authority: current Phase1 prompt; no implementation/environment/download/remote/training/inference approval. Reopen only explicit scope change/new evidence.

### D-002 — Preserve checkpoint-producing source

OBSERVED_EXISTING and preservation required by current prompt. [Source reconciliation](zeromrg_source_reconciliation.md) proves historical unscaled/no-final-norm/160-position decoder and tokenizer differ from modular defaults. Keep producing architecture for90 restoration; alternative modular substitution fails methodological/state compatibility. This is source compatibility, not a new architecture choice. Affects003–011/AP-004–008. No reapproval requested to preserve valid work; any future migration/change needs evidence and scoped owner authorization.

### D-003 — Exact COV artifact family

OBSERVED, independently checked in EV-010. CheckpointCSV vs modularJSONL samecounts but differentIDs: overlap456train/3validation/10test. Counts/seed labels cannot identify split. Keep exact checkpoint manifests, vocab, caches and raw prompts for recovery and corresponding comparisons; preserve modular artifacts separately. Alternatives substituting localJSONL or regenerating split rejected as research drift/leakage risk. Affects003,006,011–015; review records input hashes. Future paired57 membership must be drawn from approved checkpointtrain571, not copied blindly from modular paired_10_ids.json. Do not change held-out membership to improve comparability/scores.

### D-004 — Historical stopping and available-window recovery selection

OBSERVED_EXISTING; RATIFIED_SCOPED only by the executed baseline and one91–95 authorization documented in [training record](zeromrg_gate3_training_091_095_execution.md). [Recovery policy](zeromrg_gate3_recovery_decision.md) recommends A_historical_stop_available_best_v1 rather than resetting stopping or replaying earlier checkpoint. Original85/56.44423922571333/counter 5-of50 remains immutable; original tracker alone controls stopping; recovery 90+ strict winner selects real available weights. Missing85 is an explicit deviation; prospective reproducible RNG does not replay legacy trajectory. Alternative reset extends training unjustifiably; expensive rerun cannot reconstruct missing trajectory. Future chunks/final recovery claim need their own approval and acceptance of limitation. Affects004/006/007/011; reopen if genuine traceable85/newer state is recovered or metric drift appears.

### D-005 — Existing job disposition and diagnostic proposal

OPEN. Last observed exact version 1 RUNNING/no outputs; no evidence locating phase or proving stall/failure. Alternatives: bounded current verification/live logs; wait with explicit review time; cancel only with explicit approval acknowledging possible loss; justified future diagnostic after state disposition. Recommendation: verify existing job first; do not resubmit. [One-epoch design](zeromrg_gate3_one_epoch_diagnostic_plan.md) is PROPOSED/not ready; logging helper unintegrated and hard supervisor/PTB wrapper missing. Its1500/1800 budget and external-process limits are proposed safety controls, not measured feasibility or authorized defaults. Actual input could be later than90. Affects002/004/005/006. Approval for cancellation/new diagnostic must name version/input/source/bounds; no answer invented.

### D-006 — Restart attachment route

OBSERVED_EXISTING amendment to earlierGate3 publication prerequisite: approved91–95 mounted existing private **baseline kernel output** with exact input hashes; no new dataset needed. Alternatives: independently verified kernel output with reviewed identity/manifest OR new private checkpoint dataset with explicit publication approval/retrieval. Prefer existing retrievable output when suitable to avoid unnecessary upload. Slug alone does not pin bytes/version. Affects004/005/006 and persistence procedure; old text retained as historical. Each future attachment/source/privacy review and training scope remains required; no automatic create/version command.

### D-007 — Encoder/cache fidelity and future inference environment

OPEN with PROPOSED prospective pins. Named M-CLIP/companionViT-L/14 and prompt equations paper and source supported; historical cache/model revisions missing, rawprompt/report vectors approximately different. Candidate locks source/resolution audited, not Kaggle import/CUDA certification. Investigate exact existing provenance; if necessary bounded training-only live/cache diagnostic after approval, predefined evidence-based tolerance. Correct dimensions alone cannot validate pairing; arbitrary tower/cache rebuild rejected. Runtime-compatible prospective pins may support a transparently qualified reproduction, not retroactive historical identity. Affects008/010/011/017. Install/download/diagnostic authority separate; owner accepts a material unresolved fidelity limitation before final claim.

### D-008 — Metric contract and comparison scale

OBSERVED_EXISTING for locked PTB/corpusBLEU4×100 on71 validation reports; OPEN for full suite runtime/variant verification. Paper Tables1/3 require COV BLEU1–4/CIDEr/ROUGE-L and IU BLEU1–4/METEOR/ROUGE-L. Preserve native outputs plus declared×100 convention, noting paper scaling inferred. Inspect CIDEr variant/IDF aggregation and METEORresources; known-input fixtures and saved-prediction rescoring verify semantics. Diagnostic Python fallback is not valid final metric. No invented numeric paper threshold/tolerance; establish recomputation tolerance from actual fixture/runtime evidence before accepting it. Affects009/all evaluations/023; package/Java changes need scope, technical choices resolved by evidence when possible.

### D-009 — Supervised parameter scope and lineage

OPEN. Paper§3.4 says trainencoder/decoder/imageCE, but exact unfrozen layers, prompt updates, optimizer initialization and branching are unspecified. Old reproductionA14 proposes10/100 independently initialized from same selected text model. Recommendation: independent branches, fixed train-only prompts and exact visual-parameter scope ratified before implementation; compare with sequential10→100 or restricted fine-tuning as alternatives, each carrying different interpretation. Cannot claim encoder training from frozen cached visual features. COV fixed 57/full 571 membership, validation image-conditioned selection and cap/seed/reset need explicit contract; IU separate ratification. Affects012/013/014/015/020–022. Owner chooses substantive method gap-fills; authorizing code or training is a separate action.

### D-010 — Resource and implementation adaptation

OPEN until measurements. Existing COV decoder training/provenance is preserved; paired encoder training and full IU resource cost unmeasured. Alternative engineering routes include shorter approved chunks, supported caching of frozen components, checkpointing/AMP/accumulation with relevant numerical/equivalence evidence, another explicitly approved host, or labelled incomplete outcome. Do not silently reduce model/vocabulary/patience/data or inherit reference-machine limits. Define safety bounds from owner budget and measured host/workload; record changes before new experiment. Affects005/013/016–022; permissions for environment/host/resources required, no hardware recommendation asserted current.

### D-011 — IU full profile, missing fields and views

OBSERVED_EXISTING preparation; OPEN ratification/execution. Full strict3331UID/2332-333-666 and1000prompts reflects both-section exclusion; paper3955 vs supplied3851 discrepancy. IU500 package is resource contingency, not full reproduction. Original plan proposes impression+space+findings and per-view enrichment→masked mean→normalize; paper leaves delimiter/view policy ambiguous. Ratify method and decoder/tokenizer/length from actual feasibility; inspect source access terms and prospective private mount/approval before new movement. Prefer preserved full target when measurable feasibility supports it; alternative labelled500 requires owner acceptance and full-targetremaining status. Affects016–022. Reopen on population/hash/view/permission contradiction; no automatic full artifact overwrite.

### D-012 — Completion and optional work

RATIFIED_SCOPED reporting principle from current prompt; final acceptance pending. Required six dataset×ratio outcomes must have verified model/prediction/metric/provenance packages to claim full completion. Poor legitimate performance is an unsuccessful numeric reproduction, not necessarily failed implementation. Missing/unrun rows remain unachieved even if an owner accepts partial/unsuccessful closure. Paper Tables2/4 and rebuilding all competitor baselines are outside scope; paper numbers may be cited for comparison, not local results. Optional additional seeds/clinical metrics/optimization/newdatasets need separate purpose and authority, not automatic workload. Affects023/024; owner approves final closure claim.

## Approval boundaries at the current handoff

- **Granted now:** local read-only reconstruction and planning/operating documentation changes. No implementation performed.
- **Historically granted/exhausted:** one preflight; one baseline-only run; one private91–95 session. They are not future launch permissions.
- **Next instruction:** owner reviews plan then resumes002 existing-version read/status/output and CPU verification. This prompt explicitly stops before that action.
- **Separately required when applicable:** cancellation, new diagnostic/training/inference/test execution, significant method gap-fill, package/environment changes/model/data downloads, remote data/derived-artifact upload/publication, destructive replacement/deletion and Git commit/push/sharing. Permission names exact scope; no repeated approval for actions already covered in that later task.

Future approval record: date, approving owner instruction reference/exact scope, target/version/source/config/input identities, permitted actions and resource/epoch bounds, exclusions, consumed action/run EV and next permission boundary. Do not turn Codex's recommendation into the owner's decision.

### D-013 — Necessary local commit

2026-10-11; RATIFIED_SCOPED by the owner's explicit follow-up “Commit all the neccesary changes.” Stage and commit the reviewed governance records, their required historical/incident evidence and existing related helpers/test, plus credential-directory ignore protection. Alternatives blanket git-add or including unrelated supplied references/installed skills rejected to keep commit scope reviewable. EV-011 and its review record name the files/checks. No Git push or research implementation authority is granted. Future ZM-002/other execution still follows prior entry conditions; commit authorization does not constitute review/approval of research proposals.
