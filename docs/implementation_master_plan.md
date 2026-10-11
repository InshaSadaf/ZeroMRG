# ZeroMRG master implementation plan

Revision 1, 2026-10-11. Planning is authorized. **Execution awaits owner review and an action-specific instruction.** Stable roadmap namespace: ZM-001–ZM-024. Historical Gate numbers and research-plan steps retain their identities as evidence references.

## Objective and outcomes

Implement and assess *Auto-encoding clinical language for zero-shot medical report generation* on the supplied **COV-CTR and IU-Xray English-report datasets**, separately, through **0%, 10%, and 100% paired-training conditions**. Preserve verified COV-CTR work; complete its recovery branch; produce traceable models, held-out image-to-report predictions, verified dataset-specific metrics, paper comparisons, and an honest account of deviations and unsuccessful outcomes. No new architecture, losses, datasets or novelty enter scope. Chinese COV-CT/Table 2/Table 4 are outside scope.

Method sources: [paper analysis](../research/paper_analysis.md) §§3,10–16; supplied [paper](../paper/2026%20medical%20report%20generation.pdf) §§3.2–3.4,4.1–4.3, Tables1/3; [dataset analysis](../research/dataset_analysis.md). The retained [paper extract](../artifacts/reconciliation/gate1_20261010_remote/paper_text.txt) was inspected for critical split/training/metric claims. Original COV_CT naming in research instructions is stale; COV-CTR is the actual supplied and implemented dataset.

| Target | Meaning | Required endpoint |
|---|---|---|
| 0% | Train decoder by text-only reconstruction with frozen M-CLIP features; **zero paired image–report supervision**. At inference freeze all modules and enrich companion CLIP visual features with training-report prompts. | Defensible stopping and physical validation winner; every held-out image/study generated; verified metrics and provenance. Text reconstruction alone does not complete0%. |
| 10% | Image-conditioned cross-entropy on fixed approximately10% of training pairs, on the basis of auto-encoding. Paper states training visual encoder and decoder; exact parameter scope is unresolved. | Approved supervised contract; subset and gradient checks; durable latest/winner; complete held-out prediction/metric package. |
| 100% | Same supervised objective on all training pairs, excluding validation/test. | Full train coverage, separately identified approved lineage, durable selected model and complete test package. |

Fractions are paired supervision, not epochs, accuracy, dataset completion or the fraction of text available. Train reports may feed text training/prompts and must be disclosed. Existing rounding is an assumption:57/571 COV;233/2332 full IU;35/350 IU500. No required metric threshold, runtime estimate or allowable paper discrepancy is invented.

| Dataset/profile | Existing preparation | Prompts | Each0/10/100 evaluation |
|---|---|---:|---|
| COV checkpoint family |714 pairs, savedCSV split571/71/72; vocab 368, width 768, length 160; six conflicting keys/26 unannotated images excluded |500 |Exact test 72; BLEU1–4,CIDEr,ROUGE-L; Table 1 |
| IU full strict |3331 studies with impression+findings; UID split2332/333/666; preserve all mapped views |1000 |Exact test 666 studies; BLEU1–4,METEOR,ROUGE-L; Table 3 |
| IU500 contingency |500UIDs/977images;350/50/100; separate vocab 1013 and artifacts |250 |Subset test 100; same metric family; RESOURCE-CONSTRAINED, cannot satisfy full IU claim |

IU separator/view pooling/missing-section handling/tokenizer/decoder/length/stopping and encoder pins require reconciliation before execution. No implicit COV-weight transfer intoIU. IU archive3851 reports differs from paper3955; COV714 differs from paper728. UID/image-ID disjointness does not establish patient independence; COV source/patient grouping remains uncertain.

Implementation completion means required code/restoration/evaluation paths verified. Experiment completion means the declared terminal condition plus independently preserved artifacts and evaluation. Research evidence means valid interpreted comparisons within stated limits. Command exit, high BLEU, epoch 95/200, or0<10<100 ordering is insufficient. A legitimate low score can complete an experiment and be an unsuccessful result reproduction. Required outcomes cannot be marked achieved by labelling unrun rows complete.

## Authority and canonical records

No separately named Phase0 assessment exists in the bounded workspace inventory. [kaggle_gate0_assessment.md](kaggle_gate0_assessment.md) is a reliable **equivalent starting assessment** completed2026-10-10. Later Gate evidence corrects its provisional claims. It lacks a full reference-governance synthesis; this revision supplies it, not a second Phase0 investigation.

| Responsibility | Authoritative home | Relationship |
|---|---|---|
|Objective/scope/cards/status/dependencies/architecture/gates |This file |Update evidenced status or controlled amendments; preserve IDs |
|Actual stage/latest observation/active work/blockers/next action |[project_state.md](project_state.md) |Concise projection; links to plan/ledger, no competing history |
|Activities/experiments/failures/run and artifact/config provenance index |[execution_ledger.md](execution_ledger.md) |Append events; detailed machine manifests remain the source of measured fields |
|Choices/proposals/assumptions/scoped approvals |[decision_log.md](decision_log.md) |Explicit status/evidence/authority; recommendations never approvals |
|Persistent operating cycle |[root AGENTS.md](../AGENTS.md) |Links to records, complements research instructions |
|Method/data analysis |research/paper_analysis.md; research/dataset_analysis.md |Evidence sources; old scope statements resolved here |
|Detailed recovery procedures and executed evidence |docs/zeromrg_*.md; docs/evidence/*; artifacts/* |Dated records preserved; Gate 3 protocol applies with amendments below; ledger distinguishes approvals/results |

Four working records plus root instructions suffice: architecture is here; artifact provenance uses the ledger and existing manifests. No new tracker/tooling migration. Historical reproduction_plan.md/progress/setup guides stop maintaining current status; original bodies remain. Active JSONs are individual run snapshots, not project state. Raw evidence is not rewritten to match later interpretations.

Authority order: explicit owner instruction/approval; applicable project instructions; paper for method claims; inspected code/artifacts/execution for actual behavior; this roadmap for ordering; historical plans for context. Conflicts stop the affected action and enter decisions. **ref_docs/** has no execution authority despite the matching project name. Transfer only stable cards, evidence classes, fixture checks, separated selection/test, provenance, measured feasibility, retrieval and controlled amendments. Do not import its MachineA/B hardware, thresholds, beam search, package stack, authorization codes, dataset rulings or steps.

### Critical reconciliation

1. Original latest90 verified; history selects85 but saved best is physically8. Logs through120 cannot restore missing weights/SGD. [Recovery audit](zeromrg_checkpoint_recovery.md).
2. Gate 1/2 default continuation remains historically BLOCKED. Separately labelled policy A_historical_stop_available_best_v1 produced a physical90 baseline/full latest and one approved91–95 submission. It does not restore85 or turn Gate 2 into PASS. [Baseline proof](evidence/gate3_baseline_verification.json), [session](zeromrg_gate3_training_091_095_execution.md).
3. Last saved observation2026-10-11 10:11:58–10:12:08UTC: private training version 1 RUNNING/no published logs/files. **This turn made no remote query. Present status is unknown.** Server last_run_time conflicts with submission chronology; requested7200-second timeout is not measured runtime/enforcement. [Incident](zeromrg_gate3_runtime_investigation.md).
4. **COV modular and checkpoint splits are different.** Same571/71/72 counts, but image-ID overlaps train 456,validation 3,test 10. Local JSONL uses derived split seed942271317. Recovery and related0/10/100 comparisons must retain checkpointCSV manifests/vocabulary/caches/raw prompts, not data/processed/cov_ctr JSON defaults. [Review evidence](evidence/phase1_governance_review.json). Preserve both families.
5. Modular model defaults scale tokens, have final norm/512positions and normalize prompts. Producing runtime has unscaled tokens/no final norm/160positions/raw prompts and different token IDs. Strict restore mandatory. Broad experiment driver/legacy notebooks can train fresh, score test and proceed to paired stages; they are not approved recovery launchers. [Source reconciliation](zeromrg_source_reconciliation.md).
6. Older Gate 3 protocol/persistence required a new private dataset. Later approved91–95 used existing private baseline **kernel output**, pinned by hashes. Future reviewed kernel output OR separately approved new private dataset may supply restart; publication is not automatically necessary. Record server version/cells and actual mounted hashes, not slug alone.
7. Candidate inference lock is source/resolution audited, not runtime certified. Baseline verified PTB/corpusBLEU4 only; full metric families/image encoders still pending. Historical encoder/RNG/metric revisions remain unknown; prospective pins cannot repair history.

## Architecture and operating boundary

Windows repository hosts later authorized code/config/testing, documentation, project-venv CLI and non-overwriting retrieval/CPU verification. Observed HEAD7dd83c10e3e3cf43447bd31b09485208deb1f6b5; prior staged/unstaged/untracked work exists. No implicit stage/discard/commit/push. A dirty executable needs commit plus exact hashes/patch identity.

Kaggle hosts approved GPU work. /kaggle/input is immutable; each run writes a unique /kaggle/working directory. Recovery notebooks embed scripts/zeromrg_gate1_runtime.py and scripts/zeromrg_gate3_runtime.py with audited metric cells, not src defaults or a required clone. Builders/static validators bind source; preserve submitted snapshots and make a new reviewed version for changes.

Flow: raw data→validated exclusions→frozen splits/vocabulary→frozen text features/train prompts→decoder reconstruction→validation winner→frozen visual encoder/prompt attention→reports→verified metrics. Paired branches require train images/targets and differentiable approved encoder/decoder; frozen image caches cannot train the encoder. IU adds UID view grouping and ratified pooling.

COV Eq6: softmax(v P^T/sqrt(768))P over raw500 prompts. Eq5: normalize(v+attended prompts). Use the named OpenAICLIP ViT-L/14 companion, not an arbitrary768-D tower. COV decode BOS1/EOS2/PAD0/UNK3, greedy max160 includingBOS and historical punctuation handling. These producing settings include paper gap-fills. Candidate image/preprocessing pins, cache correspondence and freeze flags must be verified.

Keep existing ignored data/, artifacts/, results/ conventions; small nonsecret evidence in docs/evidence/. Do not migrate to reference storage paths. Latest is full resumable state; best is selection evidence. Working-directory export is not durable until independently retrieved/verified. No destructive replacement, raw-data edits, secret logging, signedURL recording or unapproved uploads. Review report-bearing records before sharing/committing.

## Roadmap and common rules

Status: COMPLETE=evidence and record checks pass; PARTIAL=useful work exists; NOT_STARTED=no real experiment evidence; BLOCKED=named unmet condition; PROPOSED=unapproved branch. Authorization is separate. IDs never reused: insert suffixes, retain SUPERSEDED cards with successor links, log split/move/amendment. Each retry/chunk has a new runID under the same step.

|Phase|Steps|Exit|
|---|---|---|
|A Preserve/resolve recovery|001–005|Existing job disposition; source/data contract; bounded recoverable readiness|
|B COV0|006–011|Stopped selected decoder; frozen image path/full metrics; test 72 package|
|C COV paired|012–015|Ratified supervised protocol; separate 10%/100% packages|
|D IU|016–022|Profile/data/runtime contract;0/10/100 study packages|
|E Closure|023–024|Traceable comparison, deviations and reproducibility handoff|

Dependencies in cards are acyclic.003/008/009 allow independent preparation; IU016 after COV0 need not wait for COV paired results. Waiting for10% review before100% is a **procedural/resource gate**, not model initialization from10%. No simultaneous expensive jobs assumed. Training-image smoke010 precedes **new** multi-epoch 006 to expose image failures; historical91–95 submission is preserved.010 may use actual90 for mechanics; final011 requires stopped selection007.

All cards define A–M. **K on every card also requires:** append ledger evidence→verify acceptance→update card status/state/next action→update material decisions→report before the next substantive task. No successful command alone closes a card. Do not rerun COMPLETE checks unless new contradiction/drift justifies it.

| Step | Status | Dependencies |
|---|---|---|
| [ZM-001](#zm-001) | COMPLETE | Starting evidence |
| [ZM-002](#zm-002) | BLOCKED | ZM-001 |
| [ZM-003](#zm-003) | PARTIAL | ZM-001 |
| [ZM-004](#zm-004) | BLOCKED | ZM-002 |
| [ZM-005](#zm-005) | PARTIAL | ZM-003, ZM-004 |
| [ZM-006](#zm-006) | PARTIAL | ZM-005, ZM-010 |
| [ZM-007](#zm-007) | BLOCKED | ZM-006 |
| [ZM-008](#zm-008) | PARTIAL | ZM-003, ZM-004 |
| [ZM-009](#zm-009) | PARTIAL | ZM-003 |
| [ZM-010](#zm-010) | NOT_STARTED | ZM-008 |
| [ZM-011](#zm-011) | NOT_STARTED | ZM-007, ZM-009, ZM-010 |
| [ZM-012](#zm-012) | BLOCKED | ZM-011 |
| [ZM-013](#zm-013) | PARTIAL | ZM-012, ZM-003 |
| [ZM-014](#zm-014) | NOT_STARTED | ZM-013, ZM-011 |
| [ZM-015](#zm-015) | NOT_STARTED | ZM-014, ZM-013 |
| [ZM-016](#zm-016) | PARTIAL | ZM-011 |
| [ZM-017](#zm-017) | PARTIAL | ZM-016, ZM-003, ZM-008, ZM-009 |
| [ZM-018](#zm-018) | NOT_STARTED | ZM-017 |
| [ZM-019](#zm-019) | NOT_STARTED | ZM-018, ZM-017, ZM-009 |
| [ZM-020](#zm-020) | PARTIAL | ZM-019, ZM-012, ZM-013 |
| [ZM-021](#zm-021) | NOT_STARTED | ZM-020, ZM-019 |
| [ZM-022](#zm-022) | NOT_STARTED | ZM-021, ZM-020 |
| [ZM-023](#zm-023) | NOT_STARTED | ZM-011, ZM-014, ZM-015, ZM-019, ZM-021, ZM-022 |
| [ZM-024](#zm-024) | NOT_STARTED | ZM-023 |

### ZM-001

**Reconstruct state and establish governance** — COMPLETE

| Field | Step contract |
|---|---|
| A. Purpose | Preserve accomplishments and create the controlled operating system. |
| B. Current evidence | Equivalent Gate 0, later Gates, paper extract, source/configuration, records and reference practices inspected; gaps explicit above. |
| C. Prerequisites | Reliable starting assessment and workspace read access. |
| D. Dependencies | Gate 0/later local evidence; owner planning prompt. |
| E. Planned actions | Inventory actual paths; resolve critical conflicts; write canonical records; review fields/links/graph and documentation-only changes. |
| F. Expected outputs | Roadmap, state, ledger/provenance, decisions, root instructions, phase1 review evidence. |
| G. Verification | 24 A–M cards; acyclic dependencies; links exist; source/config/notebook hashes unchanged; prior Git work retained. |
| H. Completion criteria | Consistency checks pass with unresolved research/platform facts bounded; new files reread successfully. |
| I. Failure handling | Missing assessment stops planning; unsupported claims stay unresolved. |
| J. Recovery | Read canonical records and review evidence; original document bodies/input hashes retained. |
| K. Documentation | EV-010; docs/evidence/phase1_governance_review.json; state and authority mapping. |
| L. Authorisation | Documentation/local read-only inspection authorized; no execution/code/environment changes. |
| M. Next dependency | Owner review then002;003 source review can proceed in a later scoped task. |

### ZM-002

**Resolve existing private91–95 execution** — BLOCKED

| Field | Step contract |
|---|---|
| A. Purpose | Find actual outcome and highest retrievable completed state without duplicate compute. |
| B. Current evidence | Target inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95 version 1 ID138131184; last saved incident RUNNING/no outputs; now unknown. |
| C. Prerequisites | Plan review and instruction to resume existing-run verification; network/write permissions. |
| D. Dependencies | ZM-001; EV-008/009; active_training_091_095.json and submitted/server snapshots. |
| E. Planned actions | Make bounded fresh exact-version status/source/output check; capture evidence. Terminal: retrieve to new directory and CPU-verify. Unresolved: obtain live phase evidence if available; no endless polling. ERROR: first traceback and surviving-state audit. |
| F. Expected outputs | Actual disposition/logs; independently verified completed export if available; resource observations. |
| G. Verification | Review monitor/retrieve/verify_gate3_training_091_095 helpers before use. Source/privacy/attachments, SHA/CRC/membership, actual epochs≤95,112momentum/RNG/dual ledgers/real winners. Incomplete output needs separate audit, not forced success verifier. |
| H. Completion criteria | Resolved job with verified artifacts, or owner-approved terminal disposition plus explicit unavailable-work record. RUNNING alone leaves BLOCKED. |
| I. Failure handling | Unknown phase/clock conflict stops new GPU work. Cancellation and retry require separate approval. |
| J. Recovery | Use highest downloaded verified latest;90 fallback only if nothing later survives. No assumption of zero updates. |
| K. Documentation | Append status/retrieval/diagnosis EV; update run/provenance index/state and disposition decision. |
| L. Authorisation | No action now. Historical submission authorization exhausted; verification permission is not cancellation/new-training permission. |
| M. Next dependency | 004 after disposition;003/009 preparation can proceed independently with scope. |

### ZM-003

**Freeze executable source and artifact-family contract** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Make future runs reconstructable and checkpoint compatible. |
| B. Current evidence | Runtime/notebooks/pins exist; HEAD/dirty work observed; modular split/model materially different. |
| C. Prerequisites | 001 complete; later scoped source review; no GPU required. |
| D. Dependencies | ZM-001; source reconciliation/builders/validators; D-002/003. |
| E. Planned actions | Review existing staged/unstaged ownership; bind exact runtime/notebook/metadata/lock hashes, input family/configuration. Approved changes get new snapshot and static/CPU checks; commit only if instructed. |
| F. Expected outputs | Reviewed source/configuration and dependency manifest attached to each run. |
| G. Verification | Strict keys/token IDs/positions/unscaled embeddings/raw prompts; exact saved CSV hashes; embedded cell-source equality; dirty patch/hash identity; no broad driver launch. |
| H. Completion criteria | Run reproducible from commit plus exact changed-source identity, configuration and inputs; no unexplained executable drift. |
| I. Failure handling | Source/input mismatch stops; no shape adaptation, split regeneration or convenient model substitution. |
| J. Recovery | Immutable submitted/baseline snapshots; preserve all user work. |
| K. Documentation | Ledger source/provenance event, D-002/003 and state/card readiness. |
| L. Authorisation | Future implementation edits/generation/Git actions need scope; current inspection is read-only. |
| M. Next dependency | 005/008/009/013/017; clean Git alone never proves input validity. |

### ZM-004

**Ratify recovery and incident disposition** — BLOCKED

| Field | Step contract |
|---|---|
| A. Purpose | Choose an evidenced route and bounded operation. |
| B. Current evidence | PolicyA used for baseline/one chunk;85 missing; one-epoch diagnostic proposal is not ready. |
| C. Prerequisites | 002 resolved and surviving state known; owner input for cancellation/loss/retry/deviation. |
| D. Dependencies | ZM-002; D-004/005/006; recovery and diagnostic design. |
| E. Planned actions | Inspect first actual failure/phase; choose later-state continuation, further evidence or justified fallback diagnostic. Preserve dual ledgers; declare exact input/budget/deadlines/retention before preparation. |
| F. Expected outputs | Dated approved disposition/run contract, or explicit blocker. |
| G. Verification | All surviving states considered; no blind restart91 if later state exists; no reseed/patience reset; budgets identified as measured or proposed safety bounds. |
| H. Completion criteria | Choice ratified and allowed operations recorded; unresolved existing job cannot justify duplicate continuation. |
| I. Failure handling | Insufficient evidence stays blocked; focused decision request only. |
| J. Recovery | Independently verified full state and immutable original history; controlled amendment. |
| K. Documentation | Decision log for new approval; ledger disposition; state next action. |
| L. Authorisation | Owner approval for cancellation/new attempt/budget or method change; previous run approval is limited. |
| M. Next dependency | 005/008 preparation in approved scope;006 still requires readiness/010. |

### ZM-005

**Prepare bounded observable recovery execution** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Prevent another unobservable or unrecoverable expensive attempt. |
| B. Current evidence | Completed-epoch saves/export exist. Diagnostics helper not integrated; hard supervisor/boundedPTB wrapper not ready. |
| C. Prerequisites | 003 source contract,004 approved route. |
| D. Dependencies | ZM-003,004; persistence/metric lock; D-005; execution-time platform evidence. |
| E. Planned actions | Only if justified, implement phase/batch logging and external deadlines without changing outputs; test tokenizer/supervisor/failure export. Otherwise reuse proved code. Recheck CLI/help/quota, accelerator/RAM/disk, package/Java signatures/privacy/attachments before launch. |
| F. Expected outputs | Reviewed source/metadata, relevant CPU/static evidence, resource envelope and retrieval/failure path. |
| G. Verification | Input guards before updates; one-attempt/epoch bounds; finiteCE/save-reload; unchangedPTB fixture; observability/deadlines. Diagnostic1500/1800 proposals are not measured or approved defaults. |
| H. Completion criteria | Selected route checks pass; resource/permission issues resolved; old preflight/baseline reused. |
| I. Failure handling | Drift/OOM/space/package failure: preserve state, diagnose minimal fix and approve changed method; no blanket upgrade/deletion. |
| J. Recovery | Full retrieved latest only; runtime drift reviewed before updates. |
| K. Documentation | Ledger preparation/environment evidence; decisions if amended; plan/state. |
| L. Authorisation | Code/environment/new remote diagnostics require scope; no install/submission here. |
| M. Next dependency | 006 after 010 and specific chunk approval. |

### ZM-006

**Complete controlled COV text continuation** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Finish existing text-only experiment with preserved stopping. |
| B. Current evidence | Verified90; submitted91–95 outcome unknown. Historical best 85/56.44423922571333/counter 5-of50. |
| C. Prerequisites | 002 resolved;005 ready;010 smoke; exact full state/RNG/environment; explicit new-chunk approval. |
| D. Dependencies | ZM-005,010; policyA and checkpoint-family protocol. |
| E. Planned actions | For each approved chunk restore verified latest; decoder-only FP32, SGD0.001/momentum0.99/batch6/clip5/no scheduler. Validate71 text reconstructions each epoch; save actual winners/full latest; export/retrieve/verify before next. Existing protocol max5 new epochs/session. |
| F. Expected outputs | Loss/validation predictions/BLEU4, measured timing/resources, full optimizer/RNG/ledgers and durable selected files. |
| G. Verification | Exact caches/manifests/vocab; finite state;112momentum; strict selection/original-stop tracking; no paired/test work; actual completed epoch records. |
| H. Completion criteria | Counter reaches50 or inherited total cap200 with reason; accepted states durable. No improvement after 90 implies stop 135. Epoch 95/200 is not success. |
| I. Failure handling | Failure stops; first traceback and completed output retained; no automatic retry/extension. Resource stop is incomplete unless approved labelled termination deviation. |
| J. Recovery | Discard incomplete epoch; restore last retrievable complete latest/RNG/optimizer/pointer. Missing export leaves new progress unverified. |
| K. Documentation | Each chunk new run/events/manifest; state update after verification; stopping evidence closes card. |
| L. Authorisation | Each extra Kaggle session separately authorized; historical91–95 cannot authorize a second. |
| M. Next dependency | 007 selection audit;011 still waits for image/full metrics. |

### ZM-007

**Lock COV stopping and physical winner** — BLOCKED

| Field | Step contract |
|---|---|
| A. Purpose | Freeze defensible selection before held-out access. |
| B. Current evidence | Recovery90 physical winner exists;85 missing; final stopping not established. |
| C. Prerequisites | 006 terminal evidence or explicitly approved labelled early-termination deviation. |
| D. Dependencies | ZM-006; policyA and validation artifacts; D-004. |
| E. Planned actions | Audit history/counter/strict winner/ties and prediction correspondence; rescore saved validation where supported; freeze selected hash and recovery interpretation. |
| F. Expected outputs | Selection/termination record and immutable final-inference initialization. |
| G. Verification | Physical weights/epoch/score/pointer agree; original evidence unchanged; no test-based choice. Distinguish CPU schema agreement from Java metric recomputation. |
| H. Completion criteria | Stopping justified; winner independently retrievable; available-window limitation accepted and disclosed. |
| I. Failure handling | Mismatch/untraceable winner stops test; repair evidence or document unsuccessful outcome, never silently choose latest. |
| J. Recovery | Last verified pointer and immutable validation records; preserve disputed files. |
| K. Documentation | Ledger selection/provenance; decision if deviation; plan/state. |
| L. Authorisation | Future scoped artifact review; new model validation/early-stop choice needs relevant authority. |
| M. Next dependency | 011 after 009/010. |

### ZM-008

**Qualify frozen image environment and cache provenance** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Confirm paired visual/text space and executable frozen image path. |
| B. Current evidence | Candidate pins/resolution exist; import/torchvision/image forward unexecuted; historical cache revisions unknown. |
| C. Prerequisites | 003/004 scope resolved; explicit required downloads/install/diagnostic permission. |
| D. Dependencies | ZM-003,004; Gate 3 dependency audit/locks; D-007. |
| E. Planned actions | Reverify official compatibility at execution; audit companion weights/preprocessing and separateM-CLIP/base/tokenizer pins. Inspect existing provenance first; approve bounded training-only live/cache diagnostic if necessary; never regenerate caches to hide mismatch. |
| F. Expected outputs | Runtime/weight/preprocessing fingerprint; cache-fidelity verdict and unresolved historical limits. |
| G. Verification | Correct named companion/hash, device inputs/pooling/projection, raw prompt order and freeze flags. Dimension768 alone insufficient. Define comparison tolerance from evidence before accepting live/cache agreement. |
| H. Completion criteria | Actual runtime works; incompatible pairing resolved or owner accepts declared historical limitation without claiming exact replay. |
| I. Failure handling | Mismatch stops final inference; diagnose revision/pooling/normalization; no arbitrary tower/cache replacement. |
| J. Recovery | Original caches and isolated candidate environment retained; failed fingerprints preserved. |
| K. Documentation | Ledger cache/environment evidence, D-007 and provenance/state. |
| L. Authorisation | No download/install/GPU action now; future exact operations need scope. |
| M. Next dependency | 010 smoke;017 may reuse qualified runtime. |

### ZM-009

**Verify complete evaluation harness** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Ensure scores have declared meaning and can be recomputed. |
| B. Current evidence | PTB/corpusBLEU4 verified for reconstruction; complete metric source/diagnostic fallback exist; CIDEr/ROUGE/METEOR runtime pending. |
| C. Prerequisites | 003 contract; fixture/runtime authorization; no test predictions needed. |
| D. Dependencies | ZM-003; metric lock/source; D-008. |
| E. Planned actions | Known-input fixtures; audit corpus aggregation/PTB order/lowercase/punctuation,CIDEr variant/IDF corpus,ROUGE/METEOR resources. Pin actual code/JAR/deps; preserve native and×100 scores; forbid diagnostic fallback. |
| F. Expected outputs | Verified metric contract/fixtures and saved-prediction recomputation instructions. |
| G. Verification | Nonempty equal ID-aligned inputs; recorded fixture scores; required metric output hashes. Define numerical recomputation tolerance from fixture evidence before acceptance. |
| H. Completion criteria | Required dataset suite runs on intended host;BLEU4 matches recovery; scaling/variants disclosed. |
| I. Failure handling | Java/package/aggregation mismatch blocks scoring; no replacement backend or falsely complete partial suite. |
| J. Recovery | Rescore saved predictions without GPU generation; retain failed outputs. |
| K. Documentation | Ledger fixture evidence/D-008/provenance; plan/state. |
| L. Authorisation | Future installs/runtime checks need scope; this card does not authorize test access. |
| M. Next dependency | 011/019 and supervised evaluations once model gates pass. |

### ZM-010

**Verify frozen training-image COV path** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Expose integration failures before more multi-epoch expense/test. |
| B. Current evidence | Gate 2 stopped before CLIP; verified90 may test mechanics, not final selection. |
| C. Prerequisites | 008 qualified; traceable physical checkpoint; explicit bounded GPU diagnostic approval. |
| D. Dependencies | ZM-008; historical runtime/raw500prompt/vocab/preprocessing. |
| E. Planned actions | Use fixed train image only; freeze/eval all modules; image→Eq6/Eq5→greedy decoder; preserve input/condition/termination and unchanged weights. |
| F. Expected outputs | Training-only diagnostic with shapes/finite/termination/provenance/resource evidence. |
| G. Verification | No backward/update/test;RGB224canonical transform;768condition; correct raw attention/unit residual;EOS or max160; no invented clinical threshold. |
| H. Completion criteria | Entire frozen path verified with traceable input/output; fluent report alone does not prove clinical correctness/alignment. |
| I. Failure handling | Isolate encoder/enrichment/tokenizer/decoder; no test probe. |
| J. Recovery | Unchanged verified checkpoint and train diagnostic inputs; no training rerun. |
| K. Documentation | Ledger diagnostic/provenance; state/card;D-007 if changed. |
| L. Authorisation | New inference separately authorized; current prompt prohibits it. |
| M. Next dependency | New006 continuation;011 after 007/009. |

### ZM-011

**Produce COV0 held-out outcome** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Complete core zero-paired image-to-report experiment. |
| B. Current evidence | No valid final result;54.2114/56.4442 are text-reconstruction scores. |
| C. Prerequisites | 007selected stopped model;009metrics;010path; frozen protocol and explicit test/inference approval. |
| D. Dependencies | ZM-007,009,010; checkpoint test 72;D-007/008. |
| E. Planned actions | Freeze all source/config/model/prompt/encoder hashes before test; generate every exact ID; retain raw ordered ref/pred/termination; score suite; independently retrieve and recompute; compareTable1. |
| F. Expected outputs | results/cov_ctr/<unique-run>/ prediction/metric/manifest package and recovery limitations. |
| G. Verification | Exactly72unique checkpoint-family IDs, no train/validation; correct correspondence/backend; hashes and lineage agree; no selection change. |
| H. Completion criteria | Complete verified package and interpretation even if poor legitimate scores; no minimumBLEU imposed. |
| I. Failure handling | Integrity bug quarantined; corrected rerun needs evidence/approval. Scientific discrepancy investigated without test tuning. |
| J. Recovery | Interrupted generation resumes only identical frozen state/exact ID coverage; rescore retained predictions; preserve attempts. |
| K. Documentation | Ledger evaluation/provenance/comparison; state/status;decisions for reruns. |
| L. Authorisation | Separate GPU inference/held-out authorization; broad driver prohibited. |
| M. Next dependency | 012supervised contract and016IUprofile. |

### ZM-012

**Ratify COV supervised experiment contract** — BLOCKED

| Field | Step contract |
|---|---|
| A. Purpose | Make10/100 experiments scientifically explicit before implementation. |
| B. Current evidence | Paper§3.4 trains encoder/decoder with imageCE; exact parameter scope, prompt updates, lineage and optimizer initialization unspecified. OldA14 proposes independent branches; it is not blanket approval. |
| C. Prerequisites | 011 reviewed outcome; focused owner decisions supported by paper and source evidence. |
| D. Dependencies | ZM-011;D-009; historical plan§7.5–7.6 and driver source. |
| E. Planned actions | Propose exact unfrozen visual parameters, fixed/trainable prompts, shared text-stage initialization, optimizer/RNG reset, image-conditioned validationBLEU4/patience50 and justified cap. Pin checkpoint-family train subset 57/full 571 IDs, fixed val71/test 72 and rounding; ratify before edits. |
| F. Expected outputs | Approved COV paired configuration/parameter/lineage/selection contract. |
| G. Verification | Every gap-fill marked assumption/decision; no test-driven setting; no frozen-cache claim of encoder training; shared initial state makes comparisons interpretable. |
| H. Completion criteria | Owner-ratified contract resolves all method choices necessary for013; no invented numerical acceptance thresholds. |
| I. Failure handling | Ambiguity stays blocked; recommend independent10/100 branches from same text winner rather than10→100, explaining assumption. |
| J. Recovery | Preserve original0% model/results; revise contract with new decision version. |
| K. Documentation | D-009 canonical approval; ledger contract event; plan/state. |
| L. Authorisation | Human research approval required; contract preparation is not training/upload authorization. |
| M. Next dependency | 013 paired readiness. |

### ZM-013

**Implement and qualify supervised COV path** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Convert approved contract into differentiable resumable paired pipeline. |
| B. Current evidence | Driver contains image-encoder/decoder training path and smoke mechanics; it uses modular artifacts/automatic stages and is not qualified for recovery branch. |
| C. Prerequisites | 012 approved contract;003 source; scoped code and local checks then separate bounded GPU smoke permission. |
| D. Dependencies | ZM-012,003; decoder/image wrappers;D-009/010. |
| E. Planned actions | Adapt/reuse only required code; explicit stage guards/dataset inputs; differentiable live approved encoder,correctCE; verify nonzero finite gradients and real changes on intended parameters, frozen invariants; tiny train-only update/save/reload and image-conditioned validation fixture; measure resource cost. |
| F. Expected outputs | Stage-specific reviewed source/metadata; full encoder+decoder/optimizer/RNG state; bounded training/restoration/resource evidence. |
| G. Verification | Exact subset/image mapping; trainable names/counts; finite loss/gradients; frozen prompts/text remain unchanged per contract; restore both trainable modules and momentum; generation/metric fixtures pass. |
| H. Completion criteria | No silent auto-stage/test run; required model fits measured host/budget; resumption/winner/export checks pass. |
| I. Failure handling | OOM/invalid gradients → record dominant resource/path issue. AMP/checkpointing/accumulation/batch/model change needs evidence and approval; no automatic smaller architecture. |
| J. Recovery | Fresh labelled smoke directory; reset from approved initialization after smoke; retain full latest for real runs. |
| K. Documentation | Ledger implementation/smoke/resource manifest;D-010if needed; plan/state. |
| L. Authorisation | Code and remote diagnostic separately authorized; no new package installation implicit. |
| M. Next dependency | 01410% training;015after10%review. |

### ZM-014

**Complete COV10 experiment and evaluation** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Produce traceable few-paired outcome. |
| B. Current evidence | 57IDs exist only as preparation; diagnostic hash-feature outputs are not real10% results. |
| C. Prerequisites | 013passed;011record; explicit training chunks and later test approval. |
| D. Dependencies | ZM-013,011;D-009; approved physical text winner/fixed 57subset. |
| E. Planned actions | Initialize approved independent branch; train only57pairs, validate 71images under frozen protocol; complete stopping, save/retrieve every bounded chunk; lock physical validation winner; generate/score exacttest72 with009harness. |
| F. Expected outputs | 10%training/history/latest/winner and held-out prediction/metrics package. |
| G. Verification | 57unique IDs subset of checkpointtrain571; noneval/test; correct trainable changes; restored full state; selected score recomputable;72test coverage; manifest/hash integrity. |
| H. Completion criteria | Declared stopping plus verified selected weights/full test suite and truthful result interpretation.57pairs alone is not completion. |
| I. Failure handling | Failure:diagnose first evidence/retry only approved. Lower score than0% is reported, not adjusted away. |
| J. Recovery | Last independently retrieved completed paired checkpoint includes visual weights/optimizer/RNG; no partial-epoch resume guesses. |
| K. Documentation | Ledger chunk/evaluationevents and provenance;state/status;deviations in decisions. |
| L. Authorisation | Each run bounded and authorized; test separately; no automatic onward100%. |
| M. Next dependency | 015after result/integrity/resource review;023comparison. |

### ZM-015

**Complete COV100 experiment and evaluation** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Produce full-paired comparison on same held-out cohort. |
| B. Current evidence | No real100%run; broad source/smoke does not establish completion. |
| C. Prerequisites | 014reviewed;013ready; approved independent initialization/protocol/resource budget. |
| D. Dependencies | ZM-014,013;D-009; same text winner as10%unless explicitly amended. |
| E. Planned actions | Initialize independent branch; train all 571pairs/validate 71; follow approved stopping/chunks/retrieval; freeze winner; generate/scoretest72. |
| F. Expected outputs | 100%training/state/winner and complete held-out package. |
| G. Verification | Full571coverage, fixedval/test/prompt/vocab, independent lineage, real parameter updates, restoration/metric/hash checks as014. |
| H. Completion criteria | Stopped selected experiment, durable artifacts/all metrics and comparison ready; no required superiority over10%. |
| I. Failure handling | Resource/performance contradiction recorded; approve amendment or unsuccessful/incomplete outcome; no silent data/model reduction. |
| J. Recovery | Resume verified full 100%latest; retain0/10models and outputs. |
| K. Documentation | Ledger/provenance/decisions asneeded; state/card. |
| L. Authorisation | Separate100%training and evaluation authorizations;10%approval not transferable. |
| M. Next dependency | 023comparison;IUwork can already proceed independently. |

### ZM-016

**Reconcile IU population, profile and governance** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Preserve preparedIU data and choose valid experiment scope. |
| B. Current evidence | Full3331UID artifacts/configuration and500UID/977image package prepared. Source3851reports; strict fields/multi-view policy are assumptions; no realIUtraining. |
| C. Prerequisites | 011COV0review; owner profile/method and remote-data approval where necessary. |
| D. Dependencies | ZM-011;data/processed/iu_xray/*;D-011;dataset analysis. |
| E. Planned actions | Reuse validation/manifests; verify hashes/membership/exclusions/view mapping/known byte duplicates; confirm train 2332/val333/test 666,paired233,prompts 1000. Review licence/access and exact prospective mount/visibility before any new upload. Ratify full target, section policy/pooling,decoder/tokenizer/length;500only labelled fallback. |
| F. Expected outputs | IUdata/profile/method contract and verified local manifest index; explicit remote-data permissions or blocker. |
| G. Verification | UID views never crosssplit; known duplicate-byte groups audited; raw bytes immutable; missing/unmapped images/excluded records accounted; matching counts alone not sufficient. |
| H. Completion criteria | Readiness contract names population/split/hash and chosen profile; no assumption500completes full IU. Existing preparation accepted if checks support it. |
| I. Failure handling | Unknown rights/mount/hash/view conflict stops affected action;resolve or record full IUblocked. Do not reselect/rebuild automatically. |
| J. Recovery | Preserve existingfull/subsetartifacts; any revised split gets new version and explicit comparability deviation. |
| K. Documentation | D-011decision;ledgerdata/profile/provenance;state/card. |
| L. Authorisation | Profile/research ratification and any remote upload require owner authority; existing local package is not approval. |
| M. Next dependency | 017IUruntime/cache qualification. |

### ZM-017

**Qualify IU loaders, text caches and frozen image path** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Establish the real study-level pipeline while reusing common components. |
| B. Current evidence | Validators, text preparation and view model components exist; real IU caches, trainer and image evaluation remain unverified. |
| C. Prerequisites | 016 approved profile;003 source;008 runtime;009 metrics; required downloads/GPU permissions. |
| D. Dependencies | ZM-016,003,008,009; IU manifests/configuration; D-007/011. |
| E. Planned actions | Reuse parsers/model; implement missing loader, collation and stage orchestration only. Encode train/validation text and train-only prompts with pinned encoder. Cache IDs/hashes/normalization; audit length, UNK and truncation. Test masked views, approved pooling, text CE, frozen training-image generation, full metrics and save/reload. |
| F. Expected outputs | IU-specific vocabulary, caches, prompt bank, configuration and measured smoke/readiness evidence. |
| G. Verification | Correct UID/all-view mapping; train-only vocabulary and1000 full prompts or250 subset prompts; finite features/loss; no test features used for training; metric fixture; full model/optimizer restoration and resource fit. |
| H. Completion criteria | Required mechanics/provenance verified; paper gaps and length/view policies declared and approved. |
| I. Failure handling | OOM/missing views/token length discrepancy stops; approved engineering amendment or labelled subset fallback; no implicit COV vocabulary/model substitution. |
| J. Recovery | Immutable inputs and per-profile cache names; new retry directory from approved state. |
| K. Documentation | Ledger implementation/cache/smoke events, artifact index and decisions; state/card. |
| L. Authorisation | Implementation/download/runtime/GPU checks explicitly authorized later. |
| M. Next dependency | 018 text training, then019. |

### ZM-018

**Complete IU text-only training and selection** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Produce an approved IU text-trained decoder and physical validation winner. |
| B. Current evidence | No real IU text training; COV historical epochs/patience are not IU state. |
| C. Prerequisites | 017 qualified; ratified fresh initialization/seed/validation/stopping/cap/chunk budget; training approval. |
| D. Dependencies | ZM-017; D-011; IU profile configuration; shared005 controls adapted explicitly. |
| E. Planned actions | Train decoder on frozen training-text features; validate ordered333 full or50 subset reports. Save full state/real winner each completed epoch; retrieve every approved chunk; stop under declared BLEU4/patience50 and approved cap; lock selection. |
| F. Expected outputs | IU losses, validation predictions, stopping history, full optimizer/RNG and durable winner. |
| G. Verification | No paired/test supervision; correct profile IDs/features; finite CE; recorded SGD settings; full restoration and reproducible strict validation selection. |
| H. Completion criteria | Declared stopping and physical winner verified; premature resource stop remains incomplete unless approved as a labelled deviation. |
| I. Failure handling | Preserve first failure; diagnose/retry only with approval; no reduced patience/model to force completion. |
| J. Recovery | Last retrieved complete full state; own IU counters, no inherited COV85 history. |
| K. Documentation | Ledger chunk/selection/provenance events; state/card; deviation decisions. |
| L. Authorisation | Each IU training session requires scoped resource approval. |
| M. Next dependency | 019 IU0 held-out experiment. |

### ZM-019

**Produce IU0 held-out outcome** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Complete study-level zero-paired image-to-report generation. |
| B. Current evidence | No IU outcome; existing full/subset preparation is not a result. |
| C. Prerequisites | 018 stopped winner;017 image path;009 full metrics; explicit inference/test approval. |
| D. Dependencies | ZM-018,017,009; approved profile and view policy. |
| E. Planned actions | Freeze all settings/hashes before test; generate every study from all mapped views; score BLEU1–4, METEOR and ROUGE-L; independently retrieve/recompute from saved predictions; compare Table 3 with population/split limits. |
| F. Expected outputs | results/iu_xray/<run>/ or results/iu_xray/kaggle_500/<run>/ complete prediction/metric/manifest package. |
| G. Verification | Exactly666 full or100 subset unique UIDs; correct views/references; complete metrics, termination, hashes and lineage; no tuning. |
| H. Completion criteria | Verified package and interpretation; subset completes only its labelled profile, leaving full-IU target open. |
| I. Failure handling | Quarantine integrity bugs and approve corrections; legitimate poor results reported without test optimization. |
| J. Recovery | Resume only identical frozen state and exact UID coverage; reuse predictions for scoring corrections. |
| K. Documentation | Ledger evaluation/provenance/comparison; state/status; D-011 for profile changes. |
| L. Authorisation | Separate IU GPU inference and held-out authorization. |
| M. Next dependency | 020 IU supervised contract;023 comparisons. |

### ZM-020

**Ratify and qualify IU supervised branches** — PARTIAL

| Field | Step contract |
|---|---|
| A. Purpose | Apply shared paired code with an IU-specific method/resource contract. |
| B. Current evidence | Shared model/paired logic exists; IU gradient, restoration and resource path unverified. |
| C. Prerequisites | 019 reviewed;012 paired principles;013 shared readiness; IU-specific method approval. |
| D. Dependencies | ZM-019,012,013; D-009/011; IU profile hashes. |
| E. Planned actions | Ratify visual parameters, view pooling, prompt policy, independent text-winner lineage, optimizer/RNG and stopping. Verify233/2332 full or35/350 subset IDs. Run approved tiny train-only multi-view update, full-state restoration and resource probe. |
| F. Expected outputs | Approved IU10/100 contract and verified bounded paired readiness. |
| G. Verification | Correct masked-view gradients and frozen invariants; full visual/decoder/SGD/RNG restoration; no test IDs; measured memory/runtime. |
| H. Completion criteria | IU method gaps closed, workload fits measured host and required state/input checks pass. |
| I. Failure handling | View-gradient/OOM failure blocks run; explicit engineering amendment or full-IU incompleteness, no implicit500 substitution. |
| J. Recovery | Reuse shared code and fixed IU inputs; smoke state separate from experiment initialization. |
| K. Documentation | Decision record for IU contract; ledger readiness/provenance; state. |
| L. Authorisation | Method/code/GPU smoke approval separate from later training. |
| M. Next dependency | 021 IU10. |

### ZM-021

**Complete IU10 experiment and evaluation** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Produce the IU few-paired result on a fixed study cohort. |
| B. Current evidence | Only paired IDs are prepared; no real experiment evidence. |
| C. Prerequisites | 020 qualified;019 recorded; scoped training and later evaluation approvals. |
| D. Dependencies | ZM-020,019; approved IU text winner, subset and configuration. |
| E. Planned actions | Initialize independent branch; train 233 full or35 subset pairs, validate 333 or50. Stop/persist approved chunks; lock real winner; score exact 666 or100 test UIDs with009 harness. |
| F. Expected outputs | IU10 full training state/history/winner and held-out prediction/metric package. |
| G. Verification | Correct subset/disjoint views/live encoder gradients; full restoration and selection; complete test IDs, metric fingerprint and file hashes. |
| H. Completion criteria | Stopped selected branch, durable artifacts and all required metrics; improvement over0% is not mandatory. |
| I. Failure handling | Failure/discrepancy recorded; retry justified/approved; no patient/test-based subset choice. |
| J. Recovery | Last retrieved completed full paired state; preserve0% outputs and all profiles. |
| K. Documentation | Ledger chunks/evaluation/provenance; state/card; deviation decisions. |
| L. Authorisation | Scoped IU sessions/test approval; COV permission does not transfer. |
| M. Next dependency | 022 after review;023 comparison. |

### ZM-022

**Complete IU100 experiment and evaluation** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Produce the full-paired IU comparison. |
| B. Current evidence | No real100% IU run. |
| C. Prerequisites | 021 integrity/result/resource review;020 ready; separate100% approval. |
| D. Dependencies | ZM-021,020; same approved text winner/profile as10% unless amended. |
| E. Planned actions | Initialize independent100% branch; train all 2332 full or350 subset studies, fixed validation; stop/preserve/select; generate and score fixed test cohort. |
| F. Expected outputs | IU100 training/state/winner and full held-out package. |
| G. Verification | All training UIDs/approved views; no implicit10→100 lineage; full restoration, selection, metric/hash/coverage checks. |
| H. Completion criteria | Stopped traceable experiment; full-IU success requires full profile, not500. |
| I. Failure handling | Resource/performance contradiction recorded; profile/deviation decision required; no automatic pruning. |
| J. Recovery | Verified full 100% latest; retain prior models/results. |
| K. Documentation | Ledger/provenance/state/card; decisions when changed. |
| L. Authorisation | Separate100% training/test approvals. |
| M. Next dependency | 023 comparisons and closure. |

### ZM-023

**Compare all outcomes and investigate discrepancies** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Assess the paper-guided reproduction without overclaiming. |
| B. Current evidence | Paper Tables1/3 known; final image outcomes pending. |
| C. Prerequisites | 011/014/015/019/021/022 verified, or each missing result has explicit owner-approved unsuccessful/unavailable disposition. |
| D. Dependencies | ZM-011,014,015,019,021,022; ledger/provenance; paper analysis Tables1/3. |
| E. Planned actions | Build six-row dataset×ratio matrix of actual metrics, lineage and profile; compare paper and same-cohort regimes. Check coverage, leakage, metric, preprocessing, model and protocol before explaining scientific discrepancy. Report missing original split/85/revisions and IU exclusions; do not require expected ordering. |
| F. Expected outputs | Comparison/limitations/diagnosis report with auditable links; unrun rows labelled pending, blocked or not reproduced. |
| G. Verification | Every number links scored predictions/checkpoint/configuration; reconstruction, smoke and paper values separate; no CIDEr scaling mix or invented variance/significance. |
| H. Completion criteria | Every target row addressed and discrepancy resolved or bounded. Reporting can close unsuccessful attempts, while unachieved target cards remain unachieved. |
| I. Failure handling | Integrity contradiction reopens affected card. Corrected rerun requires documented bug/protocol correction and approval; no test-driven method changes. |
| J. Recovery | Reuse predictions/artifacts; no expensive run for documentation convenience. |
| K. Documentation | Ledger comparison/diagnosis events, amendment decisions and state/status. |
| L. Authorisation | Read/report existing results in approved scope; no automatic new experiments. |
| M. Next dependency | 024 handoff; missing rows prevent claiming all outcomes achieved. |

### ZM-024

**Verify reproducibility and close scoped project** — NOT_STARTED

| Field | Step contract |
|---|---|
| A. Purpose | Deliver a recoverable implementation and final research account. |
| B. Current evidence | Durable recovery evidence exists; complete six-row outcomes do not. |
| C. Prerequisites | 023 report; accepted artifacts/decisions accessible; explicit owner review. |
| D. Dependencies | ZM-023; all referenced source/configuration/manifests/evaluation records. |
| E. Planned actions | Check fresh-session handoff and documented commands; verify artifact access/hashes and proportionate restore/metric recomputation where authorized. Package operating/source/results/limitations index and completion statement for owner review. |
| F. Expected outputs | Final reproduction report, reproducibility index, durable handoff and remaining-work record. |
| G. Verification | New session finds next action from state; all necessary artifact/source/data/config/environment links work; no terminal-history dependency; secret/report-bearing content reviewed before sharing. |
| H. Completion criteria | Full objective achieved only if both datasets0/10/100 packages verified. Alternatively owner accepts explicitly partial/unsuccessful closure without claiming missing outcomes complete. |
| I. Failure handling | Missing artifact/failed restore blocks closure; recover or record owner disposition, never silently discard. |
| J. Recovery | Canonical ledger/manifests/immutable exports; targeted verification, no automatic full training rerun. |
| K. Documentation | Ledger closure/review event; state actual completion/remaining steps; partial-closure decision if applicable. |
| L. Authorisation | Owner acceptance for final claim/sharing/Git publication; heavy replay separately approved. |
| M. Next dependency | Stop; new research scope needs new objective/authorization. |

## Controlled advancement, stopping and failure routes

| Transition | Evidence required | Authority required |
|---|---|---|
|Planning→execution|Owner reviews canonical records; exact 002 task/input/version and local/network prerequisites confirmed|Instruction to resume existing-run verification; no new job implied|
|Recovery→new decoder chunk|002 disposed;004 route ratified;003/005 contract;010 training-image mechanics; retrieved full latest; measured or explicitly bounded resource plan|Specific private target/source/attachments/epoch range/budget; each chunk separately approved|
|Text stage→final0%|007 physical validation winner/stopping;008/010 frozen image readiness;009 full metrics; untouched exact test cohort; fidelity limits settled|Specific inference/test evaluation approval|
|0%→10%|Dataset0% result reviewed;012/020 supervised gaps ratified; gradient/state/resource smoke passes; fixed train subset|Method approval and separately scoped paired training/evaluation|
|10%→100%|10% integrity/selection/evaluation reviewed; same cohort and approved independent initialization; resource budget ready|Separate100% authorization; no automatic10% checkpoint transfer|
|COV→IU|COV0% interpreted;016 data/profile/governance;017 runtime/view readiness|IU scope/profile and required download/upload/execution approval|
|Results→final closure|Complete traceability, discrepancy report, reproducibility checks, explicit missing rows|Owner reviews full or honestly partial/unsuccessful closure claim|

Before **any** substantive next task, verify prerequisites, artifacts/provenance, required evaluations, discrepancy disposition, permission and accurate state. Distinguish feasibility (could work), readiness (entry checks passed), completion (outputs verified) and authorization (owner permits action).

Stop the affected operation if a critical input/hash/assumption changes, a result contradicts acceptance, an artifact is untraceable, a significant method choice is unresolved, resources exceed the approved scope, or permission is absent. Record actual result and decide diagnosis, approved retry, controlled deviation, owner decision or unsuccessful outcome. Other independent read-only/documentation work can continue; do not mark blocked research work complete to keep moving.

| Failure / limitation | Controlled response | Recoverable boundary |
|---|---|---|
|Existing run has no phase evidence|Bounded fresh observation/live logs if available; do not infer stall/completion; owner decides cancellation|Independently verified90 until later export validated|
|ERROR/incomplete output|Record first actual traceback; separate surviving completed-state audit; no forced success verifier|Latest retrieved complete epoch, not partial in-memory state|
|Best/latest/history mismatch|Preserve disputed files; inspect pointer and actual tensors; no epoch/metric relabelling|Verified immutable winner/latest pair|
|Runtime/metric/cache drift|Reverify exact fingerprint and fixture; stop updates; record new evidence/amendment|Original frozen dependencies/predictions|
|OOM/quota/queue/disk limit|Measure actual bottleneck/quota; smaller approved chunk or approved engineering adaptation; no automatic pruning/deleting/model change|Last externally secured full state|
|Metric unavailable|Repair exact declared harness with scoped approval; no diagnostic substitute|Saved raw ref/pred pairs; no need to regenerate reports|
|Poor or suspiciously high score|Audit leakage/coverage/metric/protocol; report genuine disagreement; no test tuning|Frozen source/configuration/result package|
|FullIU infeasible|Owner-approved labelled500 contingency or explicit full-target incompleteness; no substitution in paper comparison|Existing separate full and subset manifests|
|Documentation unavailable|Report operational issue and stop additional substantive work until records can be saved|Last durable ledger/state/manifest|

No runtime/quota claim is assumed current from older Gate reports. Reverify official platform capabilities, installed CLI help, package compatibility, mounts and actual account/resource availability when executing. Inspect existing data provenance/access terms before new remote movement; available credentials or previously attached data are not blanket permission. Protect .kaggle and never print tokens/keys/signed URLs. Privacy metadata and actual server identity must both be checked.

## Durable operating cycle and handoff

Before: read [state](project_state.md), this card and relevant [decision](decision_log.md)/[ledger](execution_ledger.md) entries; verify expected outputs, prerequisites and permitted operations. Reuse established evidence; inspect originals only for a precise uncertainty.

During: stay on the card; collect source/configuration/environment/data identity and actual output evidence; label facts, inference and proposals. Keep expensive jobs bounded; do not expand to optional tools/novelty. Human questions concern only unresolved decisions the evidence cannot settle.

After each substantive implementation/configuration/training/inference/evaluation/troubleshooting activity: save attempted purpose, actual behavior including failure/deviation, checks, artifact locations/hashes and limitations; append event; update run index and decisions when needed; update this card if warranted and state/next eligible action; report clearly **before** another substantive task. An unverified outcome is recorded as unverified, not complete. Preserve sufficient restart inputs beyond terminal history.

Handoff format: active step/run/version; last verified boundary; current observation and timestamp; completed checks/evidence paths; blockers/pending decisions; next exact action and its permission/entry conditions. Related work may continue in the same session; a fresh session reads these records first. No automatic session-splitting or multi-agent requirement.

Amendment format in ledger: old step/dependency/criterion → new one; new evidence; affected artifacts/claims; decision/owner approval; superseded IDs if any. Significant research change stops for owner ratification. Pure ordering/documentation clarification can be maintained within scope and explained. Revision1 creates the ZM namespace; no prior steps deleted or renumbered.

## Definition of done and exclusions

A full successful scoped implementation/research package contains separate COV and full-IU0/10/100 outcomes, approved gap-fills, correct image-conditioned supervision, defensible validation selection/stopping, all specified held-out predictions/metrics, exact code/configuration/data/environment/encoder/checkpoint provenance, independently retrievable artifacts, reproducible operating instructions, comparisons and limitations. Expected paper scores/trends are comparison targets, not fabricated acceptance thresholds. Faithful exact-number reproduction cannot be claimed without original splits/architecture/revisions and lost85 weights; report what was actually reproduced.

A final report may honestly record unsuccessful or unrun targets with owner-approved disposition. That completes an accountable report, **not the missing experiment or full objective**. The state must distinguish full completion from partial/unsuccessful closure. IU500 outcomes never erase full-IU remaining work.

Not required for this objective: reproduce every external baseline architecture, implement ambiguous COV-CT ablations, invent clinical metrics, add novelty, switch hosts/tooling, or repeat expensive completed work for tidy documentation. Optional future multi-seed uncertainty studies, diagnostic controls, throughput improvements or licensed new datasets require distinct purpose/decision/resource approval and do not displace required targets.

## Revision1 consistency review

The [machine-readable review](evidence/phase1_governance_review.json) records repository starting identity, inspected evidence hashes, split conflict, card/graph/link/source-boundary checks and limitations. Objective/fractions checked against local paper evidence; existing accomplishments mapped in ledger; all major outputs have cards; all cards define A–M; dependency graph checked for cycles; approvals distinct from method assumptions; architecture/provenance ownership explicit; next action reconstructable from state. No code, environment, data, notebook, checkpoint, remote state, Git staging/commit/push or implementation step changed/executed. Plan remains reviewable documentation, not implementation authorization.
