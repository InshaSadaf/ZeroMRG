> **Governance authority notice, 2026-10-11:** Current project position is maintained only in [project_state.md](project_state.md); roadmap and advancement are in [implementation_master_plan.md](implementation_master_plan.md). This original body is preserved as historical evidence or a detailed technical procedure, not a competing current-state or authorization record. Later scoped approvals/results are indexed in the execution ledger. The later reviewed training session used existing private baseline kernel output; a new private dataset is not an automatic restart prerequisite. Preserve historical evidence and follow the master plan amendments.

# ZeroMRG Gate 3 recovery decision

Updated 2026-10-11 for the existing `E:\Insha Project\ZeroMRG` COV-CTR 0%-paired reproduction. **The single authorized private Gate 3 baseline-only version completed and its export was independently verified.** No decoder training, image generation or test scoring occurred. Gate 1 and Gate 2 remain their original historical assessments; this is a separate, explicitly labelled recovery protocol. Preparation documents were preserved before this update.

## Evidence and immutable history

[Gate 1](zeromrg_gate1_readiness.md) verified the historical checkpoint-producing decoder, not the incompatible modular defaults. [Gate 2](zeromrg_gate2_preflight.md) actually verified the mounted dataset/dependency hashes, complete epoch90 decoder and SGD state (112/112 nonzero finite momentum buffers), GPU forward computation and disposable full-state save/reload on Tesla T4. Gate 2 deliberately stopped on the epoch85-versus8 discrepancy. The remote best artifact downloaded in Gate 1 had the same stale bytes. **No genuine epoch85 weights have been verified.** Bounded negative searches do not prove global absence. Gate 2 showed that a header-only `kernels files` listing is insufficient negative evidence: actual output retrieval must also be attempted in any future authorized search.

Separate exclusive-created, hash-verified, read-only local copies are under `artifacts/gate3/historical_evidence/epoch90_ecc204f031b0`. [Preservation manifest](evidence/gate3_historical_preservation.json) records the three originals and Gate 1/2 verification reports. Read-only attributes protect against accidental writes; they are not tamper-proof storage. Existing source files remain untouched, and large copies are ignored by Git.

| Original artifact | Saved metadata | SHA-256 |
|---|---|---|
| `text_decoder_last.pt` | Epoch90, 57,399,152 decoder parameters, complete SGD; 459,275,387 bytes | `ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299` |
| `text_decoder_best.pt` | Actual epoch8, BLEU4 52.33887482243904; 229,639,049 bytes | `dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568` |
| `text_training_history.json` | Original90 records, epoch85 selected, best56.44423922571333; 9,876 bytes | `ccd65f2ec0ce2f569f683233f5a13c48c26bb27075aa9661bc7a272c3a14a015` |

The original historical best is **56.44423922571333 at85**, with saved **5/50** consecutive non-improvements, all unchanged. The recorded epoch90 score and freshly measured physical90 baseline both equal54.21140850351637. These are text-reconstruction validation results, not image-to-report results. The baseline was actually measured on all71 validation reports; it was not copied from historical records.

## Policy comparison

| Consideration | A: original stopping, independent recoverable-best ledger | B: independent recovery selection and stopping window | C: rerun from earlier checkpoint |
|---|---|---|---|
| Scientific validity | Defensible continuation with unchanged original stopping logic; available-window selection explicitly disclosed | Defensible new recovery experiment if clearly separated; resetting stopping creates an additional deviation | Defensible rerun branch using real earlier weights/SGD; does not reconstruct missing85 |
| Selection consequences | Actual validation winner among measured90 and later continuation epochs; can remain below historical85 score | Same available-window selection, but independent improvements can prolong training | New trajectory has its own validation-selected winner; original85 weights still unavailable |
| GPU cost | One90 validation baseline, then at most110 additional epochs to inherited200 ceiling; stop135 if no original-best improvement | Baseline plus continuation; independent patience50 would reach140 if never improved, and lower window improvements may prolong it further | Verified75 full checkpoint permits a new branch, adding15 epochs to reach90 before further work; cheaper epochs8 best lacks optimizer/history |
| Reproducibility limits | Original RNG absent, historical Torch2.10 versus observed2.11 and encoder revisions unknown; new RNG saved going forward | Same limits plus a new stopping experiment | Earlier checkpoints also lack RNG; stochastic85 trajectory cannot be recovered exactly from logs or seed alone |
| Base-paper compatibility | Keeps pretrained encoders, text-only training, decoder, SGD and validation-only selection; recovery-window restriction is disclosed | Same architecture, but additional stopping rule is not justified as a paper requirement | Same architecture possible, but rerun cost and changed trajectory do not repair original selection evidence |
| Misreporting risk | Low only if labelled recovery90+; never call it original/global best or missing85 | Higher if reset patience or extra epochs are concealed | High if new epoch85 is presented as the missing original model |

**Recommend A**, implemented as `A_historical_stop_available_best_v1`. It satisfies the user's proposed validation-only epoch90 baseline, keeps original patience and metric intact, and produces a real retrievable winner without claiming to recover unavailable weights. Approach B is not implemented. Approach C is not justified by current evidence or compute cost; rerunning cannot recreate the lost stochastic trajectory exactly. Recovery-best selection is an explicit methodological recovery deviation, even under A. No claim that this choice is prescribed by the paper is made.

## Two ledgers, no substitution

`historical_record` and `original_history` remain immutable. `original_stop_tracking` starts from the historical best85/56.44423922571333 and counter5. Only continuation91+ validation measurements update this tracking ledger: strictly greater than its current best resets the counter; ties or lower scores increment it. Remeasuring90 does **not** alter historical stopping.

`recovery_history` begins with a fresh validation90 measurement and a real saved90 recovery-baseline checkpoint. Strictly better later validation BLEU4 selects and saves a new recovery-best file. Ties keep the earlier winner. Its non-improvement counter is informational and never controls stopping. Thus a52 score can improve a51 recovery baseline while original stopping advances from5 to6. A later57 score improves both trackers, but never rewrites the fixed original85 evidence.

Report recovery results as “available-window validation-selected recovery checkpoint, original epoch85 weights unavailable.” Text reconstruction is not zero-shot image generation. Test72 remains untouched for generation/scoring/model selection until the separately approved final experiment.

## Readiness decision

**Gate 3 baseline: PASS. Recovery epoch91 is technically ready; training and final inference remain unauthorized.** The completed private version1 measured54.21140850351637 on validation71, restored/saved unchanged90 model and SGD, verified Java/PTB, and exported a full checkpoint that was independently retrieved/hash/schema/tensor verified. No new trained winner or encoder runtime is verified. All65 preparation CPU tests and11-cell static checks remain historical preparation evidence. See [execution evidence](zeromrg_gate3_baseline_execution.md), [current readiness](zeromrg_gate3_readiness.md), [source verification](evidence/gate3_notebook_validation.json) and [local test evidence](evidence/gate3_local_test_results.json). The old Gate 1 continuation remains unchanged and refuses missing85; only the separately labelled Gate 3 policy addresses available-window selection.

See [training protocol](zeromrg_gate3_training_protocol.md), [dependencies](zeromrg_gate3_dependencies.md) and [persistence](zeromrg_gate3_persistence.md). The one baseline authorization is exhausted; **do not resubmit it**. Next requires explicit approval for a reviewed new private recovery attachment and then a bounded epoch91+ training session. Actual server target is `inshasadaf114/zeromrg-covctr-gate3-recovery-baseline`; prepared metadata still records the shorter requested ID and must be reviewed before any future submission. No publication or additional execution is authorized now.
