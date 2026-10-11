# Gate 3 bounded runtime investigation

Observed 2026-10-11 10:11:58–10:12:08 UTC (15:41:58–15:42:08 IST). Decision: **INSUFFICIENT EVIDENCE of actual execution progress or failure.** Recovery readiness remains BLOCKED pending independently retrievable checkpoints. No cancellation or submission occurred.

## Observations and their limits

Exactly three targeted read-only requests were made: project-wrapper `kernels status`, SDK `get_kernel`, and SDK `list_kernel_session_output`. No polling loop was run. Evidence is under `artifacts/gate3/runtime_incident_20261011T101158310018Z`: `cli_status.txt`, `server_notebook.ipynb`, `source_static_audit.json`, and `investigation.json`. No credentials or signed download URLs are recorded.

- CLI status: `KernelWorkerStatus.RUNNING`, exit0. Exact target: `inshasadaf114/zeromrg-covctr-gate3-recovery-epochs-91-95`, version1, numeric ID138131184, private, GPU enabled. Both dataset attachments and the private baseline kernel-output attachment agree with the submission record.
- Fresh source SHA-256: `c1b92cae947fdadd281f2dab18d8215811fce3f59ac171109d399c8d0566fbb1`, identical to the originally retrieved server source; all11 cells agree. Local submitted notebook SHA was `b331e603e50654334ea61f8463208a6c5d28ad675fa3a920a058ec9baa0e5fbc`; serialization differs, source cells do not.
- Output endpoint: **zero files, zero log characters, no further page**. No traceback, completed cell, epoch, training batch, validation batch or export phase is observable. No newer checkpoint is retrievable through this endpoint. This does not prove there are no checkpoints in the running session's filesystem.
- The local submission snapshot was created at06:23:41 UTC and first execution/source checks were near06:25 UTC. Approximately3h48 elapsed to this observation. This is local wall time since submission evidence, not measured GPU runtime.
- Server `last_run_time` was `2026-10-08 10:12:01.632000` (timezone absent). It conflicts with local October11 submission evidence. The approximately72-hour arithmetic difference recorded in the snapshot is **not an established execution duration**. The API fields do not establish queue/provision/start/finish timestamps or reconcile this clock inconsistency.
- The actual training preparation/submission records request **7200 seconds**, not1800. The previous baseline-only run used1800. Installed CLI passes `timeout` into `session_timeout_seconds`; returned status metadata does not report the effective timeout or its enforcement. Source epoch limits and provider time limits are separate controls.
- Actual GPU type, utilization, memory, billed duration and remaining quota are **UNOBSERVABLE through the inspected installed interfaces**. GPU enablement is configuration, not evidence of utilization. Tesla T4 was independently observed for the baseline, not this running execution. The installed status response has only status/failure-message fields. No enabled browser interface was available in the earlier inspection; no browser modification was attempted here.

## Exact source audit

Cell references below are zero-based notebook indices, including the introduction.

| Potential cause | Source finding | Assessment |
|---|---|---|
| Incorrect bounds / repeated execution | Cell10 requires start91/end95/MAX_NEW_EPOCHS5; loop clamps to95 and each iteration asserts bounds; persistent one-attempt guard | No unbounded epoch loop or automatic restart found |
| Dependency installation | Cell7 installs only hash-pinned pycocoevalcap1.2; no explicit subprocess deadline on pip or Java version query | Possible installation/network hang; no runtime evidence locating execution here |
| DataLoader | Cell10 has571 records, batch6,96 batches, `num_workers=0` | Worker-process deadlock is not supported by this source; ordinary CUDA/runtime stalls remain possible |
| Validation | Cell8 processes exactly71 reports in12 batches; up to159 decoder calls per batch with full growing prefix and no cache | Bounded but potentially expensive; no measured phase timing or evidence of current validation |
| PTB/BLEU | Pinned PTB/Java operations used for validation and later rescoring; caller imposes no total tokenizer deadline | Possible external-process wait; no tokenizer error/traceback available |
| Progress visibility | No batch-level training/validation logs; epoch-completion print occurs only after validation, checkpoint save and integrity hashing | Long quiet intervals are plausible; status alone cannot prove progress |
| Serialization / export | Full latest about459MB each epoch; each selected winner about230MB. Initial restored90 export precedes training; final export follows final rescoring/restoration checks | Repeated CPU transfers/load/hash/write and ZIP I/O are plausible bottlenecks |
| Working-directory persistence | Extracted baseline about1.38GB, restored run evidence/latest/winners, initial roughly1.38GB ZIP, final ZIP and installed metric files remain under `/kaggle/working` | Multiple large copies can increase disk usage and final output publication cost; actual publication phase unobserved |
| Recomputed features/models | Cached text features/dependencies restored; no CLIP image encoding or feature recomputation called | No unexpected image-model loading found |
| Hidden errors / recursion | Explicit assertions/raises; no retry recursion or training exception-swallowing handler found | No source evidence of silently repeated work; absent API logs prevent runtime diagnosis |

These are plausible bottlenecks and missing observability controls, **not a diagnosed first failure**. Nothing observed supports claiming training completed or failed.

## Preserved recovery evidence

Fresh streaming SHA-256 checks match prior independently verified values for all six files, with expected sizes. Exact paths/hashes are in `investigation.json`.

| Preserved artifact | SHA-256 |
|---|---|
| Original latest90 | `ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299` |
| Stale actual best8 | `dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568` |
| Original90 history | `ccd65f2ec0ce2f569f683233f5a13c48c26bb27075aa9661bc7a272c3a14a015` |
| Verified recovery latest90 | `f641f7ef7312e74b2c79c0b76f1920498999a90328f296f50d75950784a4e404` |
| Verified recovery best90 | `6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2` |
| Independently retrieved baseline ZIP | `dec3cd3391e2c107a7c9ed6209997f00554cd47cb0940ca339a74446253680b1` |

Matching hashes preserve the previously verified model/112 momentum buffers/RNG/history/schema evidence without rerunning GPU computations. Latest independently verified completed epoch remains **90**, with validation reconstruction BLEU4 **54.21140850351637**. Original historical85/56.44423922571333 and5/50 remain preserved; current remote stopping state is unknown. Cancelling could lose all unretrieved post90 work, including completed-epoch latest/winners if Kaggle does not publish them after cancellation. No guarantee of cancelled-session output retention is made.

## Recommendation and next decision

**INSUFFICIENT EVIDENCE.** RUNNING and absent published outputs cannot distinguish useful slow training, setup, external-process wait, export/publication delay or a stall. KEEP RUNNING lacks observed recent progress; STOP RECOMMENDED lacks observed evidence of a nonproductive phase. Owner inspection of this exact version's live Kaggle Logs/Session/GPU panels would resolve more than additional identical status polling. No cancellation has been performed. Cancellation of this exact version requires explicit user approval; any future diagnostic submission requires separate approval and resolution of the existing job.

See [minimal future diagnostic design](zeromrg_gate3_one_epoch_diagnostic_plan.md). Current submitted source, notebooks, checkpoints, datasets, historical records and earlier evidence are untouched.

Local verification: all code cells in the freshly retrieved source parsed successfully; two CPU-only timeout/logging tests passed in0.106 seconds. `git diff --check` passed (line-ending notices only). Existing user modifications and staged files were preserved; no Git staging, commit or push was performed. No local GPU computation occurred. Investigation ended without another remote request or operation.
