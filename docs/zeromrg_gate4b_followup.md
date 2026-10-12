# Gate4B controlled experiment — terminal verification

2026-10-12; D-018 / EV-055–061 / ZM-010_GATE4B_IMAGE_CONDITIONING_V1. **BLOCKED_RUNTIME_DATA_MOUNT**. The sole notebook stopped before model loading or generation. [42 independent checks](../artifacts/gate4b/followup_20261012/failure_verification.json), [retrieval hashes](../artifacts/gate4b/followup_20261012/retrieval.json), [worker traceback](../artifacts/gate4b/followup_20261012/retrieved/zeromrg_gate4b_d057a78581da/execution.log), [platform log](../artifacts/gate4b/followup_20261012/platform_execution.log). The [previous unresolved handoff](zeromrg_gate4b_execution.md) remains historical evidence.

**A. Kaggle identity/status.** [inshasadaf114/zeromrg-covctr-gate4b-image-conditioning](https://www.kaggle.com/code/inshasadaf114/zeromrg-covctr-gate4b-image-conditioning), numeric ID **138214712**, **version1**, private. Exactly one submission accepted2026-10-11T19:06:55.763655+00:00. Identity/privacy/GPU flag/attachments/source cells reverified2026-10-12T07:55:59.869902+00:00. Final status **ERROR**, observed07:56:01.207611+00:00; API failuremessage empty, worker/platform logs agree on failure. One fresh status and one retrieval; monitoring stopped. No second submission.

**B. Inputs/membership.** Local precheck PASS: eight prepared hashes/reused sources and embedded configuration unchanged; canonical571training/500prompt CSV hashes, membership and saved prompt order independently verified. All five remain the first eligible training IDs outside the original bank, selected without report-content inspection. Canonical image_id is the associated report-row identifier; no separate report_id column.

| Order | Image/report-row ID | Split | Original500 prompts | Generated report |
|---|---|---|---|---|
| 1 | `25%2.jpg` | checkpoint_train | absent | not generated |
| 2 | `PIIS0140673620303603%5.png` | checkpoint_train | absent | not generated |
| 3 | `1094.png` | checkpoint_train | absent | not generated |
| 4 | `2%0.jpg` | checkpoint_train | absent | not generated |
| 5 | `1868.png` | checkpoint_train | absent | not generated |

First image scheduled once more/maxsix generations; **actual generations zero**, established from exact source/traceback/phase logs. Local precheck reverified selected image bytes/CRC; remote selected-image access never reached. No held-out images/references used.

**C. Raw matrix.** No embeddings extracted. Requested5×5 raw cosine matrix, shapes/norms/finiteness and off-diagonal min/max/mean **unavailable**. No invented matrix supplied.

**D. Enriched matrix.** No enrichment executed;5×5 matrix/statistics **unavailable**. Source preserves raw500×768 bank, `softmax(v Pᵀ/√768)` across500 prompts, weighted prompt sum, residual image addition and finalL2 normalization. Correct source does not substitute for measured conditioning.

**E. Geometry comparison.** Unresolved; none of B1–B4 can be assigned. No collapse, homogenization, preserved variation or improved alignment demonstrated.

**F. Reports/tokens/provenance.** Every report/token sequence/count/EOS/maximum-length termination/inference time **unavailable**. Planned checkpoint physical90 SHA256 `6166e36198737bfbe9e0c35f03f0b320c44e76f3e05068ef6299d74e9179c6b2`, historical six-layer57,399,152-parameter decoder, vocab368/BOS1/EOS2/PAD0/UNK3, greedy max160. ZIP and selected input guards passed; strict decoder restoration **not reached** in this run. No original checkpoint/prompt/training artifact modified.

**G. Unique counts/equality.** Counts across requested five reports and pairwise token-equality matrix **unavailable** because no observations exist. Zero generations is not five identical reports or decoder insensitivity. Intended normalized-report comparison uses NFKC+casefold+whitespace collapse, punctuation retained.

**H. Repeat determinism.** Unmeasured: neither execution of `25%2.jpg` reached model forward. No raw/enriched/token/text equality result. Prior single-image success/mocks do not certify it.

**I. Runtime/memory/failure.** Failure JSON records worker **30.350600589999885 seconds**, including setup/artifact checks; not image-inference time or full Kaggle session duration. CUDA availability and pinned Torch2.11.0+cu128/CUDA12.8/torchvision0.26.0/NumPy2.1.3/Pillow12.3.0 guards passed before failure, established by exact source flow. Logs show support packages and pinnedCLIP source installation succeeded. CLIP model weights not loaded. GPU model/free/peak memory and per-image times were not recorded.

First causal exception: `ValueError: COV data mount differs`, [worker line77](../scripts/gate4b_diagnostic_worker.py). Configured root `/kaggle/input/datasets/inshasadaf114/dataset/COV-CTR with English reports` must be a directory and resolve within `/kaggle/input`; at least one condition was false. Actual runtime mount inventory is absent, so correct path and exact cause remain unverified. Failure follows baseline ZIP SHA/selected-member CRC/SHA guards and precedes prompt loading, selected-image resolution, decoder restoration, CLIP weight loading and generation. No timeout/OOM/quota/model defect inferred.

Retrieved nine files **62,055 bytes**, plus platform log; all independently rehashed. Five executed sources match prepared source; configuration matches semantically despite formatting/hash differences. Independent **failure/provenance verifier42checks PASS**; success-output numerical verifier not invoked on missing data. No CPU model forward or additional remote query.

**J. RQ1–RQ6.** RQ1 encoder discrimination, RQ2 preservation of enrichment variation, RQ3 decoder sensitivity, RQ4 repeat determinism and RQ5 generic/collapsed reports remain unanswered. RQ6 modality gap remains unmeasured. This failure identifies a data-access prerequisite and says nothing about semantic usefulness of embeddings or decoder response. Changing the image has **not yet been tested** by this five-image experiment.

**K. Limits/additional comparison.** Five training images; associated reports were available in text reconstruction, so prompt exclusion does not establish unseen-report generalization/patient independence/clinical correctness. Image-image similarities cannot quantify `P(z_text)` versus `P(z_image_enriched)`; greater cosine is not proven alignment. Historical85 weights missing, stopping incomplete and encoder/cache revision unverified. Epoch90 text-reconstruction BLEU4 **54.21140850351637** is not image-to-report BLEU4.

Training-text cache SHA `7275ebb5ea1a9ef273eee38b16e45fe6a3f42f154bb844ca2ba890e09853863a` may support a **separately approved** descriptive CPU comparison after image vectors exist: verify training-only IDs/conditioning normalization, compare corresponding cached text and enriched-image vectors plus cross-sample controls, and describe cosine/norm/centroid geometry. No encoder rerun/alignment training/test references; this remains limited and cannot prove clinical accuracy. Not conducted here.

**L. Decision.** **Gate4B BLOCKED_RUNTIME_DATA_MOUNT**; terminal failure/provenance verified, scientific acceptance unmet. Original Gate4 single-image mechanics PASS remains valid. ZM-008/009PARTIAL, ZM-011unstarted. No retry, model/decoding changes or extra execution.

**M. Gate5 readiness.** Preserve epoch90 configuration as candidate; **not ready to advance to Gate5**. Require Gate4B numerical/token evidence, owner disposition of physical90/fidelity limitations and metric readiness before held-out execution. No selection/diversity tuning justified.

**N. Next approval.** Authorize **local mount-resolution correction and focused tests only**, establish justified runtime mount-path evidence or a narrowly scoped way to obtain it, fail closed on ambiguity, reverify selected image hashes, and prepare a reviewable revised notebook without submission. Any subsequent Kaggle execution requires **new explicit one-run authorization**; D-017's ONE submission consumed. No correction/new run/Gate5/further monitoring performed in this handoff.

Scientific-critical-thinking skillv1.5 guided appraisal. Attribution: Kassis,T., Agarwal,V., He,Y., Patel,D., & Brueckner,A.M.(2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065); currentv2 metadata verified2026-10-12. No experimental findings attributed to this procedural source.
