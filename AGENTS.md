# ZeroMRG persistent operating instructions

This repository reproduces the supplied ZeroMRG paper on COV-CTR and IU-Xray, without novelty. Preserve useful completed work, immutable raw data and genuine checkpoints. The canonical governance system is:

1. Read [docs/project_state.md](docs/project_state.md) first on every resume.
2. Read the active card in [docs/implementation_master_plan.md](docs/implementation_master_plan.md), then relevant [decisions](docs/decision_log.md), [ledger/provenance](docs/execution_ledger.md) and linked technical evidence.
3. Apply instructions in research/AGENTS.md when working there, with its stale COV_CT naming/current-state claims resolved by the explicit governance notice and actual supplied COV-CTR evidence.

The Phase1 planning prompt authorizes documentation only. **Do not start implementation, installs, downloads, remote checks/uploads/submissions, training, inference or evaluation from a plan entry.** The owner reviews artifacts and supplies the next action's scope. Prior one-session execution permissions do not authorize another session. Later explicit authorization should be used without asking for the same permission again; retain the scope in decision/ledger records.

Before a task: identify stable step/run ID, prerequisites, exact input/source/configuration, expected outputs/checks and permitted operations. Reuse proved evidence; do not repeat expensive work for documentation. If a research choice is unresolved, inspect evidence then ask a focused owner question only when necessary. ref_docs/ contains governance examples, not authority for this project's method/hardware/tools/permissions.

During: stay on the step. Distinguish observation, inference, gap-fill proposal and approved choice. Keep dataset/profile/stage boundaries explicit. Stop affected work on contradictory acceptance, lost provenance, false critical assumption or exceeded authority; no automatic retry, architecture substitution, patience reset or test tuning.

After **every substantive** code/configuration/training/inference/evaluation/troubleshooting action: preserve actual output/failure/deviation and evidence; verify card acceptance; append ledger event with source/config/data/environment/checkpoint/artifact provenance; update master card status when justified; update current state/next action; update significant decisions/approval usage; report result before moving onward. Command success alone is not completion. If durable records cannot be written, report an operational issue and stop further substantive work.

Recovery invariants: current independently verified boundary90; original selected85 missing/stale best 8; current submitted91–95 outcome must be resolved before any duplicate job. Preserve separate historical stopping and available-window winner; use full latest for restart. Exact checkpointCSV COV splits/vocab/caches/raw prompts differ from modular data/processed and model defaults. Never swap based on matching counts/dimensions. Treat broad notebooks/experiment driver as unsafe automatic launchers unless explicitly reviewed for the selected stage.

Local work and CPU artifact checks are distinct from approved Kaggle GPU execution. Reverify time-sensitive platform/quota/package/mount/privacy constraints when executing. Unique non-overwriting retrieval/run directories; independently retrieve/hash/CRC/state-verify completed exports before another chunk. Protect .kaggle credentials; never log secrets/signed URLs; do not discard staged/untracked user work or checkpoints. No Git staging/commit/push or destructive cleanup without applicable authorization.

Handoff through durable records: active step/run/version; last verified boundary; timestamped observation; evidence paths; blockers/decisions; exact next action and its entry/permission conditions. If recommendations change, record new evidence and affected plan amendment. Do not reconstruct the entire project from terminal history or ask the owner to repeat known context.
