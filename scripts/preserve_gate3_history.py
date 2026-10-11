"""Hash-verified, non-overwriting, read-only copies of historical evidence."""
import hashlib
import json
import shutil
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/gate3/historical_evidence/epoch90_ecc204f031b0"
SOURCE = ROOT / "artifacts/reconciliation/gate1_20261010_checkpoint_audit/epoch90"
EXPECTED = {
    "text_decoder_last.pt": "ecc204f031b0fa83b4734c3228f20ba2172ce291288f31a6551921126546c299",
    "text_decoder_best.pt": "dba64646e89f1c55c69c9e3a7361141829e4dc7ec32114f98f2edce8b89a0568",
    "text_training_history.json": "ccd65f2ec0ce2f569f683233f5a13c48c26bb27075aa9661bc7a272c3a14a015",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    sources = {name: SOURCE / name for name in EXPECTED}
    for name in ("gate1_checkpoint_audit.json", "gate1_remote_best_audit.json",
                 "gate1_notebook_validation.json", "gate1_local_test_results.json",
                 "gate2_preflight_report.json", "gate2_execution_log.json", "gate2_preflight_audit.json"):
        sources[name] = ROOT / "docs/evidence" / name
    manifest = {name: dict(sha256=digest(path), bytes=path.stat().st_size) for name, path in sources.items()}
    assert all(manifest[name]["sha256"] == value for name, value in EXPECTED.items())
    history = json.loads(sources["text_training_history.json"].read_text())
    selected = max(history, key=lambda row: row["validation_bleu4"])
    assert len(history) == 90 and selected["epoch"] == 85 and selected["validation_bleu4"] == 56.44423922571333
    DEST.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        target = DEST / name
        if target.exists():
            assert digest(target) == manifest[name]["sha256"], "Existing snapshot differs; never overwrite"
        else:
            with path.open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            assert digest(target) == manifest[name]["sha256"]
        target.chmod(stat.S_IREAD)
    record = dict(directory=str(DEST.relative_to(ROOT)), files=manifest,
                  historical_best_epoch=85, historical_best_validation_bleu4=56.44423922571333,
                  historical_non_improvement_counter=5, patience=50,
                  preservation="Exclusive creation, hash verification and read-only copies; originals unmodified",
                  immutability_limit="Filesystem read-only is an accidental-write guard, not tamper-proof storage")
    encoded = json.dumps(record, indent=2) + "\n"
    local_manifest = DEST / "preservation_manifest.json"
    if local_manifest.exists():
        assert local_manifest.read_text(encoding="utf-8") == encoded
    else:
        local_manifest.write_text(encoded, encoding="utf-8")
        local_manifest.chmod(stat.S_IREAD)
    (ROOT / "docs/evidence/gate3_historical_preservation.json").write_text(encoded, encoding="utf-8")
    # A separate extension preserves narrative reports without altering the first immutable manifest.
    reports_dir = DEST / "reports"
    reports_dir.mkdir(exist_ok=True)
    reports = {}
    for name in ("zeromrg_gate1_readiness.md", "zeromrg_checkpoint_recovery.md",
                 "zeromrg_source_reconciliation.md", "zeromrg_checkpoint_persistence.md", "zeromrg_gate2_preflight.md"):
        source = ROOT / "docs" / name
        target = reports_dir / name
        source_digest = digest(source)
        if not target.exists():
            with source.open("rb") as handle, target.open("xb") as output:
                shutil.copyfileobj(handle, output)
        assert digest(target) == source_digest, "Narrative evidence changed; refuse overwrite"
        target.chmod(stat.S_IREAD)
        reports[name] = dict(sha256=source_digest, bytes=target.stat().st_size)
    report_record = json.dumps(dict(directory=str(reports_dir.relative_to(ROOT)), files=reports), indent=2) + "\n"
    report_manifest = reports_dir / "reports_manifest.json"
    if report_manifest.exists():
        assert report_manifest.read_text(encoding="utf-8") == report_record
    else:
        report_manifest.write_text(report_record, encoding="utf-8")
        report_manifest.chmod(stat.S_IREAD)
    (ROOT / "docs/evidence/gate3_report_preservation.json").write_text(report_record, encoding="utf-8")
    print("Historical checkpoint/history/Gate 1/Gate 2 copies verified:", DEST)


if __name__ == "__main__":
    main()
