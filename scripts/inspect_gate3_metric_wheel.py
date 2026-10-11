"""Inspect a pinned metric wheel and hash its actual PTB/BLEU implementation; no install."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate3/dependency_audit"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    a = json.loads((ROOT / "docs/evidence/gate3_dependency_audit.json").read_text())
    wheel = next(f for f in a["packages"]["pycocoevalcap"]["files"] if f["filename"].endswith("py3-none-any.whl"))
    path = OUT / wheel["filename"]
    if not path.exists():
        with urllib.request.urlopen(wheel["url"], timeout=60) as response, path.open("xb") as target:
            for block in iter(lambda: response.read(1048576), b""):
                target.write(block)
    assert digest(path) == wheel["digests"]["sha256"], "Metric wheel integrity mismatch"
    files = {}
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for name in ("pycocoevalcap/tokenizer/ptbtokenizer.py", "pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar",
                     "pycocoevalcap/bleu/bleu.py", "pycocoevalcap/bleu/bleu_scorer.py"):
            content = archive.read(name)
            files[name] = hashlib.sha256(content).hexdigest()
        # Only tiny Python source is extracted for syntax/source inspection; no package installation.
        for name in archive.namelist():
            if name.startswith(("pycocoevalcap/bleu/", "pycocoevalcap/tokenizer/")) and name.endswith(".py"):
                destination = OUT / "metric_source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                compile(destination.read_text(), name, "exec")
    report = dict(package="pycocoevalcap==1.2", wheel=wheel, implementation_sha256=files,
                  local_java="UNAVAILABLE; tokenizer runtime not executed locally",
                  status="Wheel integrity, Python syntax and implementation hashes verified; no install")
    (ROOT / "docs/evidence/gate3_metric_implementation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
