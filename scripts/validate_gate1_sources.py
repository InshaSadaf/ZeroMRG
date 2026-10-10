"""Static notebook validation only; never execute notebook code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = (ROOT / "scripts/zeromrg_gate1_runtime.py").read_text(encoding="utf-8")
    audit = json.loads((ROOT / "artifacts/reconciliation/gate1_20261010_checkpoint_audit/checkpoint_audit.json").read_text(encoding="utf-8"))
    canonical = next(item for item in audit["archives"] if item.get("verification", {}).get("epoch") == 90)
    report = []
    for directory in sorted((ROOT / "notebooks/gate1").iterdir()):
        meta = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
        path = directory / meta["code_file"]
        raw = path.read_bytes()
        nb = json.loads(raw)
        assert nb["nbformat"] == 4 and len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
        assert meta["is_private"] is True and meta["enable_gpu"] is True
        assert meta["id"] != "inshasadaf114/zeromrg-workflow"
        assert meta["dataset_sources"] == ["inshasadaf114/dataset", "inshasadaf114/zeromrg-checkpoint-backup5-zip"]
        assert "".join(nb["cells"][1]["source"]).strip() == runtime.strip()
        pinned = {}
        for statement in ast.parse("".join(nb["cells"][2]["source"])).body:
            assert isinstance(statement, ast.Assign) and len(statement.targets) == 1
            pinned[statement.targets[0].id] = ast.literal_eval(statement.value)
        assert pinned["CANONICAL_DEPENDENCY_HASHES"] == canonical["dependency_hashes"]
        assert pinned["RUNTIME_SOURCE"] == runtime
        assert pinned["RUNTIME_SHA256"] == hashlib.sha256(runtime.encode()).hexdigest()
        assert (directory / meta["code_file"]).is_file() and meta["enable_tpu"] is False
        code = []
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                source = "".join(cell["source"])
                compile(source, f"{path.name}:cell{index}", "exec")
                code.append(source)
        if directory.name == "preflight":
            calls = [n for s in code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Call)]
            assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {"step", "backward"}
                           for n in calls), "Preflight must not train"
            assert "validation_bleu4()" not in "\n".join(code)
        else:
            assert "ENABLE_APPROVED_TRAINING = False" in code[-1]
            assert "EXPECTED_SOURCE_EPOCH" not in "\n".join(code)
            assert "restore_training(checkpoint, decoder, best, DEVICE)" in "\n".join(code)
            assert "session_final_epoch = min(200, NEXT_EPOCH + 4)" in "\n".join(code)
        report.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(),
                           code_cells=len(code), python_compile="PASS", embedded_source="PASS",
                           pinned_dependency_hashes="PASS", metadata="PASS", execution="NOT RUN"))
    for path in (ROOT / "scripts").glob("*gate1*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    destination = ROOT / "docs/evidence/gate1_notebook_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
