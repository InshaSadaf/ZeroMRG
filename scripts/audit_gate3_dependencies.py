"""Read official package metadata/source. Download inspection artifacts; never install."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/gate3/dependency_audit"
PINS = {"pycocoevalcap": "1.2", "pycocotools": "2.0.10", "multilingual-clip": "1.0.10",
        "transformers": "4.57.6", "tokenizers": "0.22.2", "sentencepiece": "0.2.1",
        "huggingface-hub": "0.36.0", "ftfy": "6.3.1", "regex": "2025.11.3",
        "tqdm": "4.67.1"}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ZeroMRG-Gate3-source-audit"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"mode": "metadata/source inspection only; no installs or model downloads", "packages": {}}
    for name, version in PINS.items():
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        cached = OUT / f"{name}-{version}.json"
        payload = cached.read_bytes() if cached.exists() else fetch(url)
        if not cached.exists():
            cached.write_bytes(payload)
        data = json.loads(payload)
        result["packages"][name] = dict(version=version, source=url,
            requires_python=data["info"]["requires_python"], requires_dist=data["info"]["requires_dist"],
            files=[{key: item[key] for key in ("filename", "url", "size", "digests", "requires_python")}
                   for item in data["urls"]])
        print(name, version, "metadata verified", flush=True)
    prior_path = ROOT / "docs/evidence/gate3_dependency_audit.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    commit = prior.get("clip_commit") or json.loads(fetch("https://api.github.com/repos/openai/CLIP/commits/main"))["sha"]
    result["clip_commit"] = commit
    for filename in ("clip/clip.py", "requirements.txt", "setup.py"):
        source = f"https://raw.githubusercontent.com/openai/CLIP/{commit}/{filename}"
        cached = OUT / ("clip_" + filename.replace("/", "_"))
        payload = cached.read_bytes() if cached.exists() else fetch(source)
        if not cached.exists():
            cached.write_bytes(payload)
        result.setdefault("clip_files", {})[filename] = dict(url=source, sha256=hashlib.sha256(payload).hexdigest())
    for model in ("M-CLIP/XLM-Roberta-Large-Vit-L-14", "FacebookAI/xlm-roberta-large"):
        revision = prior.get("model_revisions", {}).get(model)
        if revision is None:
            revision = json.loads(fetch("https://huggingface.co/api/models/" + model))["sha"]
        result.setdefault("model_revisions", {})[model] = revision
    destination = ROOT / "docs/evidence/gate3_dependency_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Source revisions:", commit, result["model_revisions"])


if __name__ == "__main__":
    main()
