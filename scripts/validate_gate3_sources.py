"""Static notebook/source safety verification; never execute notebook cells."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    folder = ROOT / 'notebooks/gate3'
    path = folder / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb'
    notebook = json.loads(path.read_text())
    cells = notebook['cells']
    check(notebook['nbformat'] == 4 and len({c['id'] for c in cells}) == len(cells), 'Notebook IDs/format differ')
    trees = []
    for index, c in enumerate(cells):
        if c['cell_type'] == 'code':
            source = ''.join(c['source'])
            compile(source, f'gate3-cell-{index}', 'exec')
            trees.append(ast.parse(source))
            check(c['execution_count'] is None and c['outputs'] == [], 'Notebook has execution output')
    runtime = (ROOT / 'scripts/zeromrg_gate1_runtime.py').read_text()
    check(''.join(cells[1]['source']).strip() == runtime.strip(), 'Gate1 runtime drift')
    recovery = (ROOT / 'scripts/zeromrg_gate3_runtime.py').read_text()
    imported = next(n for n in ast.parse(recovery).body if isinstance(n, ast.ImportFrom) and n.module == 'zeromrg_gate1_runtime')
    lines = recovery.splitlines(keepends=True)
    embedded = ''.join(lines[:imported.lineno-1] + lines[imported.end_lineno:])
    check(''.join(cells[2]['source']).strip() == embedded.strip(), 'Gate3 runtime drift')
    pins = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(''.join(cells[3]['source'])).body}
    check(pins['RUNTIME_SOURCE'] == runtime and pins['RECOVERY_RUNTIME_SOURCE'] == recovery, 'Bundled source differs')
    check(pins['RUNTIME_HASHES'] == {'gate1': hashlib.sha256(runtime.encode()).hexdigest(),
                                   'gate3': hashlib.sha256(recovery.encode()).hexdigest()}, 'Runtime digest differs')
    check((folder / 'requirements-bleu.lock').read_text() == pins['METRIC_LOCK'], 'Metric lock drift')
    metric = json.loads((ROOT / 'docs/evidence/gate3_metric_implementation.json').read_text())
    check(pins['METRIC_IMPLEMENTATION'] == metric['implementation_sha256'], 'Metric implementation drift')
    check(pins['METRIC_WHEEL_SHA256'] == metric['wheel']['digests']['sha256'], 'Metric wheel drift')
    for name, text in pins['HISTORICAL_REPORTS'].items():
        check(hashlib.sha256(text.encode()).hexdigest() == pins['HISTORICAL_REPORT_HASHES'][name], 'Evidence digest differs')
        source = ROOT / 'docs' / name
        if not source.exists():
            source = ROOT / 'docs/evidence' / name
        check(source.read_bytes() == text.encode(), 'Historical embedded report changed')
    configuration = ast.parse(''.join(cells[4]['source']))
    flag = next(n.value for n in configuration.body if isinstance(n, ast.Assign) and
                n.targets[0].id == 'ENABLE_APPROVED_RECOVERY_TRAINING')
    check(ast.literal_eval(flag) is False, 'Training enabled by default')
    source = ''.join(cells[-1]['source'])
    training = ast.parse(source)
    definition = training.body[0]
    check(isinstance(definition, ast.FunctionDef) and definition.name == 'run_approved_training_chunk', 'Unexpected training entry')
    gate = training.body[1]
    check(isinstance(gate, ast.If) and isinstance(gate.test, ast.Name) and
          gate.test.id == 'ENABLE_APPROVED_RECOVERY_TRAINING', 'Training entry is not guarded')
    for tree in trees[:-1]:
        check(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                      n.func.attr in {'backward', 'step', 'zero_grad'} for n in ast.walk(tree)), 'Updates outside guarded training')
    whole = '\n'.join(''.join(c['source']) for c in cells)
    check("records['test']" not in whole and "feature_batch(rows, 'test')" not in whole and
          'import clip' not in whole and 'multilingual_clip' not in whole, 'Unexpected test/image inference path')
    check('max_length=160' in whole and "records['validation']" in whole and 'measure_validation(90' in whole,
          'Baseline/validation protocol missing')
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    check(metadata['is_private'] is True and metadata['enable_gpu'] is True and metadata['enable_tpu'] is False,
          'Privacy/accelerator metadata differs')
    check(metadata['id'] == 'inshasadaf114/zeromrg-covctr-gate3-recovery' and metadata['code_file'] == path.name,
          'Wrong/new kernel target')
    check(metadata['dataset_sources'] == ['inshasadaf114/dataset', 'inshasadaf114/zeromrg-checkpoint-backup5-zip'],
          'Dataset attachments changed')
    old = {'preflight/ZeroMRG_COV_CTR_Preflight_Only_Kaggle.ipynb': '3d9731b0e36bf0aa0ab591f3dfa370aabeb6512f0670d6201c141f0ca7b54991',
           'continuation/ZeroMRG_COV_CTR_Checkpoint_Continuation_Kaggle.ipynb': 'f4ffaf4872be61095f4888bd8db4c53c05b138c89620f2affd542bb16bbc58fb'}
    for name, digest in old.items():
        check(sha(ROOT / 'notebooks/gate1' / name) == digest, 'Audited Gate1 notebook changed')
    result = dict(status='PASS_STATIC_ONLY', cells=len(cells), notebook_sha256=sha(path),
                  metadata_sha256=sha(folder / 'kernel-metadata.json'), runtime_hashes=pins['RUNTIME_HASHES'],
                  training_default=False, baseline='UNMEASURED', remote_execution=False,
                  gate1_notebooks_unchanged=True, inference_runtime='UNVERIFIED')
    (ROOT / 'docs/evidence/gate3_notebook_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    validate()
