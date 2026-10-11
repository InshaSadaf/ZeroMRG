"""Baseline execution safety record; never execute notebook cells or submit remotely."""
import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'notebooks/gate3'
    notebook_path = folder / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    assignments = []
    whole = ''
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        whole += source
        assert cell['outputs'] == [] and cell['execution_count'] is None
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'CHECKPOINT_PATH', 'ENABLE_APPROVED_RECOVERY_TRAINING'}:
                        assignments.append([index, target.id, ast.unparse(node.value)])
    assert [row for row in assignments if row[1] == 'ENABLE_APPROVED_RECOVERY_TRAINING'] == [[4, 'ENABLE_APPROVED_RECOVERY_TRAINING', 'False']]
    assert [row for row in assignments if row[1] == 'CHECKPOINT_PATH'] == [[5, 'CHECKPOINT_PATH',
        "Path('/kaggle/input/datasets/inshasadaf114/zeromrg-checkpoint-backup5-zip/text_decoder_last.pt')"]]
    assert all(text in whole for text in ("records['validation']", 'range(0, 71, 6)', 'PTBTokenizer()',
        'Bleu(4).compute_score', '100.0 * float', 'torch.no_grad()', 'measure_validation(90',
        "'Legacy90 initialization is baseline-only; training requires a retrieved and reviewed Gate3 checkpoint'"))
    historical_path = ROOT / 'artifacts/reconciliation/gate1_20261010_remote/zeromrg-workflow.ipynb'
    historical = json.loads(historical_path.read_text(encoding='utf-8'))
    historical_source = '\n'.join(''.join(c['source']) for c in historical['cells'])
    assert 'PTBTokenizer' in historical_source and 'Bleu(4)' in historical_source
    metric_excerpt = '\n\n'.join(''.join(c['source']) for c in historical['cells']
                                  if 'PTBTokenizer' in ''.join(c['source']) or 'Bleu(4)' in ''.join(c['source']))
    metadata = json.loads((folder / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert metadata['is_private'] is True and metadata['dataset_sources'] == [
        'inshasadaf114/dataset', 'inshasadaf114/zeromrg-checkpoint-backup5-zip']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'artifacts/gate3' / ('baseline_' + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    for filename in (notebook_path.name, 'kernel-metadata.json', 'requirements-bleu.lock'):
        shutil.copy2(folder / filename, directory / filename)
    (directory / 'historical_metric_excerpt.txt').write_text(metric_excerpt, encoding='utf-8')
    record = dict(directory=str(directory.relative_to(ROOT)), target=metadata['id'], prepared_private=True,
        datasets=metadata['dataset_sources'], training=False, checkpoint_assignments=assignments,
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256((folder / 'kernel-metadata.json').read_bytes()).hexdigest(),
        historical_source_sha256=hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        metric='Historical PTB + corpus Bleu(4) x100; exact historical package/Java versions unavailable',
        timeout_seconds=1800, submitted=False, authorization='one private baseline-only execution')
    (directory / 'pre_submission_checks.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'artifacts/gate3/active_baseline_record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


def verify_server(directory):
    directory = Path(directory)
    record = json.loads((directory / 'pre_submission_checks.json').read_text(encoding='utf-8'))
    local = json.loads((directory / 'ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb').read_text(encoding='utf-8'))
    server_folder = directory / 'server_source'
    server_metadata = json.loads((server_folder / 'kernel-metadata.json').read_text(encoding='utf-8'))
    server_path = server_folder / server_metadata['code_file']
    server = json.loads(server_path.read_text(encoding='utf-8'))
    assert len(local['cells']) == len(server['cells'])
    assert [(c['cell_type'], ''.join(c['source'])) for c in local['cells']] == [
        (c['cell_type'], ''.join(c['source'])) for c in server['cells']]
    assert server_metadata['is_private'] is True
    assert server_metadata['dataset_sources'] == record['datasets']
    assert server_metadata['enable_gpu'] is True and server_metadata['enable_tpu'] is False
    record.update(submitted=True, version=1, actual_target=server_metadata['id'], id_no=server_metadata['id_no'],
        server_private=True, server_datasets=server_metadata['dataset_sources'], server_source_matches=True,
        server_source_sha256=hashlib.sha256(server_path.read_bytes()).hexdigest(),
        target_slug_warning=server_metadata['id'] != record['target'])
    (directory / 'submission_verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'artifacts/gate3/active_baseline_record.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        verify_server(sys.argv[1])
    else:
        main()
