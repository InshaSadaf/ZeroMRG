"""Static/CPU and read-only remote checks; no training or submission."""
import ast
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from zeromrg_gate1_runtime import sha256, require
from prepare_gate3_training_091_095 import ROOT, TARGET, SOURCE_TARGET, FOLDER


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cells(value):
    return [(row['cell_type'], ''.join(row['source'])) for row in value['cells']]


def local_checks():
    prepared = read(ROOT / 'artifacts/gate3/training_091_095_preparation.json')
    meta = read(FOLDER / 'kernel-metadata.json')
    notebook = read(FOLDER / meta['code_file'])
    base = read(ROOT / 'notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb')
    require(meta['id'] == TARGET and meta['is_private'] is True and meta['enable_gpu'] is True and
        meta['enable_tpu'] is False and meta['kernel_sources'] == [SOURCE_TARGET], 'Wrong private target/attachment')
    require(meta['dataset_sources'] == ['inshasadaf114/dataset', 'inshasadaf114/zeromrg-checkpoint-backup5-zip'],
            'Wrong dataset attachments')
    require(sha256(FOLDER / meta['code_file']) == prepared['notebook_sha256'] and
            sha256(FOLDER / 'kernel-metadata.json') == prepared['metadata_sha256'], 'Prepared source changed')
    for i in (1, 2, 3, 6, 7):
        require(cells(notebook)[i] == cells(base)[i], 'Audited runtime/dependencies/metric changed')
    for i, row in enumerate(notebook['cells']):
        if row['cell_type'] == 'code':
            require(row['outputs'] == [] and row['execution_count'] is None, 'Executed local notebook')
            compile(''.join(row['source']), f'notebook-cell-{i}', 'exec')
    config = {node.targets[0].id: ast.literal_eval(node.value) for node in
              ast.parse(cells(notebook)[4][1]).body if isinstance(node, ast.Assign)}
    require(config['ENABLE_APPROVED_RECOVERY_TRAINING'] is True and config['MAX_NEW_EPOCHS'] == 5 and
            config['AUTHORIZED_START_EPOCH'] == 91 and config['AUTHORIZED_END_EPOCH'] == 95, 'Wrong authorization bounds')
    training = cells(notebook)[10][1]
    function_only = training.split('\nif ENABLE_APPROVED_RECOVERY_TRAINING:')[0]
    guard_cases = 0
    for start, ceiling, count in ((90, 95, 5), (92, 95, 5), (91, 96, 5), (91, 95, 6)):
        namespace = dict(require=require, ENABLE_APPROVED_RECOVERY_TRAINING=True, MAX_NEW_EPOCHS=count,
            AUTHORIZED_START_EPOCH=91, AUTHORIZED_END_EPOCH=ceiling,
            metadata_state=dict(epoch=start-1, original_stop_tracking=dict(epochs_without_improvement=5)),
            recovery_checks=dict(next_epoch=start))
        exec(compile(function_only, 'bounded-guard-fixture', 'exec'), namespace)
        try:
            namespace['run_approved_training_chunk']()
        except ValueError:
            guard_cases += 1
        else:
            raise ValueError('Out-of-bounds fixture was not rejected')
        require('_gate3_training_attempt_started' not in namespace, 'Invalid range advanced to training')
    require(guard_cases == 4, 'Guard tests incomplete')
    require('range(start, min(AUTHORIZED_END_EPOCH, 200, start + MAX_NEW_EPOCHS - 1) + 1)' in training,
            'Loop ceiling missing')
    require("batch_size=6" in training and 'clip_grad_norm_(decoder.parameters(), 5.0' in training and
            'restore_recovery_rng(resume_rng)' in training and 'persist_epoch()' in training, 'Training settings altered')
    return prepared, meta, notebook, guard_cases


def main():
    prepared, meta, notebook, tests = local_checks()
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    from requests.exceptions import HTTPError
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = SOURCE_TARGET.split('/')
        response = client.kernels.kernels_api_client.get_kernel(request)
        source_meta, source_notebook = response.metadata, json.loads(response.blob.source)
        require(source_meta.is_private is True and source_meta.current_version_number == 1,
                'Baseline source privacy/version changed')
        require(cells(source_notebook) == cells(read(ROOT / 'notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb')),
                'Remote baseline source changed')
        output_request = ApiListKernelSessionOutputRequest()
        output_request.user_name, output_request.kernel_slug = SOURCE_TARGET.split('/')
        outputs = client.kernels.kernels_api_client.list_kernel_session_output(output_request)
        names = [row.file_name for row in outputs.files]
        expected = prepared['input_baseline']['zip']['file']
        require(sum(Path(name).name == expected for name in names) == 1, 'Verified baseline export not currently listed')
        request.user_name, request.kernel_slug = TARGET.split('/')
        try:
            client.kernels.kernels_api_client.get_kernel(request)
        except HTTPError as error:
            target_probe_http = error.response.status_code
            require(target_probe_http in (403, 404), 'Unexpected target probe HTTP' + str(target_probe_http))
        else:
            raise ValueError('Training target exists; refusing to overwrite or resubmit')
    # A protected/not-found pull may return403. Do not equate that with absence.
    # Exhaust the authenticated owned-kernel catalog independently, with a bounded search.
    owned = []
    for page in range(1, 11):
        rows = api.kernels_list(page=page, page_size=20, mine=True)
        owned.extend(row.ref for row in rows)
        if len(rows) < 20:
            break
    else:
        raise ValueError('Owned-notebook catalog exceeded bounded200 limit; absence unverified')
    require(SOURCE_TARGET in owned and TARGET not in owned, 'Training target absence/account ownership unverified')
    target_absent = True
    status = str(api.kernels_status(SOURCE_TARGET))
    require('COMPLETE' in status, 'Baseline source run not complete')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'artifacts/gate3' / ('training_091_095_' + stamp)
    directory.mkdir(exist_ok=False)
    for name in (meta['code_file'], 'kernel-metadata.json', 'requirements-bleu.lock'):
        shutil.copy2(FOLDER / name, directory / name)
    record = dict(prepared, status='PASS_PRE_EXECUTION', submitted=False,
        directory=str(directory.relative_to(ROOT)), source_private=True, source_version=1,
        source_status=status, source_output_files=names, new_target_absent=target_absent,
        target_probe_http=target_probe_http, owned_catalog_complete=True, owned_notebooks=owned,
        guard_test_cases_passed=tests, model_optimizer_input_inspection='PASS_CPU_ONLY',
        authorization='One private training execution, epochs91-95 only; no inference/test scoring',
        attachment_policy='Existing private kernel outputs; exact ZIP/latest/best hashes pin identity; no dataset publication',
        historical_starting_patience='5/50', historical_best_epoch=85,
        historical_best_bleu4=56.44423922571333, recovery_starting_best_epoch=90,
        recovery_starting_best_bleu4=54.21140850351637)
    (directory / 'pre_execution_checks.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    with (ROOT / 'artifacts/gate3/active_training_091_095.json').open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: record[k] for k in ('status', 'target', 'directory', 'source_version', 'source_private',
                                          'new_target_absent', 'guard_test_cases_passed')}, indent=2))


if __name__ == '__main__':
    main()
