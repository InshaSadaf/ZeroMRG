"""Read-only source/privacy/status/log monitoring of the one submitted session."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from check_gate3_training_091_095 import ROOT, TARGET, local_checks, cells
from zeromrg_gate1_runtime import require, sha256


def main():
    active = ROOT / 'artifacts/gate3/active_training_091_095.json'
    record = json.loads(active.read_text())
    directory = ROOT / record['directory']
    _, meta, notebook, _ = local_checks()
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = TARGET.split('/')
        response = client.kernels.kernels_api_client.get_kernel(request)
        server, blob = response.metadata, response.blob
        require(server.is_private is True and server.current_version_number == 1 and server.ref == TARGET,
                'Submitted target/privacy/version differs')
        require(server.dataset_data_sources == meta['dataset_sources'] and server.kernel_data_sources == meta['kernel_sources'],
                'Submitted attachments differ')
        require(cells(json.loads(blob.source)) == cells(notebook), 'Server notebook source differs')
        server_dir = directory / 'server_source'
        if not server_dir.exists():
            server_dir.mkdir(exist_ok=False)
            (server_dir / 'notebook.ipynb').write_text(blob.source, encoding='utf-8')
            (server_dir / 'metadata.json').write_text(json.dumps(dict(id=server.ref, id_no=server.id,
                version=server.current_version_number, is_private=server.is_private,
                dataset_sources=server.dataset_data_sources, kernel_sources=server.kernel_data_sources,
                enable_gpu=server.enable_gpu), indent=2)+'\n', encoding='utf-8')
        record.update(submitted=True, version=1, id_no=server.id, server_private=True,
            server_source_matches=True, server_notebook_sha256=sha256(server_dir / 'notebook.ipynb'))
        if not (directory / 'submission_acknowledgement.txt').exists():
            (directory / 'submission_acknowledgement.txt').write_text(
                'Kernel version1 successfully pushed once. Original wrapper exit0.\n'
                'https://www.kaggle.com/code/' + TARGET + '\n', encoding='utf-8')
        status = api.kernels_status(TARGET)
        output_request = ApiListKernelSessionOutputRequest()
        output_request.user_name, output_request.kernel_slug = TARGET.split('/')
        outputs = client.kernels.kernels_api_client.list_kernel_session_output(output_request)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot = dict(observed_at_utc=stamp, target=TARGET, status=str(status),
                    private=True, version=1, listed_outputs=[item.file_name for item in outputs.files])
    (directory / ('status_' + stamp + '.json')).write_text(json.dumps(snapshot, indent=2)+'\n', encoding='utf-8')
    log_text = outputs.log or ''
    if log_text:
        (directory / ('log_' + stamp + '.json')).write_text(log_text, encoding='utf-8')
        log = json.loads(log_text)
        important = [row['data'].strip() for row in log if any(term in row['data'] for term in
            ('EPOCH_COMPLETED', 'Traceback', 'ValueError', 'RuntimeError', 'Recovery state verified',
             'AUTHORIZED_TRAINING_SESSION_COMPLETE', 'Approved private baseline'))]
        for line in important[-8:]:
            print(line)
    record['latest_status'] = snapshot['status']
    record['latest_observation_utc'] = stamp
    active.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=str(status), private=True, version=1, output_count=len(outputs.files)), indent=2))


if __name__ == '__main__':
    main()
