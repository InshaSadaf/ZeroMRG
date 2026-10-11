"""Retrieve existing session outputs only, with streaming and non-overwriting paths."""
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_gate3_training_091_095 import ROOT, TARGET


def main():
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    import requests
    api = KaggleApi()
    api.authenticate()
    status = str(api.kernels_status(TARGET))
    if not any(term in status for term in ('COMPLETE', 'ERROR', 'CANCELLED', 'FAILED')):
        raise ValueError('Existing run has not ended; no retrieval or resubmission')
    files, token, log = [], None, None
    with api.build_kaggle_client() as client:
        for page in range(20):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = TARGET.split('/')
            if token:
                request.page_token = token
            output = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(output.files)
            log = output.log or log
            token = output.next_page_token
            if not token:
                break
        else:
            raise ValueError('Output list exceeded20-page bound')
    exports = [item for item in files if Path(item.file_name).name.startswith('recovery_export_epoch_') and
               item.file_name.endswith('.zip')]
    final_export = max(exports, key=lambda item: int(Path(item.file_name).name.split('_')[3]), default=None)
    small = {'recovery_readiness_report.json', 'configuration.json', 'recovery_history.json',
             'training_session_report.json', 'requirements-bleu.lock'}
    selected = [item for item in files if Path(item.file_name).name in small or
        Path(item.file_name).name.startswith(('validation_epoch_', 'epoch_completion_'))]
    if final_export:
        selected += [item for item in files if item.file_name == final_export.file_name[:-4]+'.sha256.json']
        selected.append(final_export)
    if not any(Path(item.file_name).name == 'training_session_report.json' for item in files):
        # Preserve possibly completed later states after a failure; never infer95 from initial90 export.
        selected += [item for item in files if Path(item.file_name).name == 'recovery_latest.pt' or
                     Path(item.file_name).name.startswith('recovery_best_epoch_') and item.file_name.endswith('.pt')]
    require_names = [item.file_name for item in selected]
    if len(require_names) != len(set(require_names)):
        raise ValueError('Duplicate selected output paths')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'artifacts/gate3' / ('training_091_095_retrieval_' + stamp)
    directory.mkdir(exist_ok=False)
    record = dict(target=TARGET, version=1, terminal_status=status, directory=str(directory.relative_to(ROOT)),
        listed_files=[item.file_name for item in files], selected_files=require_names,
        downloads=[], complete=False, export_present=bool(final_export), mode='read-only existing output retrieval')
    index = directory / 'retrieval_index.json'
    index.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    if log:
        (directory / 'execution.log').write_text(log, encoding='utf-8')
    (ROOT / 'artifacts/gate3/active_training_091_095_retrieval.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Retrieval directory:', directory, flush=True)
    print('Terminal status:', status, '| files:', len(files), '| selected:', len(selected), flush=True)
    total = 0
    for item in selected:
        path = (directory / item.file_name).resolve()
        if not path.is_relative_to(directory.resolve()) or path.exists():
            raise ValueError('Unsafe/existing output path')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.part')
        count, digest, started, last_print = 0, hashlib.sha256(), time.monotonic(), time.monotonic()
        # Signed URLs are never emitted or saved.
        with requests.get(item.url, stream=True, timeout=(30, 90)) as response:
            if response.status_code != 200:
                raise ValueError('Download HTTP'+str(response.status_code)+': '+path.name)
            length = int(response.headers.get('content-length', '0'))
            if total+length > 5*2**30 or shutil.disk_usage(directory).free < length+2**30:
                raise ValueError('Bounded size/free space failed')
            with partial.open('xb') as handle:
                for chunk in response.iter_content(chunk_size=4*2**20):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    count += len(chunk)
                    digest.update(chunk)
                    if total+count > 5*2**30:
                        raise ValueError('5GiB cumulative download ceiling exceeded')
                    if time.monotonic()-last_print > 25:
                        print(path.name, 'downloaded MiB:', round(count/2**20, 1), flush=True)
                        last_print = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
        if length and length != count:
            raise ValueError('Truncated download: '+path.name)
        os.replace(partial, path)
        total += count
        record['downloads'].append(dict(file=item.file_name, bytes=count, sha256=digest.hexdigest(),
                                        elapsed_seconds=time.monotonic()-started))
        index.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('Retrieved:', path.name, '| bytes:', count, flush=True)
    record.update(complete=True, total_bytes=total)
    index.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'artifacts/gate3/active_training_091_095_retrieval.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Retrieval complete; checkpoint verification still required.', flush=True)


if __name__ == '__main__':
    main()
