"""Read-only selective recovery of existing Kaggle outputs; never submit a run."""
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'inshasadaf114/zeromrg-covctr-gate3-recovery-baseline'
MAX_BYTES = 5 * 2**30


def main():
    # Use the exact authenticated configuration used by scripts/kaggle.ps1.
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    import requests
    api = KaggleApi()
    api.authenticate()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'artifacts/gate3' / ('baseline_retrieval_' + stamp)
    directory.mkdir(exist_ok=False)
    record = dict(target=TARGET, directory=str(directory.relative_to(ROOT)),
                  mode='read-only output listing/download; no submission', listed_files=[], downloads=[], complete=False)
    (ROOT / 'artifacts/gate3/active_baseline_retrieval.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    entries, token, log = [], None, None
    with api.build_kaggle_client() as client:
        for page in range(20):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = TARGET.split('/')
            if token:
                if not hasattr(request, 'page_token'):
                    raise ValueError('Output pagination unsupported; do not claim complete listing')
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            entries.extend(response.files)
            log = response.log or log
            token = response.next_page_token
            if not token:
                break
        else:
            raise ValueError('Bounded20-page output limit reached')
    record['listed_files'] = [entry.file_name for entry in entries]
    if log:
        log_path = directory / 'execution.log'
        with log_path.open('x', encoding='utf-8') as handle:
            handle.write(log)
        record['log_sha256'] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    small = {'recovery_readiness_report.json', 'validation_epoch_090.json', 'configuration.json',
             'recovery_history.json', 'requirements-bleu.lock'}
    exports = [entry for entry in entries if Path(entry.file_name).name.startswith('recovery_export_epoch_090_')
               and entry.file_name.endswith('.zip')]
    selected = [entry for entry in entries if Path(entry.file_name).name in small or
                (Path(entry.file_name).name.startswith('recovery_export_epoch_090_') and entry.file_name.endswith('.sha256.json'))]
    selected += exports
    if not exports:
        selected += [entry for entry in entries if Path(entry.file_name).name in
                     {'recovery_latest.pt', 'recovery_best_epoch_090.pt'}]
    names = [entry.file_name for entry in selected]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate output paths; refuse overwrite')
    record['selected_files'] = names
    index_path = directory / 'retrieval_index.json'
    index_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Evidence directory:', directory, flush=True)
    print('Listed', len(entries), 'outputs; selected', len(selected), '; export ZIPs:', len(exports), flush=True)
    total = 0
    for entry in selected:
        path = (directory / entry.file_name).resolve()
        if not path.is_relative_to(directory.resolve()) or path.exists():
            raise ValueError('Unsafe or existing output path; refuse overwrite')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        started = time.monotonic()
        digest, count, last_print = hashlib.sha256(), 0, started
        # Signed URLs are intentionally neither printed nor saved.
        with requests.get(entry.url, stream=True, timeout=(30, 90)) as response:
            if response.status_code != 200:
                raise ValueError(f'Output download HTTP{response.status_code}: {path.name}')
            length = int(response.headers.get('content-length', '0'))
            if length + total > MAX_BYTES or shutil.disk_usage(directory).free < length + 2**30:
                raise ValueError('Download size/free-space bound exceeded')
            with partial.open('xb') as handle:
                for chunk in response.iter_content(chunk_size=4 * 2**20):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if total + count > MAX_BYTES:
                        raise ValueError('Cumulative5GiB retrieval bound exceeded')
                    if time.monotonic() - last_print > 25:
                        print(path.name, 'downloaded MiB:', round(count / 2**20, 1), flush=True)
                        last_print = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
        if length and count != length:
            raise ValueError('Truncated output download: ' + path.name)
        os.replace(partial, path)
        total += count
        record['downloads'].append(dict(file=entry.file_name, bytes=count, sha256=digest.hexdigest(),
                                        elapsed_seconds=time.monotonic()-started))
        index_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('Retrieved:', entry.file_name, '| bytes:', count, flush=True)
    record.update(complete=True, total_bytes=total, export_present=bool(exports))
    index_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'artifacts/gate3/active_baseline_retrieval.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Existing outputs retrieved. Integrity/model verification still required.', flush=True)


if __name__ == '__main__':
    main()
