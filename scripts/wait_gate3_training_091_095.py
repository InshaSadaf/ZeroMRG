"""Wait on the already submitted job using status reads only; never resubmit."""
import json
import os
import time
from datetime import datetime, timezone

from prepare_gate3_training_091_095 import ROOT, TARGET


def main():
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    record = json.loads((ROOT / 'artifacts/gate3/active_training_091_095.json').read_text())
    directory = ROOT / record['directory']
    start = time.monotonic()
    while time.monotonic()-start < 7800:
        status = str(api.kernels_status(TARGET))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (directory / ('wait_status_' + stamp + '.json')).write_text(json.dumps(
            dict(observed_at_utc=stamp, target=TARGET, status=status), indent=2)+'\n', encoding='utf-8')
        print(stamp, status, flush=True)
        if any(term in status for term in ('COMPLETE', 'ERROR', 'CANCELLED', 'FAILED')):
            return
        time.sleep(45)  # Bounded short polling sleep; root remains available for updates.
    raise RuntimeError('Status monitoring bound reached; job status unresolved. Do not submit another run.')


if __name__ == '__main__':
    main()
