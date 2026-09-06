from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.utils.config import load_config
from zeromrg.utils.provenance import create_provenance, detect_environment, save_provenance


class ProvenanceTests(unittest.TestCase):
    def test_environment_detection(self) -> None:
        never_exists = lambda _: False
        self.assertEqual(
            detect_environment({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, never_exists),
            "KAGGLE",
        )
        self.assertEqual(
            detect_environment({"ZEROMRG_ENVIRONMENT": "LOCAL"}, never_exists), "LOCAL"
        )
        self.assertEqual(
            detect_environment({"ZEROMRG_ENVIRONMENT": "UNKNOWN"}, never_exists), "UNKNOWN"
        )
        self.assertEqual(detect_environment({}, never_exists), "LOCAL")

    def test_provenance_is_json_serializable_and_saveable(self) -> None:
        config = load_config(ROOT / "configs" / "iu_xray.yaml")
        record = create_provenance(
            config,
            execution_environment="LOCAL",
            repository=ROOT,
        )
        self.assertEqual(record["random_seed"], 42)
        self.assertEqual(record["dataset_name"], "iu_xray")
        self.assertEqual(record["execution_environment"], "LOCAL")
        self.assertEqual(record["config_hash"], config.digest())
        self.assertIn("gpu_name", record)
        self.assertIn("gpu_vram_bytes", record)
        json.dumps(record)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run" / "provenance.json"
            save_provenance(record, destination)
            loaded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(loaded["config_hash"], config.digest())


if __name__ == "__main__":
    unittest.main()
