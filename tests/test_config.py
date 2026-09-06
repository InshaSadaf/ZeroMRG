from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.utils.config import ConfigError, load_config, resolve_paths


class ConfigTests(unittest.TestCase):
    def test_dataset_and_environment_composition(self) -> None:
        config = load_config(
            [ROOT / "configs" / "cov_ctr.yaml", ROOT / "configs" / "kaggle.yaml"],
            overrides={"paths.datasets.cov_ctr": "/kaggle/input/example-cov"},
        )
        self.assertEqual(config["dataset"]["name"], "cov_ctr")
        self.assertEqual(config["dataset"]["prompt_count"], 500)
        self.assertEqual(config["paths"]["datasets"]["cov_ctr"], "/kaggle/input/example-cov")
        self.assertTrue(config["paths"]["outputs"]["results"].startswith("/kaggle/working/"))
        self.assertEqual(config.label_for("dataset.prompt_count"), "PAPER-STATED")
        self.assertEqual(
            config.label_for("paths.datasets.cov_ctr"), "IMPLEMENTATION-ASSUMPTION"
        )

    def test_local_paths_resolve_without_drive_literals(self) -> None:
        config = load_config(
            [ROOT / "configs" / "base.yaml", ROOT / "configs" / "local.yaml"],
            overrides={"paths.datasets.iu_xray": "alternate/iu.zip"},
        )
        resolved = resolve_paths(config, ROOT)
        self.assertTrue(Path(resolved["datasets"]["iu_xray"]).is_absolute())
        self.assertTrue(resolved["datasets"]["iu_xray"].endswith(str(Path("alternate/iu.zip"))))
        self.assertEqual(config["reproducibility"]["seed"], 42)

    def test_hash_is_stable_and_changes_with_override(self) -> None:
        path = ROOT / "configs" / "iu_xray.yaml"
        first = load_config(path)
        second = load_config(path)
        changed = load_config(path, overrides={"reproducibility.seed": 43})
        self.assertEqual(first.digest(), second.digest())
        self.assertNotEqual(first.digest(), changed.digest())

    def test_unlabelled_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("value: 1\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
