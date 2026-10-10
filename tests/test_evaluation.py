from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.evaluation import diagnostic_caption_metrics, write_experiment_graphs


class EvaluationTests(unittest.TestCase):
    def test_diagnostic_metrics_identical_reports(self) -> None:
        metrics = diagnostic_caption_metrics(["No acute disease ."], ["No acute disease ."])
        self.assertEqual(metrics["BLEU-4"], 100.0)
        self.assertEqual(metrics["ROUGE-L"], 100.0)
        self.assertEqual(metrics["exact_match_percent"], 100.0)
        self.assertIn("not_paper_comparable", metrics["metric_backend"])

    def test_svg_graph_bundle_is_written(self) -> None:
        histories = {"text": [{"epoch": 1, "train_loss": 2.0, "validation_bleu4": 1.0}]}
        metrics = {"zero": {"BLEU-1": 2.0, "BLEU-2": 1.0, "BLEU-3": 0.5, "BLEU-4": 0.25, "ROUGE-L": 3.0}}
        with tempfile.TemporaryDirectory() as directory:
            files = write_experiment_graphs(Path(directory), histories, metrics)
            self.assertEqual(len(files), 3)
            for path in files:
                self.assertTrue(path.is_file())
                self.assertIn("<svg", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
