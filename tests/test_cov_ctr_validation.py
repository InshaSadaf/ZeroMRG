from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_utils import write_png
from zeromrg.data.schemas import ConflictSample, ExcludedSample
from zeromrg.data.validate_cov_ctr import validate_cov_ctr


class CovCtrValidationTests(unittest.TestCase):
    def test_join_conflicts_and_missing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png(root / "a.png", (1, 2, 3))
            write_png(root / "b.png", (4, 5, 6))
            write_png(root / "orphan.png", (7, 8, 9))
            columns = ["image_id", "findings", "terminologies", "COVID", "impression", "reports_En"]
            rows = [
                ["a.png", "发现", "术语", "0", "印象", "valid English report"],
                ["b.png", "发现", "术语", "1", "印象", "first report"],
                ["b.png", "不同", "术语", "1", "印象", "second report"],
                ["missing.png", "发现", "术语", "0", "印象", "missing image report"],
            ]
            with (root / "reports_ZH_EN.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows)

            result = validate_cov_ctr(str(root))
            counts = result.summary["counts"]
            self.assertEqual(counts["valid_samples"], 1)
            self.assertEqual(counts["conflicting_duplicate_keys"], 1)
            self.assertEqual(counts["images_without_reports"], 1)
            self.assertEqual(counts["reports_without_images"], 1)
            self.assertEqual(result.valid_samples[0].image_ids, ["a.png"])
            self.assertTrue(any(isinstance(record, ConflictSample) for record in result.audit_records))
            reasons = [record.reason for record in result.audit_records if isinstance(record, ExcludedSample)]
            self.assertIn("image_without_report", reasons)
            self.assertIn("report_without_image", reasons)


if __name__ == "__main__":
    unittest.main()

