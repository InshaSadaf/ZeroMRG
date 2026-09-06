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
from zeromrg.data.validate_iu_xray import validate_iu_xray


class IuXrayValidationTests(unittest.TestCase):
    def test_uid_grouping_sections_and_projection_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images" / "images_normalized"
            for index in range(1, 6):
                write_png(image_root / f"image{index}.dcm.png", (index, index, index))
            write_png(image_root / "orphan.dcm.png", (20, 20, 20))

            report_columns = ["uid", "MeSH", "Problems", "image", "indication", "comparison", "findings", "impression"]
            report_rows = [
                ["1", "normal", "normal", "exam", "", "", "findings 1", "impression 1"],
                ["2", "normal", "normal", "exam", "", "", "findings 2", ""],
                ["3", "normal", "normal", "exam", "", "", "", "impression 3"],
                ["4", "normal", "normal", "exam", "", "", "", ""],
                ["5", "normal", "normal", "exam", "", "", "findings 5", "impression 5"],
            ]
            with (root / "indiana_reports.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(report_columns)
                writer.writerows(report_rows)

            projection_rows = [
                ["1", "image1.dcm.png", "Frontal"],
                ["1", "image2.dcm.png", "Lateral"],
                ["2", "image3.dcm.png", "Frontal"],
                ["3", "image4.dcm.png", "Frontal"],
                ["4", "image5.dcm.png", "Frontal"],
                ["5", "missing.dcm.png", "Frontal"],
            ]
            with (root / "indiana_projections.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["uid", "filename", "projection"])
                writer.writerows(projection_rows)

            result = validate_iu_xray(str(root))
            counts = result.summary["counts"]
            self.assertEqual(counts["valid_uids"], 1)
            self.assertEqual(counts["both_sections_uids"], 2)
            self.assertEqual(counts["findings_only_uids"], 1)
            self.assertEqual(counts["impression_only_uids"], 1)
            self.assertEqual(counts["neither_section_uids"], 1)
            self.assertEqual(counts["projection_rows_missing_images"], 1)
            self.assertEqual(counts["images_missing_projection_metadata"], 1)
            self.assertEqual(result.valid_studies[0].study_id, "1")
            self.assertEqual(result.valid_studies[0].image_ids, ["image1.dcm.png", "image2.dcm.png"])
            self.assertEqual(result.summary["mapping"]["views_per_uid"], {"1": 4, "2": 1})


if __name__ == "__main__":
    unittest.main()

