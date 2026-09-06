from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.data.report_text import (
    normalize_cov_ctr_target,
    normalize_iu_xray_target,
    normalize_report_text,
)


class ReportTextTests(unittest.TestCase):
    def test_nfc_whitespace_case_and_punctuation(self) -> None:
        text = "  Cafe\u0301\n\tLung opacity, unchanged.  "
        self.assertEqual(normalize_report_text(text), "Café Lung opacity, unchanged.")

    def test_cov_target_uses_english_report_only(self) -> None:
        record = {
            "report_fields": {
                "reports_En": "  English\nreport. ",
                "impression": "不应附加",
            }
        }
        self.assertEqual(normalize_cov_ctr_target(record), "English report.")

    def test_iu_order_and_redaction_preservation(self) -> None:
        record = {
            "report_fields": {
                "impression": "  Normal XXXX. ",
                "findings": " Lungs\nare clear. ",
                "indication": "must not be appended",
            }
        }
        self.assertEqual(
            normalize_iu_xray_target(record), "Normal XXXX. Lungs are clear."
        )

    def test_iu_requires_both_sections(self) -> None:
        with self.assertRaises(ValueError):
            normalize_iu_xray_target(
                {"report_fields": {"impression": "Normal.", "findings": "  "}}
            )


if __name__ == "__main__":
    unittest.main()

