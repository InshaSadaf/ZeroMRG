"""Paper-aligned report normalization without tokenization or truncation."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping

NORMALIZATION_VERSION = "nfc_whitespace_v1"
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_report_text(text: str) -> str:
    """Apply NFC, trim edges, and collapse whitespace while preserving content."""

    if not isinstance(text, str):
        raise TypeError("report text must be a string")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip())


def normalize_cov_ctr_target(record: Mapping[str, object]) -> str:
    fields = record.get("report_fields")
    if not isinstance(fields, Mapping):
        raise ValueError("COV-CTR record has no report_fields mapping")
    raw = fields.get("reports_En", record.get("report_text"))
    if not isinstance(raw, str):
        raise ValueError("COV-CTR reports_En target is missing")
    normalized = normalize_report_text(raw)
    if not normalized:
        raise ValueError("COV-CTR reports_En target is blank")
    return normalized


def normalize_iu_xray_target(record: Mapping[str, object]) -> str:
    fields = record.get("report_fields")
    if not isinstance(fields, Mapping):
        raise ValueError("IU-Xray record has no report_fields mapping")
    impression = fields.get("impression")
    findings = fields.get("findings")
    if not isinstance(impression, str) or not normalize_report_text(impression):
        raise ValueError("IU-Xray impression is missing or blank")
    if not isinstance(findings, str) or not normalize_report_text(findings):
        raise ValueError("IU-Xray findings are missing or blank")
    return normalize_report_text(impression) + " " + normalize_report_text(findings)

