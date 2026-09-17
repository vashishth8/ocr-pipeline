"""Tests for STTL artifact integrity and routing-contract verification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from verify_ocr_artifact import main as verify_main
from verify_ocr_artifact import verify_artifact

SOURCE_SHA256 = "a" * 64
SOURCE_NAME = "fixture.pdf"
JOB_NAME = "fixture-job"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ArtifactVerificationTests(unittest.TestCase):
    def write_fixture(self, root: Path, *, include_raw_surya: bool = True) -> Path:
        job_dir = root / JOB_NAME
        job_dir.mkdir()
        summary = {
            "pipeline_version": "1.10",
            "source": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "source_page_count": 1,
            "completed_pages": 1,
            "pending_pages": 0,
            "state": "complete",
            "classification_counts": {
                "DIGITAL": 0,
                "SCANNED": 1,
                "MIXED": 0,
                "OCR_NEEDED": 0,
            },
            "outcome_counts": {"surya_primary": 1},
            "native_pages": 0,
            "tesseract_accepted_pages": 0,
            "tesseract_rejected_no_fallback_pages": 0,
            "surya_primary_pages": 1,
            "surya_escalated_pages": 0,
            "surya_quality_escalated_pages": 0,
            "surya_structure_escalated_pages": 0,
            "surya_hybrid_text_pages": 0,
            "mean_tesseract_confidence": None,
            "normalized_output": "fixture_cascade.json",
            "rich_output": "fixture_rich.json",
        }
        write_json(
            job_dir / "job.json",
            {
                "pipeline_version": "1.10",
                "source": {"file_name": SOURCE_NAME, "content_sha256": SOURCE_SHA256},
                "config": {"language": "hin", "ocr_engine": "surya", "fallback_engine": "surya"},
            },
        )
        write_json(job_dir / "document.json", summary)
        manifest_record = {
            "status": "complete",
            "page": 1,
            "classification": "SCANNED",
            "engine": "surya",
            "outcome": "surya_primary",
            "route": "surya",
            "classifier_route": "tesseract",
            "escalation_reason": "primary",
            "surya_batch": 1,
            "native_text": "",
            "text": "भारत सरकार",
            "raster": {"pdf_bbox": [0.0, 0.0, 100.0, 200.0]},
            "blocks": [],
        }
        (job_dir / "pages.jsonl").write_text(
            json.dumps(manifest_record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (job_dir / "combined.txt").write_text("--- Page 1 ---\nभारत सरकार\n", encoding="utf-8")
        routing = {
            "classification_counts": summary["classification_counts"],
            "outcome_counts": summary["outcome_counts"],
            "primary_ocr_engine": "surya",
            "fallback_engine": "surya",
            "tesseract_rejected_no_fallback_pages": 0,
            "surya_primary_pages": 1,
            "surya_quality_escalated_pages": 0,
            "surya_structure_escalated_pages": 0,
            "surya_hybrid_text_pages": 0,
        }
        page = {
            "page": 1,
            "bbox": [0.0, 0.0, 100.0, 200.0],
            "coordinate_space": "pdf_points",
        }
        write_json(
            job_dir / "fixture_cascade.json",
            {
                "schema_version": "cascade-ocr/v1",
                "engine": "cascade",
                "source": SOURCE_NAME,
                "page_count": 1,
                "coordinate_space": "pdf_points",
                "pages": [
                    {
                        **page,
                        "text": "भारत सरकार",
                        "blocks": [],
                        "metadata": {
                            "engine": "surya",
                            "outcome": "surya_primary",
                            "route": "surya",
                            "classifier_route": "tesseract",
                        },
                    }
                ],
                "metadata": {"pipeline_version": "1.10", "routing": routing},
            },
        )
        write_json(
            job_dir / "fixture_rich.json",
            {
                "schema_version": "cascade-ocr/rich-v1",
                "engine": "cascade",
                "source": SOURCE_NAME,
                "page_count": 1,
                "coordinate_space": "pdf_points",
                "pages": [
                    {
                        **page,
                        "authoritative": {
                            "engine": "surya",
                            "outcome": "surya_primary",
                            "text": "भारत सरकार",
                            "blocks": [],
                        },
                        "routing": {"route": "surya", "classifier_route": "tesseract"},
                    }
                ],
                "metadata": {"pipeline_version": "1.10", "routing": routing},
            },
        )
        if include_raw_surya:
            write_json(
                job_dir / "surya" / "batch-000001" / "surya-input-000001" / "results.json",
                {"page-000001": []},
            )
        return job_dir

    def test_valid_complete_surya_primary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = verify_artifact(self.write_fixture(Path(temporary)))

        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["checks"]["effective_summary"]["surya_primary_pages"], 1)

    def test_rejects_rich_artifact_routing_that_disagrees_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = self.write_fixture(Path(temporary))
            rich_path = job_dir / "fixture_rich.json"
            rich = json.loads(rich_path.read_text(encoding="utf-8"))
            rich["pages"][0]["routing"]["route"] = "tesseract"
            write_json(rich_path, rich)
            report = verify_artifact(job_dir)

        self.assertFalse(report["valid"])
        self.assertIn("rich_manifest_mismatch", {entry["code"] for entry in report["errors"]})

    def test_missing_raw_surya_evidence_is_a_strict_error_or_explicit_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = self.write_fixture(Path(temporary), include_raw_surya=False)
            strict_report = verify_artifact(job_dir)
            retained_output_report = verify_artifact(job_dir, require_raw_surya=False)

        self.assertFalse(strict_report["valid"])
        self.assertIn(
            "missing_surya_raw_batch", {entry["code"] for entry in strict_report["errors"]}
        )
        self.assertTrue(retained_output_report["valid"])
        self.assertIn(
            "missing_surya_raw_batch",
            {entry["code"] for entry in retained_output_report["warnings"]},
        )

    def test_cli_writes_a_machine_readable_verification_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = self.write_fixture(root)
            report_path = root / "verification.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["verify_ocr_artifact.py", str(job_dir), "--json-out", str(report_path)],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(verify_main(), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            self.assertEqual(report["schema_version"], "sttl-ocr-artifact-verification/v1")


if __name__ == "__main__":
    unittest.main()
