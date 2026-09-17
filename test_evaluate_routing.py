"""Tests for reviewed-text routing/engine evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evaluate_routing import evaluate_routing
from evaluate_routing import main as evaluate_routing_main

SOURCE_SHA256 = "a" * 64


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rich_page(
    page: int,
    *,
    engine: str,
    route: str,
    outcome: str,
    text: str,
    classifier_route: str | None = None,
) -> dict:
    routing = {"route": route}
    if classifier_route is not None:
        routing["classifier_route"] = classifier_route
    return {
        "page": page,
        "authoritative": {
            "engine": engine,
            "outcome": outcome,
            "text": text,
            "blocks": [],
        },
        "routing": routing,
    }


class RoutingEvaluationTests(unittest.TestCase):
    def write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        job_path = root / "job.json"
        rich_path = root / "report_rich.json"
        truth_path = root / "truth.json"
        write_json(job_path, {"source": {"content_sha256": SOURCE_SHA256}})
        write_json(
            rich_path,
            {
                "schema_version": "cascade-ocr/rich-v1",
                "metadata": {"pipeline_version": "1.10"},
                "pages": [
                    rich_page(
                        1,
                        engine="surya",
                        route="surya",
                        classifier_route="tesseract",
                        outcome="surya_primary",
                        text="भारत सरकार",
                    ),
                    rich_page(
                        2,
                        engine="tesseract5",
                        route="tesseract",
                        outcome="tesseract_accepted",
                        text="wrong OCR",
                    ),
                    rich_page(
                        3,
                        engine="pymupdf",
                        route="native_text",
                        outcome="native_text_accepted",
                        text="not reviewed",
                    ),
                ],
            },
        )
        write_json(
            truth_path,
            {
                "schema_version": "sttl-human-transcription/v1",
                "document_id": "fixture",
                "source_sha256": SOURCE_SHA256,
                "pages": [
                    {
                        "page": 1,
                        "text": "भारत सरकार",
                        "review": {"status": "adjudicated", "auto_accept": True},
                    },
                    {
                        "page": 2,
                        "text": "correct OCR",
                        "review": {"status": "adjudicated", "auto_accept": False},
                    },
                ],
            },
        )
        return rich_path, truth_path, job_path

    def test_scores_only_reviewed_pages_and_stratifies_authoritative_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rich_path, truth_path, job_path = self.write_fixture(Path(temporary))
            report = evaluate_routing(rich_path, truth_path, job_path=job_path)

        self.assertEqual(report["coverage"]["rich_pages_available"], 3)
        self.assertEqual(report["coverage"]["pages_evaluated"], 2)
        self.assertEqual(report["coverage"]["unreviewed_rich_pages"], [3])
        self.assertEqual(report["evaluation"]["evaluated_page_ids"], [1, 2])
        self.assertEqual(report["by_engine"]["surya"]["pages"], 1)
        self.assertEqual(report["by_engine"]["surya"]["weighted_CER"], 0.0)
        self.assertEqual(report["by_route"]["surya"]["pages"], 1)
        self.assertEqual(report["by_outcome"]["surya_primary"]["pages"], 1)
        self.assertEqual(report["by_engine_route"]["surya / surya"]["pages"], 1)
        row = report["per_page"][0]
        self.assertEqual(row["route"], "surya")
        self.assertEqual(row["classifier_route"], "tesseract")

    def test_rejects_truth_bound_to_a_different_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rich_path, truth_path, job_path = self.write_fixture(Path(temporary))
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            truth["source_sha256"] = "b" * 64
            write_json(truth_path, truth)
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluate_routing(rich_path, truth_path, job_path=job_path)

    def test_rejects_reviewed_page_absent_from_rich_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rich_path, truth_path, job_path = self.write_fixture(Path(temporary))
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            truth["pages"].append(
                {
                    "page": 4,
                    "text": "missing",
                    "review": {"status": "adjudicated", "auto_accept": False},
                }
            )
            write_json(truth_path, truth)
            with self.assertRaisesRegex(ValueError, "pages absent"):
                evaluate_routing(rich_path, truth_path, job_path=job_path)

    def test_cli_writes_machine_readable_report_and_per_page_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rich_path, truth_path, job_path = self.write_fixture(root)
            output_dir = root / "evaluation"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_routing.py",
                        "--rich-json",
                        str(rich_path),
                        "--human-truth",
                        str(truth_path),
                        "--job-json",
                        str(job_path),
                        "--out-dir",
                        str(output_dir),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(evaluate_routing_main(), 0)

            report = json.loads(
                (output_dir / "routing_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["aggregate"]["pages"], 2)
            self.assertEqual(report["by_engine"]["surya"]["pages"], 1)
            self.assertTrue((output_dir / "routing_per_page.csv").is_file())


if __name__ == "__main__":
    unittest.main()
