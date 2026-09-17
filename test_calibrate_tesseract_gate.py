import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import calibrate_tesseract_gate as calibration

SAFE_TEXT = "यह एक साफ और सही परीक्षण पृष्ठ है"
UNSAFE_REFERENCE = "यह सत्यापित पाठ पूरी तरह अलग और महत्वपूर्ण है"
UNSAFE_OCR = "गलत लेकिन भरोसेमंद दिखने वाला ओसीआर पाठ यहाँ है"


def write_json(path: Path, value: dict) -> str:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def tesseract_layer(text: str, confidence: float, attempted: bool = True) -> dict:
    return {
        "attempted": attempted,
        "text": text,
        "words": [{"text": word, "confidence": confidence} for word in text.split()],
    }


def make_corpus(
    root: Path,
    *,
    attempted: bool = True,
    source_sha256: str = "a" * 64,
    truth_source_sha256: str | None = None,
    bind_evidence: bool = True,
    include_second_truth_page: bool = True,
) -> Path:
    run = root / "run"
    truth_dir = root / "truth"
    run.mkdir()
    truth_dir.mkdir()
    rich = {
        "pages": [
            {
                "page": 1,
                "layers": {"tesseract5": tesseract_layer(SAFE_TEXT, 99, attempted)},
            },
            {
                "page": 2,
                "layers": {"tesseract5": tesseract_layer(UNSAFE_OCR, 72, attempted)},
            },
        ]
    }
    rich_path = run / "report_rich.json"
    evidence_sha256 = write_json(rich_path, rich)
    write_json(
        run / "job.json",
        {"source": {"content_sha256": source_sha256}},
    )
    truth_pages = [
        {
            "page": 1,
            "text": SAFE_TEXT,
            "review": {
                "status": "adjudicated",
                "auto_accept": True,
                "reasons": [],
            },
        }
    ]
    if include_second_truth_page:
        truth_pages.append(
            {
                "page": 2,
                "text": UNSAFE_REFERENCE,
                "review": {
                    "status": "adjudicated",
                    "auto_accept": False,
                    "reasons": ["critical_field_error"],
                },
            }
        )
    truth_path = truth_dir / "report.json"
    write_json(
        truth_path,
        {
            "schema_version": calibration.HUMAN_TRANSCRIPTION_SCHEMA_VERSION,
            "document_id": "report-001",
            "source_sha256": truth_source_sha256 or source_sha256,
            "tesseract_evidence_sha256": evidence_sha256 if bind_evidence else "b" * 64,
            "pages": truth_pages,
        },
    )
    corpus_path = root / "development.json"
    write_json(
        corpus_path,
        {
            "schema_version": calibration.CORPUS_SCHEMA_VERSION,
            "split": "development",
            "documents": [
                {
                    "id": "report-001",
                    "ocr_json": "run/report_rich.json",
                    "human_truth_json": "truth/report.json",
                }
            ],
        },
    )
    return corpus_path


class TesseractGateCalibrationTests(unittest.TestCase):
    def test_replays_candidates_and_preserves_reviewed_confusion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = make_corpus(root)
            output = root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                report = calibration.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--out-dir",
                        str(output),
                        "--candidate",
                        "current:65:0.70:60",
                        "--candidate",
                        "strict:90:0.90:90",
                        "--max-false-accepts",
                        "0",
                    ]
                )

            current, strict = report["candidates"]
            self.assertEqual(
                current["summary"]["confusion_matrix"],
                {
                    "safe_accepted": 1,
                    "unsafe_accepted_false_accepts": 1,
                    "safe_rejected_false_rejects": 0,
                    "unsafe_rejected": 0,
                },
            )
            self.assertEqual(
                strict["summary"]["confusion_matrix"],
                {
                    "safe_accepted": 1,
                    "unsafe_accepted_false_accepts": 0,
                    "safe_rejected_false_rejects": 0,
                    "unsafe_rejected": 1,
                },
            )
            # Neither setting dominates the other: the current gate covers
            # more pages but lets one unsafe page through; strict eliminates
            # that escape at lower coverage.
            self.assertTrue(current["summary"]["pareto_optimal"])
            self.assertTrue(strict["summary"]["pareto_optimal"])
            self.assertEqual(
                report["selection_policy"]["highest_coverage_eligible_candidate"],
                "strict",
            )
            self.assertEqual(current["summary"]["accepted_only_accuracy"]["pages"], 2)
            self.assertEqual(strict["summary"]["accepted_only_accuracy"]["pages"], 1)
            self.assertNotIn("reference_text", current["page_results"][0])
            self.assertNotIn("tesseract_text", current["page_results"][0])

            persisted = json.loads(
                (output / "tesseract_gate_calibration.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, report)
            csv_text = (output / "tesseract_gate_candidates.csv").read_text(encoding="utf-8")
            self.assertIn("unsafe_accepted_false_accepts", csv_text)
            self.assertIn("strict", csv_text)

    def test_refuses_missing_raw_tesseract_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = make_corpus(Path(temporary), attempted=False)
            with self.assertRaisesRegex(ValueError, "no raw attempted Tesseract evidence"):
                calibration.load_calibration_pages(corpus)

    def test_refuses_truth_bound_to_different_tesseract_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = make_corpus(Path(temporary), bind_evidence=False)
            with self.assertRaisesRegex(
                ValueError, "does not bind its reviewed Tesseract evidence"
            ):
                calibration.load_calibration_pages(corpus)

    def test_refuses_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = make_corpus(Path(temporary), truth_source_sha256="c" * 64)
            with self.assertRaisesRegex(ValueError, "source_sha256 does not match"):
                calibration.load_calibration_pages(corpus)

    def test_refuses_page_set_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = make_corpus(Path(temporary), include_second_truth_page=False)
            with self.assertRaisesRegex(ValueError, "page IDs do not exactly match"):
                calibration.load_calibration_pages(corpus)


if __name__ == "__main__":
    unittest.main()
