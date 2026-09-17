import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evaluate_ocr import (
    COMPARISON_PROTOCOL_VERSION,
    MATCHING_EVIDENCE_POLICY_VERSION,
)
from evaluate_ocr import (
    main as evaluate_ocr_main,
)


class EvaluationComparisonProtocolTests(unittest.TestCase):
    def test_report_records_path_free_comparison_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            structure_path = root / "structure.json"
            ocr_path = root / "chandra.json"
            output_dir = root / "evaluation"

            # Deliberately retain non-canonical whitespace and a different page
            # array order. The digest must describe those exact source bytes;
            # the evaluator must separately report its deterministic page order.
            reference_bytes = (
                b"{\n"
                b'  "pages" : [\n'
                b'    {"page": 2, "text": "embedded two"},\n'
                b'    {"page": 1, "text": "embedded one"}\n'
                b"  ]\n"
                b"}\n"
            )
            structure_bytes = json.dumps(
                {
                    "pages": [
                        {
                            "page": 2,
                            "blocks": [{"type": "paragraph", "text": "two"}],
                        },
                        {
                            "page": 1,
                            "blocks": [{"type": "paragraph", "text": "one"}],
                        },
                    ]
                },
                indent=1,
            ).encode("utf-8")
            ocr_bytes = json.dumps(
                {
                    "pages": [
                        {
                            "page": 2,
                            "blocks": [
                                {
                                    "text": "two",
                                    "label": "paragraph",
                                    "reading_order": 1,
                                }
                            ],
                        },
                        {
                            "page": 1,
                            "blocks": [
                                {
                                    "text": "one",
                                    "label": "paragraph",
                                    "reading_order": 1,
                                }
                            ],
                        },
                    ]
                },
                separators=(",", ":"),
            ).encode("utf-8")
            reference_path.write_bytes(reference_bytes)
            structure_path.write_bytes(structure_bytes)
            ocr_path.write_bytes(ocr_bytes)

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_ocr.py",
                        "--reference",
                        str(reference_path),
                        "--structure",
                        str(structure_path),
                        "--engine",
                        "chandra",
                        "--ocr-json",
                        str(ocr_path),
                        "--out-dir",
                        str(output_dir),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertIsNone(evaluate_ocr_main())

            evaluation = json.loads(
                (output_dir / "chandra_evaluation.json").read_text(encoding="utf-8")
            )
            protocol = evaluation["comparison_protocol"]

            self.assertEqual(protocol["version"], COMPARISON_PROTOCOL_VERSION)
            self.assertEqual(protocol["engine"], "chandra")
            self.assertEqual(
                protocol["reference_sha256"],
                hashlib.sha256(reference_bytes).hexdigest(),
            )
            self.assertEqual(
                protocol["structure_sha256"],
                hashlib.sha256(structure_bytes).hexdigest(),
            )
            self.assertEqual(
                protocol["ocr_json_sha256"],
                hashlib.sha256(ocr_bytes).hexdigest(),
            )
            self.assertEqual(
                protocol["matching_evidence"],
                f"{MATCHING_EVIDENCE_POLICY_VERSION}/text-and-geometry",
            )
            self.assertEqual(
                protocol["page_alignment"],
                {
                    "method": "exact page-ID set equality; ascending numeric page-ID order",
                    "evaluated_page_ids": [1, 2],
                },
            )

            policy = protocol["matching_evidence_policy"]
            self.assertEqual(policy["version"], MATCHING_EVIDENCE_POLICY_VERSION)
            self.assertEqual(policy["mode"], "text-and-geometry")
            self.assertEqual(policy["text_token_jaccard"]["threshold"], 0.25)
            self.assertTrue(policy["bbox_iou"]["enabled"])
            self.assertEqual(policy["bbox_iou"]["threshold"], 0.25)
            self.assertEqual(
                policy["structure"]["canonical_semantic_type"],
                "must_match",
            )
            self.assertEqual(
                policy["reading_order"]["canonical_semantic_type"],
                "not_required",
            )

            # Paths remain in the legacy ``per_page_csv`` convenience field,
            # but never enter the comparison protocol used across machines.
            self.assertNotIn(str(root), json.dumps(protocol, sort_keys=True))
            self.assertIn("aggregate", evaluation)
            self.assertIn("per_page_csv", evaluation)


if __name__ == "__main__":
    unittest.main()
