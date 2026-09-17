import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import compare_ocr


def evaluation(
    engine,
    *,
    pages=2,
    protocol=True,
    matching_evidence="sttl-matching-evidence/v1/text-only",
):
    value = {
        "engine": engine,
        "pages_evaluated": pages,
        "aggregate": {
            "weighted_CER": 0.10,
            "weighted_WER": 0.20,
            "mean_reading_order_inversion_rate": 0.30,
            "mean_structure_macro_f1": 0.40,
        },
        "reading_order": {
            "per_page": [
                {"page": page, "status": "measured", "inversion_rate": 0.30}
                for page in range(1, pages + 1)
            ],
        },
    }
    if protocol:
        value["comparison_protocol"] = {
            "version": "evaluation-comparison/v1",
            "reference_sha256": "a" * 64,
            "structure_sha256": "b" * 64,
            "matching_evidence": matching_evidence,
            "page_alignment": {
                "evaluated_page_ids": list(range(1, pages + 1)),
            },
        }
    return value


class CompareOcrTests(unittest.TestCase):
    def test_cascade_cli_labels_left_engine_and_winners(self) -> None:
        cascade = evaluation("cascade")
        chandra = evaluation("chandra")
        chandra["aggregate"].update(
            {
                "weighted_CER": 0.20,
                "weighted_WER": 0.10,
                "mean_reading_order_inversion_rate": 0.40,
                "mean_structure_macro_f1": 0.50,
            }
        )
        chandra["reading_order"]["per_page"] = [
            {"page": page, "status": "measured", "inversion_rate": 0.40} for page in range(1, 3)
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cascade_path = root / "cascade.json"
            chandra_path = root / "chandra.json"
            out = root / "comparison.json"
            cascade_path.write_text(json.dumps(cascade), encoding="utf-8")
            chandra_path.write_text(json.dumps(chandra), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                result = compare_ocr.main(
                    [
                        "--cascade",
                        str(cascade_path),
                        "--chandra",
                        str(chandra_path),
                        "--out",
                        str(out),
                    ]
                )

            persisted = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result, persisted)
        self.assertEqual(result["left_engine"], "cascade")
        self.assertIn("cascade", result)
        self.assertNotIn("surya", result)
        self.assertEqual(result["comparability"], "verified")
        self.assertEqual(
            result["winner_by_metric"],
            {
                "weighted_CER": "cascade",
                "weighted_WER": "chandra",
                "reading_order_inversion_rate": "cascade",
                "structure_macro_f1": "chandra",
            },
        )

    def test_surya_flag_remains_supported_and_is_dynamically_labeled(self) -> None:
        result = compare_ocr.build_comparison(
            evaluation("surya"),
            evaluation("chandra"),
            left_engine="surya",
        )

        self.assertEqual(result["left_engine"], "surya")
        self.assertEqual(result["winner_by_metric"]["weighted_CER"], "tie")
        self.assertIn("surya", result)

    def test_left_flags_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                compare_ocr.build_parser().parse_args(
                    [
                        "--cascade",
                        "cascade.json",
                        "--surya",
                        "surya.json",
                        "--chandra",
                        "chandra.json",
                    ]
                )

    def test_rejects_wrong_declared_engine(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 'cascade'"):
            compare_ocr.build_comparison(
                evaluation("surya"),
                evaluation("chandra"),
                left_engine="cascade",
            )

        with self.assertRaisesRegex(ValueError, "expected 'chandra'"):
            compare_ocr.build_comparison(
                evaluation("cascade"),
                evaluation("surya"),
                left_engine="cascade",
            )

    def test_rejects_mismatched_page_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "page counts do not match"):
            compare_ocr.build_comparison(
                evaluation("cascade", pages=2),
                evaluation("chandra", pages=3),
                left_engine="cascade",
            )

    def test_legacy_evaluations_are_unverified_and_suppress_winners(self) -> None:
        result = compare_ocr.build_comparison(
            evaluation("cascade", protocol=False),
            evaluation("chandra", protocol=False),
            left_engine="cascade",
        )

        self.assertEqual(result["comparability"], "unverified")
        self.assertEqual(
            result["winner_by_metric"],
            {
                "weighted_CER": None,
                "weighted_WER": None,
                "reading_order_inversion_rate": None,
                "structure_macro_f1": None,
            },
        )
        self.assertEqual(result["winner_suppression_reason"], "unverified_comparability")

    def test_allow_unverified_explicitly_restores_winners(self) -> None:
        result = compare_ocr.build_comparison(
            evaluation("cascade", protocol=False),
            evaluation("chandra", protocol=False),
            left_engine="cascade",
            allow_unverified=True,
        )

        self.assertEqual(result["comparability"], "unverified")
        self.assertEqual(result["winner_by_metric"]["weighted_CER"], "tie")
        self.assertTrue(result["unverified_comparison_allowed"])

    def test_rejects_mismatched_complete_protocols(self) -> None:
        cascade = evaluation("cascade")
        chandra = evaluation("chandra")
        chandra["comparison_protocol"]["reference_sha256"] = "c" * 64

        with self.assertRaisesRegex(ValueError, "comparison_protocol values do not match"):
            compare_ocr.build_comparison(
                cascade,
                chandra,
                left_engine="cascade",
            )

    def test_rejects_same_count_but_different_evaluated_page_ids(self) -> None:
        cascade = evaluation("cascade")
        chandra = evaluation("chandra")
        chandra["comparison_protocol"]["page_alignment"]["evaluated_page_ids"] = [1, 3]

        with self.assertRaisesRegex(ValueError, "comparison_protocol values do not match"):
            compare_ocr.build_comparison(
                cascade,
                chandra,
                left_engine="cascade",
            )

    def test_geometry_enabled_layout_winners_are_inconclusive(self) -> None:
        cascade = evaluation(
            "cascade",
            matching_evidence="sttl-matching-evidence/v1/text-and-geometry",
        )
        chandra = evaluation(
            "chandra",
            matching_evidence="sttl-matching-evidence/v1/text-and-geometry",
        )

        result = compare_ocr.build_comparison(
            cascade,
            chandra,
            left_engine="cascade",
        )

        self.assertEqual(result["comparability"], "verified")
        self.assertEqual(
            result["winner_by_metric"]["reading_order_inversion_rate"],
            "inconclusive",
        )
        self.assertEqual(
            result["winner_by_metric"]["structure_macro_f1"],
            "inconclusive",
        )
        self.assertIn("--matching-evidence text-only", result["layout_metric_caveat"])

    def test_text_only_reading_order_uses_only_common_measured_pages(self) -> None:
        cascade = evaluation("cascade")
        chandra = evaluation("chandra")
        cascade["reading_order"]["per_page"] = [
            {"page": 1, "status": "measured", "inversion_rate": 0.0},
            {"page": 2, "status": "measured", "inversion_rate": 1.0},
        ]
        chandra["reading_order"]["per_page"] = [
            {"page": 1, "status": "measured", "inversion_rate": 0.5},
            {"page": 2, "status": "inconclusive", "inversion_rate": None},
        ]

        result = compare_ocr.build_comparison(
            cascade,
            chandra,
            left_engine="cascade",
        )

        self.assertEqual(result["winner_by_metric"]["reading_order_inversion_rate"], "cascade")
        self.assertEqual(
            result["paired_reading_order"],
            {
                "status": "measured",
                "pages": [1],
                "cascade_mean_inversion_rate": 0.0,
                "chandra_mean_inversion_rate": 0.5,
                "winner": "cascade",
            },
        )

    def test_rejects_non_finite_or_non_scalar_metrics(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, 10**1000, "0.1", True):
            with self.subTest(invalid=repr(invalid)):
                chandra = copy.deepcopy(evaluation("chandra"))
                chandra["aggregate"]["weighted_CER"] = invalid
                with self.assertRaisesRegex(
                    ValueError,
                    "aggregate.weighted_CER must be a finite numeric scalar",
                ):
                    compare_ocr.build_comparison(
                        evaluation("cascade"),
                        chandra,
                        left_engine="cascade",
                    )


if __name__ == "__main__":
    unittest.main()
