"""Tests for cross-engine-safe text-only layout evaluation."""

from __future__ import annotations

import unittest

from evaluate_ocr import (
    matching_evidence_policy,
    order_similarity_details,
    structure_scores,
)


def block(*, text: str, order: int) -> dict:
    return {
        "type": "paragraph",
        "label": "paragraph",
        "text": text,
        "bbox": [10, 10 + order * 20, 90, 25 + order * 20],
        "reading_order": order,
        "geometry": {
            "coordinate_space": "pdf_points",
            "coordinate_frame": "unrotated_pdf_points",
            "page_bbox": [0, 0, 100, 100],
        },
    }


class MatchingEvidencePolicyTests(unittest.TestCase):
    def test_text_only_does_not_match_blocks_on_geometry_alone(self) -> None:
        reference = [block(text="पहला संदर्भ", order=1), block(text="दूसरा संदर्भ", order=2)]
        ocr = [block(text="unrelated one", order=2), block(text="unrelated two", order=1)]

        hybrid_structure = structure_scores(reference, ocr)
        text_only_structure = structure_scores(
            reference,
            ocr,
            matching_evidence="text-only",
        )
        hybrid_order = order_similarity_details(reference, ocr)
        text_only_order = order_similarity_details(
            reference,
            ocr,
            matching_evidence="text-only",
        )

        self.assertEqual(hybrid_structure["matching"]["matched_blocks"], 2)
        self.assertEqual(text_only_structure["matching"]["matched_blocks"], 0)
        self.assertEqual(hybrid_order["matched_blocks"], 2)
        self.assertEqual(text_only_order["matched_blocks"], 0)
        self.assertEqual(
            text_only_structure["matching"]["criteria"]["matching_evidence"],
            "text-only",
        )

    def test_text_only_policy_is_explicit_and_disables_bbox_candidates(self) -> None:
        policy = matching_evidence_policy("text-only")

        self.assertEqual(policy["mode"], "text-only")
        self.assertEqual(
            policy["one_to_one_assignment"]["candidate_rule"],
            "text_token_jaccard",
        )
        self.assertFalse(policy["bbox_iou"]["enabled"])


if __name__ == "__main__":
    unittest.main()
