import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz

from pdf_pipeline import (
    PipelineConfig,
    _PendingSuryaPage,
    build_parser,
    config_from_args,
    flush_surya,
    hybridize_surya_layout_with_tesseract,
    normalized_page,
    rich_page,
    surya_layer,
    tesseract_layer,
)


def word(text, bbox, reading_order, confidence=96.0):
    return {
        "text": text,
        "bbox": bbox,
        "reading_order": reading_order,
        "confidence": confidence,
    }


class HybridSuryaTextTests(unittest.TestCase):
    def test_pure_tesseract_keeps_tesseract_as_its_layout_engine(self) -> None:
        record = {
            "page": 1,
            "engine": "tesseract5",
            "outcome": "tesseract_accepted",
            "text": "Plain OCR text",
            "blocks": [{
                "block_type": "Text",
                "type": "text",
                "text": "Plain OCR text",
                "reading_order": 1,
            }],
        }
        document = fitz.open()
        page = document.new_page()
        normalized = normalized_page(record, page)
        rich = rich_page(record, page)

        self.assertEqual(normalized["metadata"]["layout_engine"], "tesseract5")
        self.assertEqual(normalized["metadata"]["text_engine"], "tesseract5")
        self.assertEqual(rich["authoritative"]["layout_engine"], "tesseract5")
        self.assertEqual(rich["authoritative"]["text_engine"], "tesseract5")
        document.close()

    def test_cli_can_keep_surya_text_on_structure_routes(self) -> None:
        args = build_parser().parse_args([
            "fixture.pdf",
            "--structure-aware",
            "--no-structure-hybrid-text",
        ])
        config = config_from_args(args)
        self.assertTrue(config.structure_aware)
        self.assertFalse(config.structure_hybrid_text)

    def test_hybrid_uses_only_confident_words_in_their_surya_regions(self) -> None:
        surya_blocks = [
            {
                "block_type": "SectionHeader",
                "type": "sectionheader",
                "text": "Project Ledger draft",
                "bbox": [0, 0, 180, 30],
                "reading_order": 1,
            },
            {
                "block_type": "Table",
                "type": "table",
                "text": "Item Value Cable 48 source",
                "bbox": [0, 40, 240, 140],
                "reading_order": 2,
                "html": "<table><tr><td>Item Value Cable 48 source</td></tr></table>",
                "table": {"rows": []},
            },
            {
                "block_type": "Figure",
                "type": "figure",
                "text": "",
                "bbox": [0, 150, 240, 220],
                "reading_order": 3,
                "retain_empty": True,
            },
        ]
        tesseract_words = [
            word("Project", [10, 5, 60, 20], 1),
            word("Ledger", [70, 5, 125, 20], 2),
            word("draft", [130, 5, 170, 20], 3),
            word("Item", [10, 55, 42, 70], 4),
            word("Value", [75, 55, 120, 70], 5),
            word("Cable", [10, 85, 55, 100], 6),
            word("48", [75, 85, 90, 100], 7),
            word("source", [100, 85, 150, 100], 8),
            word("uncertain", [155, 85, 215, 100], 9, confidence=42.0),
            word("outside", [300, 85, 350, 100], 10, confidence=42.0),
        ]

        hybrid, details = hybridize_surya_layout_with_tesseract(
            surya_blocks,
            tesseract_words,
            confident_word_threshold=60.0,
        )

        self.assertTrue(details["applied"])
        self.assertEqual(details["hybridized_regions"], 2)
        self.assertEqual(details["selected_tesseract_words"], 8)
        self.assertEqual(hybrid[0]["text"], "Project Ledger draft")
        self.assertEqual(hybrid[1]["text"], "Item Value Cable 48 source")
        self.assertNotIn("table", hybrid[1])
        self.assertEqual(hybrid[1]["surya_structure"]["table"], {"rows": []})
        provenance = hybrid[1]["text_provenance"]
        self.assertEqual(provenance["layout_engine"], "surya")
        self.assertEqual(provenance["text_engine"], "tesseract5")
        self.assertEqual(provenance["surya_text"], "Item Value Cable 48 source")
        self.assertEqual(provenance["tesseract_word_reading_orders"], [4, 5, 6, 7, 8])
        self.assertEqual(provenance["tesseract_word_ordering"], "spatial_row_major")
        self.assertIn("token_jaccard", provenance["agreement"])
        self.assertEqual(hybrid[2]["text"], "")
        # The raw engine payload is not modified while preparing the hybrid.
        self.assertEqual(surya_blocks[0]["text"], "Project Ledger draft")

    def test_hybrid_declines_partial_or_overlapping_regions(self) -> None:
        partial_blocks = [
            {"block_type": "Text", "type": "text", "text": "Alpha", "bbox": [0, 0, 80, 20], "reading_order": 1},
            {"block_type": "Text", "type": "text", "text": "Bravo", "bbox": [0, 30, 80, 50], "reading_order": 2},
        ]
        partial, partial_details = hybridize_surya_layout_with_tesseract(
            partial_blocks,
            [word("Alpha", [5, 2, 35, 17], 1)],
            confident_word_threshold=60.0,
        )
        self.assertFalse(partial_details["applied"])
        self.assertEqual(partial_details["skip_reason"], "insufficient_per_region_agreement")
        self.assertEqual([block["text"] for block in partial], ["Alpha", "Bravo"])

        overlapping_blocks = [
            {"block_type": "Text", "type": "text", "text": "Alpha", "bbox": [0, 0, 100, 100], "reading_order": 1},
            {"block_type": "SectionHeader", "type": "sectionheader", "text": "Alpha", "bbox": [0, 0, 60, 25], "reading_order": 2},
        ]
        overlapping, overlap_details = hybridize_surya_layout_with_tesseract(
            overlapping_blocks,
            [word("Alpha", [5, 2, 35, 17], 1)],
            confident_word_threshold=60.0,
        )
        self.assertFalse(overlap_details["applied"])
        self.assertEqual(overlap_details["skip_reason"], "overlapping_surya_text_regions")
        self.assertEqual([block["text"] for block in overlapping], ["Alpha", "Alpha"])

    def test_hybrid_restores_row_major_order_from_table_word_geometry(self) -> None:
        # The engines contain exactly the same cells, but Tesseract's global
        # word order is column-major. Recover visible row order before the
        # exact table-sequence check, rather than trusting that global order.
        surya_blocks = [{
            "block_type": "Table",
            "type": "table",
            "text": "A1 B1 A2 B2",
            "bbox": [0, 0, 120, 80],
            "reading_order": 1,
        }]
        tesseract_words = [
            word("A1", [5, 5, 25, 20], 1),
            word("A2", [5, 45, 25, 60], 2),
            word("B1", [70, 5, 90, 20], 3),
            word("B2", [70, 45, 90, 60], 4),
        ]

        blocks, details = hybridize_surya_layout_with_tesseract(
            surya_blocks,
            tesseract_words,
            confident_word_threshold=60.0,
        )

        self.assertTrue(details["applied"])
        self.assertEqual(blocks[0]["text"], "A1 B1 A2 B2")
        provenance = blocks[0]["text_provenance"]
        self.assertEqual(provenance["tesseract_word_ordering"], "spatial_row_major")
        self.assertEqual(provenance["tesseract_word_reading_orders"], [1, 3, 2, 4])
        self.assertEqual(provenance["minimum_sequence_ratio"], 1.0)

    def test_hybrid_declines_table_with_nonidentical_sequence_or_signed_value(self) -> None:
        reordered, reordered_details = hybridize_surya_layout_with_tesseract(
            [{
                "block_type": "Table",
                "type": "table",
                "text": "A B C D E",
                "bbox": [0, 0, 160, 30],
                "reading_order": 1,
            }],
            [
                word("A", [5, 5, 15, 20], 1),
                word("C", [25, 5, 35, 20], 2),
                word("B", [45, 5, 55, 20], 3),
                word("D", [65, 5, 75, 20], 4),
                word("E", [85, 5, 95, 20], 5),
            ],
            confident_word_threshold=60.0,
        )
        signed, signed_details = hybridize_surya_layout_with_tesseract(
            [{
                "block_type": "Text",
                "type": "text",
                "text": "Balance -100",
                "bbox": [0, 0, 160, 30],
                "reading_order": 1,
            }],
            [
                word("Balance", [5, 5, 55, 20], 1),
                word("100", [65, 5, 85, 20], 2),
            ],
            confident_word_threshold=60.0,
        )

        self.assertFalse(reordered_details["applied"])
        self.assertEqual(reordered[0]["text"], "A B C D E")
        self.assertIn(
            "low_token_sequence_ratio",
            reordered_details["rejected_regions"][0]["rejection_reasons"],
        )
        self.assertFalse(signed_details["applied"])
        self.assertEqual(signed[0]["text"], "Balance -100")
        self.assertIn(
            "critical_value_token_mismatch",
            signed_details["rejected_regions"][0]["rejection_reasons"],
        )

    def test_hybrid_declines_when_any_canonical_surya_token_is_missing(self) -> None:
        blocks, details = hybridize_surya_layout_with_tesseract(
            [{
                "block_type": "Text",
                "type": "text",
                "text": "one two three four five six seven eight nine ten",
                "bbox": [0, 0, 300, 30],
                "reading_order": 1,
            }],
            [
                word(text, [index * 25, 5, index * 25 + 20, 20], index)
                for index, text in enumerate(
                    "one two three four five six seven eight nine".split(),
                    start=1,
                )
            ],
            confident_word_threshold=60.0,
        )

        self.assertFalse(details["applied"])
        self.assertEqual(blocks[0]["text"], "one two three four five six seven eight nine ten")
        self.assertIn("low_token_f1", details["rejected_regions"][0]["rejection_reasons"])

    def test_hybrid_declines_high_confidence_word_outside_surya_regions(self) -> None:
        blocks, details = hybridize_surya_layout_with_tesseract(
            [{
                "block_type": "Text",
                "type": "text",
                "text": "Alpha",
                "bbox": [0, 0, 100, 30],
                "reading_order": 1,
            }],
            [
                word("Alpha", [5, 5, 35, 20], 1),
                word("outside", [150, 5, 200, 20], 2),
            ],
            confident_word_threshold=60.0,
        )

        self.assertFalse(details["applied"])
        self.assertEqual(details["skip_reason"], "too_many_unassigned_tesseract_words")
        self.assertEqual(details["unassigned_high_confidence_tesseract_word_ratio"], 0.5)
        self.assertEqual(blocks[0]["text"], "Alpha")

    def test_hybrid_declines_visual_surya_text(self) -> None:
        blocks, details = hybridize_surya_layout_with_tesseract(
            [
                {"block_type": "Text", "type": "text", "text": "Body", "bbox": [0, 0, 100, 20], "reading_order": 1},
                {"block_type": "Figure", "type": "figure", "text": "Surya visual description", "bbox": [0, 30, 100, 80], "reading_order": 2},
            ],
            [word("Body", [5, 2, 35, 17], 1)],
            confident_word_threshold=60.0,
        )
        self.assertFalse(details["applied"])
        self.assertEqual(details["skip_reason"], "non_tesseract_textual_surya_regions")
        self.assertEqual(blocks[1]["text"], "Surya visual description")

    def test_structure_flush_preserves_raw_surya_and_marks_split_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = [
                _PendingSuryaPage(
                    page=1,
                    inspection={"classification": "SCANNED", "route": "tesseract", "signals": {}, "native_text": ""},
                    image_path=root / "page-000001.png",
                    quality={"accepted": True, "mean_word_confidence": 95.0},
                    raster={"raster_width": 100, "raster_height": 100, "pdf_bbox": [0, 0, 100, 100]},
                    tesseract_text="Reliable table words source",
                    tesseract_words=[
                        word("Reliable", [5, 5, 45, 20], 1),
                        word("table", [50, 5, 78, 20], 2),
                        word("words", [5, 30, 34, 45], 3),
                        word("source", [40, 30, 80, 45], 4),
                    ],
                    tesseract_blocks=[],
                    escalation_reason="structure",
                    structure_gate={"enabled": True, "escalate": True},
                )
            ]
            prediction = {
                "page-000001.png": {
                    "text": "Reliable table words source",
                    "blocks": [{
                        "block_type": "Table",
                        "type": "table",
                        "text": "Reliable table words source",
                        "bbox": [0, 0, 100, 60],
                        "reading_order": 1,
                        "html": "<table><tr><td>Reliable table words source</td></tr></table>",
                    }],
                }
            }
            records = {}
            with patch("pdf_pipeline.surya_batch", return_value=prediction):
                flush_surya(
                    pending,
                    batch_number=1,
                    job_dir=root,
                    config=PipelineConfig(structure_aware=True),
                    manifest_path=root / "pages.jsonl",
                    records=records,
                )

            record = records[1]
            self.assertEqual(record["text"], "Reliable table words source")
            self.assertTrue(record["hybrid_text"]["applied"])
            self.assertEqual(record["surya_candidate"]["text"], "Reliable table words source")
            self.assertEqual(record["surya_candidate"]["blocks"][0]["text"], "Reliable table words source")
            self.assertEqual(record["blocks"][0]["text_engine"], "tesseract5")
            self.assertFalse(tesseract_layer(record)["selected"])
            self.assertFalse(tesseract_layer(record)["selected_text"])
            self.assertFalse(tesseract_layer(record)["selected_layout"])
            self.assertTrue(tesseract_layer(record)["contributes_to_authoritative_text"])
            self.assertFalse(surya_layer(record)["selected"])
            self.assertFalse(surya_layer(record)["selected_layout"])
            self.assertFalse(surya_layer(record)["selected_text"])
            self.assertTrue(surya_layer(record)["contributes_to_authoritative_layout"])
            document = fitz.open()
            page = document.new_page()
            rich = rich_page(record, page)
            self.assertEqual(rich["authoritative"]["layout_engine"], "surya")
            self.assertEqual(rich["authoritative"]["text_engine"], "tesseract5")
            self.assertEqual(rich["layers"]["surya"]["text"], "Reliable table words source")
            self.assertEqual(rich["authoritative"]["blocks"][0]["text"], "Reliable table words source")
            document.close()

    def test_hybrid_marks_only_reconstructed_authoritative_text_as_selected(self) -> None:
        record = {
            "page": 1,
            "engine": "surya",
            "outcome": "surya_escalated",
            "text": "A1 B1 A2 B2",
            "blocks": [{
                "block_type": "Table",
                "type": "table",
                "text": "A1 B1 A2 B2",
                "reading_order": 1,
                "layout_engine": "surya",
                "text_engine": "tesseract5",
            }],
            "hybrid_text": {"applied": True},
            "tesseract_candidate": {
                "text": "A1 A2 B1 B2",
                "words": [],
                "blocks": [],
                "quality": {"accepted": True},
            },
            "surya_candidate": {
                "text": "A1 B1 A2 B2",
                "blocks": [],
            },
        }
        document = fitz.open()
        page = document.new_page()
        rich = rich_page(record, page)
        tesseract = rich["layers"]["tesseract5"]
        surya = rich["layers"]["surya"]

        self.assertEqual(rich["authoritative"]["text"], "A1 B1 A2 B2")
        self.assertEqual(tesseract["text"], "A1 A2 B1 B2")
        self.assertFalse(tesseract["selected"])
        self.assertFalse(tesseract["selected_text"])
        self.assertTrue(tesseract["contributes_to_authoritative_text"])
        self.assertFalse(surya["selected"])
        self.assertFalse(surya["selected_layout"])
        self.assertTrue(surya["contributes_to_authoritative_layout"])
        document.close()

    def test_quality_rejection_never_enables_the_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = [
                _PendingSuryaPage(
                    page=1,
                    inspection={"classification": "SCANNED", "route": "tesseract", "signals": {}, "native_text": ""},
                    image_path=root / "page-000001.png",
                    quality={"accepted": False, "mean_word_confidence": 10.0},
                    raster={"raster_width": 100, "raster_height": 100, "pdf_bbox": [0, 0, 100, 100]},
                    tesseract_text="Do not use this",
                    tesseract_words=[word("Do", [5, 5, 20, 20], 1), word("not", [25, 5, 45, 20], 2)],
                    tesseract_blocks=[],
                    escalation_reason="quality",
                )
            ]
            prediction = {
                "page-000001.png": {
                    "text": "Surya is authoritative",
                    "blocks": [{
                        "block_type": "Text",
                        "type": "text",
                        "text": "Surya is authoritative",
                        "bbox": [0, 0, 100, 60],
                        "reading_order": 1,
                    }],
                }
            }
            records = {}
            with patch("pdf_pipeline.surya_batch", return_value=prediction):
                flush_surya(
                    pending,
                    batch_number=1,
                    job_dir=root,
                    config=PipelineConfig(structure_aware=True),
                    manifest_path=root / "pages.jsonl",
                    records=records,
                )

            self.assertEqual(records[1]["text"], "Surya is authoritative")
            self.assertNotIn("hybrid_text", records[1])
            self.assertNotIn("surya_candidate", records[1])
            self.assertFalse(tesseract_layer(records[1])["selected"])

    def test_declined_structure_hybrid_is_recorded_as_pure_surya(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = [_PendingSuryaPage(
                page=1,
                inspection={"classification": "SCANNED", "route": "tesseract", "signals": {}, "native_text": ""},
                image_path=root / "page-000001.png",
                quality={"accepted": True, "mean_word_confidence": 95.0},
                raster={"raster_width": 100, "raster_height": 100, "pdf_bbox": [0, 0, 100, 100]},
                tesseract_text="Different words",
                tesseract_words=[word("Different", [5, 5, 45, 20], 1), word("words", [50, 5, 80, 20], 2)],
                tesseract_blocks=[],
                escalation_reason="structure",
                structure_gate={"enabled": True, "escalate": True},
            )]
            prediction = {"page-000001.png": {"text": "Surya-only region", "blocks": [{
                "block_type": "Text", "type": "text", "text": "Surya-only region", "bbox": [0, 0, 100, 60], "reading_order": 1,
            }]}}
            records = {}
            with patch("pdf_pipeline.surya_batch", return_value=prediction):
                flush_surya(
                    pending, batch_number=1, job_dir=root, config=PipelineConfig(structure_aware=True),
                    manifest_path=root / "pages.jsonl", records=records,
                )
            record = records[1]
            self.assertFalse(record["hybrid_text"]["applied"])
            self.assertEqual(record["hybrid_text"]["skip_reason"], "insufficient_per_region_agreement")
            self.assertEqual(record["text"], "Surya-only region")
            self.assertFalse(tesseract_layer(record)["selected_text"])
            self.assertTrue(surya_layer(record)["selected_text"])


if __name__ == "__main__":
    unittest.main()
