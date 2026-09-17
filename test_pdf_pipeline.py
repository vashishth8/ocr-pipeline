import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf as fitz

from evaluate_ocr import (
    aligned_pages,
    canonical_label,
    geometry_compatibility,
    load_cascade,
    order_similarity,
    order_similarity_details,
    page_ocr_blocks,
    structure_scores,
    visual_description_labels,
)
from evaluate_ocr import (
    main as evaluate_ocr_main,
)
from evaluate_ocr import (
    tokens as evaluation_tokens,
)
from evaluate_ocr import (
    wer as evaluation_wer,
)
from pdf_pipeline import (
    PipelineConfig,
    _PendingSuryaPage,
    build_parser,
    canonical_blocks,
    classify_page_signals,
    config_from_args,
    extract_surya_layout,
    flush_surya,
    html_to_table,
    html_to_text,
    page_bbox,
    parse_tesseract_tsv,
    parse_tesseract_tsv_layout,
    primary_ocr_engine,
    process_document,
    reading_order_check,
    render_page,
    scale_ocr_items_to_pdf_points,
    summarise_document,
    tesseract_page,
    tesseract_quality,
    tesseract_structure_gate,
    text_agreement,
    validate_args,
    validate_pipeline_config,
)
from run_chandra import first_text, normalize_chandra
from run_chandra import main as chandra_main


class PipelineUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig()

    def test_usable_native_text_is_digital(self) -> None:
        result = classify_page_signals(
            text="This is a normal digital PDF page with enough useful text. " * 3,
            text_block_count=3,
            image_count=0,
            image_area_ratio=0.0,
            config=self.config,
        )
        self.assertEqual((result["classification"], result["route"]), ("DIGITAL", "native_text"))

    def test_dominant_image_with_text_is_mixed(self) -> None:
        result = classify_page_signals(
            text="This page has native text but the visual scan is dominant. " * 3,
            text_block_count=2,
            image_count=1,
            image_area_ratio=0.9,
            config=self.config,
        )
        self.assertEqual((result["classification"], result["route"]), ("MIXED", "tesseract"))

    def test_private_use_native_text_is_auditable_and_strict_profile_routes_it(self) -> None:
        # A custom-font PDF can contain a tiny amount of private-use Unicode
        # amid otherwise valid native text. The regular profile permits its
        # configured 5% garbage allowance; a strict zero-garbage pilot profile
        # makes the page route to local OCR without a code-path special case.
        text = ("यह एक पर्याप्त लंबा हिंदी परीक्षण पाठ है। " * 10) + "\ue019"
        ordinary = classify_page_signals(
            text=text,
            text_block_count=2,
            image_count=0,
            image_area_ratio=0.0,
            config=self.config,
        )
        self.assertEqual(ordinary["signals"]["native_private_use_count"], 1)
        self.assertEqual(
            (ordinary["classification"], ordinary["route"]), ("DIGITAL", "native_text")
        )

        strict = classify_page_signals(
            text=text,
            text_block_count=2,
            image_count=0,
            image_area_ratio=0.0,
            config=PipelineConfig(max_native_garbage_ratio=0.0),
        )
        self.assertEqual((strict["classification"], strict["route"]), ("OCR_NEEDED", "tesseract"))

    def test_tsv_quality_gate(self) -> None:
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        for number, word in enumerate(("Reliable", "OCR", "text", "here"), 1):
            tsv += f"5\t1\t1\t1\t1\t{number}\t0\t0\t1\t1\t93.0\t{word}\n"
        text, confidences = parse_tesseract_tsv(tsv)
        self.assertEqual(text, "Reliable OCR text here")
        self.assertTrue(tesseract_quality(text, confidences, self.config)["accepted"])

    def test_hindi_quality_and_comparison_preserve_devanagari_marks(self) -> None:
        text = "यह हिंदी भाषा का पर्याप्त परीक्षण पाठ है और भारत में हिंदी बोली जाती है।"
        quality = tesseract_quality(
            text, [93.0] * len(text.split()), PipelineConfig(language="hin")
        )
        self.assertTrue(quality["accepted"], quality)
        self.assertEqual(quality["plausible_word_ratio"], 1.0)

        agreement = text_agreement("हिंदी", "हिदी")
        self.assertEqual(agreement["token_f1"], 0.0)
        self.assertEqual(agreement["sequence_ratio"], 0.0)
        self.assertEqual(evaluation_tokens("हिंदी भाषा।"), ["हिंदी", "भाषा"])
        self.assertEqual(evaluation_wer("हिंदी", "हिदी"), 1.0)
        # Canonically equivalent Devanagari encodings should compare equally.
        self.assertEqual(evaluation_tokens("क़िला"), evaluation_tokens("क़िला"))

    def test_hindi_auto_prefers_surya_but_compact_profile_stays_tesseract(self) -> None:
        # The accuracy-first Hindi route should select Surya automatically,
        # including for code-mixed and script-name language settings. A caller
        # can still make its compact resource choice explicit and deterministic.
        self.assertEqual(primary_ocr_engine(PipelineConfig(language="hin")), "surya")
        self.assertEqual(primary_ocr_engine(PipelineConfig(language="hin+eng")), "surya")
        self.assertEqual(primary_ocr_engine(PipelineConfig(language="script/Devanagari")), "surya")
        self.assertEqual(primary_ocr_engine(PipelineConfig(language="eng")), "tesseract")
        self.assertEqual(
            primary_ocr_engine(PipelineConfig(language="hin", fallback_engine="none")),
            "tesseract",
        )
        self.assertEqual(
            primary_ocr_engine(PipelineConfig(language="hin", ocr_engine="tesseract")),
            "tesseract",
        )

    def test_hindi_dry_run_reports_resolved_primary_plan_without_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "hindi-plan.pdf"
            document = fitz.open()
            document.new_page()
            document.new_page()
            document.save(pdf_path)
            document.close()
            native = {
                "classification": "DIGITAL",
                "route": "native_text",
                "signals": {},
                "native_text": "usable native text",
            }
            ocr_needed = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            with patch("pdf_pipeline.inspect_page", side_effect=[native, ocr_needed]):
                plan = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(language="hin"),
                    dry_run=True,
                )

        self.assertEqual(plan["state"], "dry_run")
        self.assertEqual(plan["primary_ocr_engine"], "surya")
        self.assertEqual(plan["ocr_candidate_pages"], 1)
        self.assertEqual(plan["planned_surya_primary_pages"], 1)
        self.assertEqual(plan["planned_tesseract_primary_pages"], 0)

    def test_missing_hindi_tesseract_data_has_an_actionable_error(self) -> None:
        failure = CalledProcessError(
            1,
            ["tesseract"],
            stderr="Failed loading language 'hin'\nCould not initialize tesseract.",
        )
        with (
            patch("pdf_pipeline.executable_for", return_value="tesseract"),
            patch("pdf_pipeline.subprocess.run", side_effect=failure),
        ):
            with self.assertRaisesRegex(RuntimeError, "language data for 'hin' is unavailable"):
                tesseract_page(Path("page.png"), PipelineConfig(language="hin"))

    def test_missing_local_tesseract_data_names_the_local_verification_command(self) -> None:
        failure = CalledProcessError(
            1,
            ["tesseract"],
            stderr="Failed loading language 'hin'\nCould not initialize tesseract.",
        )
        with (
            patch("pdf_pipeline.executable_for", return_value="tesseract"),
            patch("pdf_pipeline.subprocess.run", side_effect=failure),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"tesseract --list-langs --tessdata-dir models/tessdata",
            ):
                tesseract_page(
                    Path("page.png"),
                    PipelineConfig(language="hin", tessdata_dir="models/tessdata"),
                )

    def test_workspace_tessdata_directory_is_forwarded_to_tesseract(self) -> None:
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        for number, word in enumerate(("यह", "हिंदी", "OCR", "पाठ"), 1):
            tsv += f"5\t1\t1\t1\t1\t{number}\t0\t0\t1\t1\t93.0\t{word}\n"
        with (
            patch("pdf_pipeline.executable_for", return_value="tesseract"),
            patch("pdf_pipeline.subprocess.run", return_value=SimpleNamespace(stdout=tsv)) as run,
        ):
            tesseract_page(
                Path("page.png"),
                PipelineConfig(language="hin", tessdata_dir="models/tessdata"),
            )
        self.assertEqual(
            run.call_args.args[0],
            [
                "tesseract",
                "page.png",
                "stdout",
                "--tessdata-dir",
                "models/tessdata",
                "-l",
                "hin",
                "--psm",
                "3",
                "-c",
                "tessedit_create_tsv=1",
            ],
        )

    def test_tsv_parser_keeps_following_rows_after_a_literal_quote(self) -> None:
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        tsv += '5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tquoted"word\n'
        tsv += "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t91\tafterward\n"
        text, confidences = parse_tesseract_tsv(tsv)
        self.assertEqual(text, 'quoted"word afterward')
        self.assertEqual(confidences, [90.0, 91.0])

    def test_low_confidence_escalates(self) -> None:
        quality = tesseract_quality("Some uncertain OCR output here", [20, 30, 25, 35], self.config)
        self.assertFalse(quality["accepted"])
        self.assertIn("low_mean_confidence", quality["rejection_reasons"])

    def test_surya_html_text(self) -> None:
        text = html_to_text(
            "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>10</td></tr></table>"
        )
        self.assertIn("Item Value", text)
        self.assertIn("A 10", text)

    def test_surya_layout_keeps_html_table_confidence_and_visual_block(self) -> None:
        prediction = {
            "blocks": [
                {
                    "label": "Table",
                    "raw_label": "Table",
                    "reading_order": 0,
                    "confidence": 0.98,
                    "html": "<table><tr><th>A</th><th>B</th></tr><tr><td colspan='2'>value</td></tr></table>",
                    "bbox": [10, 20, 110, 80],
                    "polygon": [[10, 20], [110, 20], [110, 80], [10, 80]],
                    "skipped": False,
                    "error": False,
                },
                {
                    "label": "Diagram",
                    "raw_label": "Diagram",
                    "reading_order": 1,
                    "html": "",
                    "bbox": [20, 90, 100, 150],
                    "skipped": True,
                    "error": False,
                },
                {
                    # Surya does not always mark visual regions as skipped.
                    # Its semantic label and geometry remain useful even
                    # when no textual description was generated.
                    "label": "Figure",
                    "raw_label": "Figure",
                    "reading_order": 2,
                    "html": "",
                    "bbox": [30, 160, 120, 220],
                    "skipped": False,
                    "error": False,
                },
            ]
        }
        text, blocks = extract_surya_layout(prediction)
        self.assertIn("A B", text)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["confidence"], 0.98)
        self.assertIn("<table>", blocks[0]["html"])
        self.assertEqual(blocks[0]["table"]["rows"][1][0]["colspan"], 2)
        self.assertEqual(blocks[1]["type"], "diagram")
        self.assertEqual(blocks[1]["text"], "")
        self.assertTrue(blocks[2]["retain_empty"])
        self.assertEqual(blocks[2]["type"], "figure")
        self.assertEqual(len(canonical_blocks(blocks)), 3)

    def test_ocr_bboxes_are_scaled_to_pdf_points_and_source_retained(self) -> None:
        items = [
            {
                "text": "word",
                "bbox": [100, 50, 300, 150],
                "polygon": [[100, 50], [300, 50], [300, 150], [100, 150]],
            }
        ]
        raster = {
            "raster_width": 1000,
            "raster_height": 500,
            "pdf_bbox": [10, 20, 510, 270],
        }
        scaled = scale_ocr_items_to_pdf_points(items, raster)[0]
        self.assertEqual(scaled["source_bbox"], [100.0, 50.0, 300.0, 150.0])
        self.assertEqual(scaled["bbox"], [60.0, 45.0, 160.0, 95.0])
        self.assertEqual(scaled["source_polygon"][0], [100, 50])
        self.assertEqual(scaled["polygon"][0], [60.0, 45.0])
        self.assertEqual(scaled["coordinate_space"], "pdf_points")

    def test_rotated_raster_coordinates_align_with_unrotated_pymupdf_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "rotated.pdf"
            document = fitz.open()
            page = document.new_page(width=200, height=100)
            page.insert_text((20, 30), "ROTATED TEXT")
            page.set_rotation(90)
            document.save(pdf_path)
            document.close()

            document = fitz.open(pdf_path)
            page = document[0]
            raster = render_page(page, root / "rotated.png", 72)
            scaled = scale_ocr_items_to_pdf_points(
                [
                    {
                        "text": "word",
                        "bbox": [60, 20, 80, 40],
                        "polygon": [[60, 20], [80, 20], [80, 40], [60, 40]],
                    }
                ],
                raster,
            )[0]
            self.assertEqual(raster["rendered_pdf_bbox"], [0.0, 0.0, 100.0, 200.0])
            self.assertEqual(raster["pdf_bbox"], [0.0, 0.0, 200.0, 100.0])
            self.assertEqual(scaled["bbox"], [20.0, 20.0, 40.0, 40.0])
            self.assertEqual(
                scaled["polygon"], [[20.0, 40.0], [20.0, 20.0], [40.0, 20.0], [40.0, 40.0]]
            )
            self.assertEqual(page_bbox(page), [0.0, 0.0, 200.0, 100.0])
            native_bbox = page.get_text("dict")["blocks"][0]["bbox"]
            self.assertLessEqual(native_bbox[2], page_bbox(page)[2])
            self.assertLessEqual(native_bbox[3], page_bbox(page)[3])
            document.close()

    def test_html_table_preserves_header_and_spans(self) -> None:
        table = html_to_table(
            "<table><tr><th rowspan='2'>Header</th><th>Value</th></tr><tr><td>10<br/>20</td></tr></table>"
        )
        self.assertIsNotNone(table)
        assert table is not None
        self.assertTrue(table["rows"][0][0]["is_header"])
        self.assertEqual(table["rows"][0][0]["rowspan"], 2)
        self.assertEqual(table["rows"][1][0]["text"], "10\n20")

    def test_native_document_is_resumable_without_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "digital.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "A sufficiently long native text page for the production OCR pipeline. " * 4,
            )
            document.save(pdf_path)
            document.close()

            output_root = root / "output"
            first = process_document(
                pdf_path,
                input_root=root,
                output_root=output_root,
                config=self.config,
            )
            second = process_document(
                pdf_path,
                input_root=root,
                output_root=output_root,
                config=self.config,
            )
            self.assertEqual(first["state"], "complete")
            self.assertEqual(second["native_pages"], 1)
            job_dir = output_root / "digital"
            records = [
                json.loads(line) for line in (job_dir / "pages.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertTrue((job_dir / "combined.txt").is_file())
            job = json.loads((job_dir / "job.json").read_text())
            summary = json.loads((job_dir / "document.json").read_text())
            self.assertEqual(job["source"]["file_name"], "digital.pdf")
            self.assertEqual(len(job["source"]["content_sha256"]), 64)
            self.assertEqual(summary["source"], "digital.pdf")
            self.assertEqual(summary["normalized_output"], "digital_cascade.json")
            self.assertEqual(summary["rich_output"], "digital_rich.json")
            self.assertNotIn(str(root), json.dumps({"job": job, "summary": summary}))
            self.assertEqual(first["job_dir"], "digital")
            normalized = json.loads((job_dir / "digital_cascade.json").read_text())
            self.assertEqual(normalized["page_count"], 1)
            self.assertEqual(normalized["pages"][0]["block_type"], "Page")
            self.assertEqual(normalized["metadata"]["reading_order"]["status"], "maintained")
            self.assertEqual(normalized["pages"][0]["coordinate_space"], "pdf_points")
            rich = json.loads((job_dir / "digital_rich.json").read_text())
            self.assertEqual(rich["schema_version"], "cascade-ocr/rich-v1")
            native = rich["pages"][0]["layers"]["pymupdf"]
            self.assertTrue(native["blocks"][0]["lines"][0]["spans"][0]["style"]["font"])

    def test_tesseract_layout_exports_contiguous_blocks(self) -> None:
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        tsv += "5\t1\t1\t1\t1\t1\t10\t10\t30\t10\t90\tFirst\n"
        tsv += "5\t1\t1\t1\t1\t2\t45\t10\t40\t10\t91\tline\n"
        tsv += "5\t1\t1\t1\t2\t1\t10\t30\t50\t10\t92\tSecond\n"
        text, confidences, words, blocks = parse_tesseract_tsv_layout(tsv)
        self.assertEqual(text, "First line Second")
        self.assertEqual(len(words), 3)
        self.assertEqual(
            words[0]["source_tsv"], {"page": 1, "block": 1, "paragraph": 1, "line": 1, "word": 1}
        )
        self.assertEqual([block["reading_order"] for block in blocks], [1, 2])
        self.assertEqual(blocks[0]["source_tsv"]["line"], 1)
        self.assertTrue(reading_order_check(blocks)["contiguous_block_order"])

    @staticmethod
    def _word(
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        block: int,
        line: int,
        word: int,
        paragraph: int = 1,
    ) -> dict:
        return {
            "text": text,
            "bbox": [x0, y0, x1, y1],
            "source_tsv": {
                "page": 1,
                "block": block,
                "paragraph": paragraph,
                "line": line,
                "word": word,
            },
        }

    def _aligned_table_words(self) -> list[dict]:
        words: list[dict] = []
        for line in range(1, 6):
            y0 = 100 + (line - 1) * 60
            words.extend(
                [
                    self._word(f"Item{line}", 80, y0, 150, y0 + 20, block=1, line=line, word=1),
                    self._word(f"Value{line}", 520, y0, 600, y0 + 20, block=1, line=line, word=2),
                ]
            )
        return words

    def test_structure_gate_detects_repeated_aligned_columns(self) -> None:
        gate = tesseract_structure_gate(
            self._aligned_table_words(),
            {"raster_width": 800, "raster_height": 1_000},
        )
        self.assertTrue(gate["escalate"])
        self.assertIn("repeated_aligned_columns", gate["reasons"])
        self.assertEqual(gate["signals"]["aligned_columns"][0]["line_count"], 5)

    def test_structure_gate_ignores_wrapped_prose(self) -> None:
        words: list[dict] = []
        for line in range(1, 6):
            y0 = 100 + (line - 1) * 60
            for word in range(1, 7):
                x0 = 80 + (word - 1) * 45
                words.append(
                    self._word(
                        f"word{word}",
                        x0,
                        y0,
                        x0 + 35,
                        y0 + 20,
                        block=1,
                        line=line,
                        word=word,
                    )
                )
        gate = tesseract_structure_gate(words, {"raster_width": 800, "raster_height": 1_000})
        self.assertFalse(gate["escalate"])
        self.assertEqual(gate["reasons"], [])

    def test_structure_gate_detects_two_substantial_text_columns(self) -> None:
        words: list[dict] = []
        for block, start_x in ((1, 50), (2, 500)):
            for line in range(1, 6):
                y0 = 50 + (line - 1) * 80
                for word in range(1, 9):
                    x0 = start_x + (word - 1) * 24
                    words.append(
                        self._word(
                            f"b{block}w{word}",
                            x0,
                            y0,
                            x0 + 18,
                            y0 + 20,
                            block=block,
                            line=line,
                            word=word,
                        )
                    )
        gate = tesseract_structure_gate(words, {"raster_width": 800, "raster_height": 1_000})
        self.assertTrue(gate["escalate"])
        self.assertIn("multiple_text_columns", gate["reasons"])
        self.assertEqual(len(gate["signals"]["multi_column_pairs"]), 1)

    def test_structure_gate_ignores_small_side_by_side_labels(self) -> None:
        words: list[dict] = []
        for block, start_x in ((1, 50), (2, 500)):
            for line in range(1, 3):
                for word in range(1, 3):
                    x0 = start_x + (word - 1) * 30
                    words.append(
                        self._word(
                            f"b{block}w{word}",
                            x0,
                            100 + line * 30,
                            x0 + 20,
                            120 + line * 30,
                            block=block,
                            line=line,
                            word=word,
                        )
                    )
        gate = tesseract_structure_gate(words, {"raster_width": 800, "raster_height": 1_000})
        self.assertFalse(gate["escalate"])

    def test_structure_escalation_makes_surya_authoritative_and_retains_tesseract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "structured.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            quality = {
                "accepted": True,
                "mean_word_confidence": 95.0,
                "rejection_reasons": [],
            }
            table_html = (
                "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>10</td></tr></table>"
            )
            table = html_to_table(table_html)
            assert table is not None
            prediction = {
                "page-000001.png": {
                    "text": "Item Value\nA 10",
                    "blocks": [
                        {
                            "block_type": "Table",
                            "type": "table",
                            "text": "Item Value\nA 10",
                            "html": table_html,
                            "table": table,
                            "bbox": [80, 100, 720, 700],
                            "polygon": [[80, 100], [720, 100], [720, 700], [80, 700]],
                        }
                    ],
                },
            }
            with (
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    return_value=(
                        "Item1 Value1 Item2 Value2",
                        quality,
                        self._aligned_table_words(),
                        [],
                    ),
                ),
                patch("pdf_pipeline.surya_batch", return_value=prediction),
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(structure_aware=True),
                )

            self.assertEqual(summary["surya_escalated_pages"], 1)
            self.assertEqual(summary["surya_quality_escalated_pages"], 0)
            self.assertEqual(summary["surya_structure_escalated_pages"], 1)
            job_dir = root / "output" / "structured"
            record = json.loads((job_dir / "pages.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(record["engine"], "surya")
            self.assertEqual(record["outcome"], "surya_escalated")
            self.assertEqual(record["escalation_reason"], "structure")
            self.assertTrue(record["tesseract_quality"]["accepted"])
            self.assertEqual(record["tesseract_candidate"]["text"], "Item1 Value1 Item2 Value2")
            self.assertTrue(record["structure_gate"]["escalate"])
            rich = json.loads((job_dir / "structured_rich.json").read_text(encoding="utf-8"))
            rich_page = rich["pages"][0]
            self.assertFalse(rich_page["layers"]["tesseract5"]["selected"])
            self.assertEqual(rich_page["routing"]["escalation_reason"], "structure")
            self.assertTrue(rich_page["routing"]["structure_gate"]["escalate"])
            self.assertTrue(
                rich_page["layers"]["surya"]["blocks"][0]["table"]["rows"][0][0]["is_header"]
            )

    def test_structure_routing_is_opt_in_and_default_tesseract_route_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "plain.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            quality = {"accepted": True, "mean_word_confidence": 95.0, "rejection_reasons": []}
            with (
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    return_value=(
                        "Item1 Value1 Item2 Value2",
                        quality,
                        self._aligned_table_words(),
                        [],
                    ),
                ),
                patch(
                    "pdf_pipeline.surya_batch",
                    side_effect=AssertionError("default route must not call Surya"),
                ),
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(),
                )
            self.assertEqual(summary["tesseract_accepted_pages"], 1)
            self.assertEqual(summary["surya_escalated_pages"], 0)
            record = json.loads(
                (root / "output" / "plain" / "pages.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(record["engine"], "tesseract5")
            self.assertNotIn("structure_gate", record)

    def test_primary_surya_bypasses_tesseract_and_is_selected_in_rich_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "hindi.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()

            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            prediction = {
                "page-000001.png": {
                    "text": "भारत सरकार",
                    "blocks": [
                        {
                            "block_type": "Text",
                            "type": "text",
                            "text": "भारत सरकार",
                            "bbox": [80, 100, 480, 160],
                            "reading_order": 1,
                        }
                    ],
                },
            }
            with (
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    side_effect=AssertionError("primary Surya route must bypass Tesseract"),
                ),
                patch("pdf_pipeline.surya_batch", return_value=prediction) as surya_batch,
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(language="hin"),
                )

            self.assertEqual(summary["surya_primary_pages"], 1)
            self.assertEqual(summary["surya_escalated_pages"], 0)
            self.assertEqual(summary["tesseract_accepted_pages"], 0)
            surya_batch.assert_called_once()

            job_dir = root / "output" / "hindi"
            record = json.loads((job_dir / "pages.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(record["engine"], "surya")
            self.assertEqual(record["outcome"], "surya_primary")
            self.assertEqual(record["route"], "surya")
            self.assertEqual(record["classifier_route"], "tesseract")
            self.assertEqual(record["escalation_reason"], "primary")
            self.assertNotIn("tesseract_quality", record)
            self.assertNotIn("tesseract_candidate", record)

            rich = json.loads((job_dir / "hindi_rich.json").read_text(encoding="utf-8"))
            rich_page = rich["pages"][0]
            self.assertEqual(rich_page["authoritative"]["engine"], "surya")
            self.assertFalse(rich_page["layers"]["tesseract5"]["attempted"])
            self.assertTrue(rich_page["layers"]["surya"]["attempted"])
            self.assertTrue(rich_page["layers"]["surya"]["selected"])
            self.assertEqual(rich_page["routing"]["route"], "surya")
            self.assertEqual(rich_page["routing"]["classifier_route"], "tesseract")
            self.assertEqual(rich_page["routing"]["escalation_reason"], "primary")

    def test_forced_surya_for_english_has_generic_primary_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "english.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()

            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            prediction = {
                "page-000001.png": {
                    "text": "Government Gazette",
                    "blocks": [
                        {
                            "block_type": "Text",
                            "type": "text",
                            "text": "Government Gazette",
                            "bbox": [80, 100, 480, 160],
                            "reading_order": 1,
                        }
                    ],
                },
            }
            with (
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    side_effect=AssertionError("forced Surya route must bypass Tesseract"),
                ),
                patch("pdf_pipeline.surya_batch", return_value=prediction),
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(language="eng", ocr_engine="surya"),
                )

            self.assertEqual(summary["surya_primary_pages"], 1)
            job_dir = root / "output" / "english"
            normalized = json.loads((job_dir / "english_cascade.json").read_text(encoding="utf-8"))
            rich = json.loads((job_dir / "english_rich.json").read_text(encoding="utf-8"))
            for artifact in (normalized, rich):
                self.assertEqual(
                    artifact["metadata"]["pipeline"], "PyMuPDF -> Surya OCR (primary route)"
                )
                self.assertEqual(artifact["metadata"]["routing"]["primary_ocr_engine"], "surya")

    def test_resumed_primary_surya_uses_next_batch_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "resumed.pdf"
            document = fitz.open()
            document.new_page()
            document.new_page()
            document.save(pdf_path)
            document.close()

            job_dir = root / "output" / "resumed"
            job_dir.mkdir(parents=True)
            existing = {
                "page": 1,
                "status": "complete",
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "engine": "surya",
                "outcome": "surya_primary",
                "signals": {},
                "native_text": "",
                "text": "already completed",
                "surya_batch": 1,
            }
            (job_dir / "pages.jsonl").write_text(
                json.dumps(existing, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            prediction = {
                "page-000002.png": {
                    "text": "नई पंक्ति",
                    "blocks": [
                        {
                            "block_type": "Text",
                            "type": "text",
                            "text": "नई पंक्ति",
                            "bbox": [80, 100, 480, 160],
                            "reading_order": 1,
                        }
                    ],
                },
            }
            with (
                # The pre-written manifest is the resume fixture; its job
                # metadata is irrelevant to verifying the persisted batch
                # counter and would only duplicate check_or_create_job tests.
                patch("pdf_pipeline.check_or_create_job"),
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    side_effect=AssertionError("primary Surya route must bypass Tesseract"),
                ),
                patch("pdf_pipeline.surya_batch", return_value=prediction) as surya_batch,
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(language="hin"),
                )

            self.assertEqual(summary["surya_primary_pages"], 2)
            surya_batch.assert_called_once()
            self.assertEqual(
                surya_batch.call_args.args[1],
                job_dir / "surya" / "batch-000002",
            )
            self.assertEqual(surya_batch.call_args.args[0].name, "surya-input-000002")

            records = {
                record["page"]: record
                for record in (
                    json.loads(line)
                    for line in (job_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(records[1]["surya_batch"], 1)
            self.assertEqual(records[2]["surya_batch"], 2)

    def test_compact_only_mode_preserves_rejected_tesseract_as_unselected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "compact.pdf"
            document = fitz.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            raster = {
                "raster_width": 800,
                "raster_height": 1_000,
                "pdf_bbox": [0, 0, 595, 842],
                "dpi": 200,
            }
            inspection = {
                "classification": "OCR_NEEDED",
                "route": "tesseract",
                "signals": {},
                "native_text": "",
            }
            quality = {
                "accepted": False,
                "mean_word_confidence": 21.0,
                "rejection_reasons": ["low_mean_confidence"],
            }
            with (
                patch("pdf_pipeline.inspect_page", return_value=inspection),
                patch("pdf_pipeline.render_page", return_value=raster),
                patch(
                    "pdf_pipeline.tesseract_page",
                    return_value=("unreliable Hindi OCR", quality, [], []),
                ) as tesseract_page_mock,
                patch(
                    "pdf_pipeline.surya_batch",
                    side_effect=AssertionError("compact-only mode must not call Surya"),
                ),
            ):
                summary = process_document(
                    pdf_path,
                    input_root=root,
                    output_root=root / "output",
                    config=PipelineConfig(language="hin", fallback_engine="none"),
                )

            self.assertEqual(summary["tesseract_rejected_no_fallback_pages"], 1)
            self.assertEqual(summary["surya_escalated_pages"], 0)
            tesseract_page_mock.assert_called_once()
            job_dir = root / "output" / "compact"
            record = json.loads((job_dir / "pages.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(record["engine"], "none")
            self.assertEqual(record["outcome"], "tesseract_rejected_no_fallback")
            self.assertEqual(record["text"], "")
            self.assertEqual(record["tesseract_candidate"]["text"], "unreliable Hindi OCR")
            self.assertNotIn("surya_batch", record)
            self.assertFalse((job_dir / "surya").exists())
            rich = json.loads((job_dir / "compact_rich.json").read_text(encoding="utf-8"))
            self.assertEqual(rich["pages"][0]["authoritative"]["engine"], "none")
            self.assertEqual(rich["pages"][0]["authoritative"]["text"], "")
            self.assertFalse(rich["pages"][0]["layers"]["tesseract5"]["selected"])
            self.assertFalse(rich["pages"][0]["layers"]["surya"]["attempted"])

    def test_compact_only_cli_profile_disallows_structure_escalation(self) -> None:
        args = build_parser().parse_args(
            [
                "fixture.pdf",
                "--language",
                "script/Devanagari",
                "--ocr-engine",
                "tesseract",
                "--fallback-engine",
                "none",
            ]
        )
        config = config_from_args(args)
        self.assertEqual(config.language, "script/Devanagari")
        self.assertEqual(config.ocr_engine, "tesseract")
        self.assertEqual(config.fallback_engine, "none")
        self.assertEqual(primary_ocr_engine(config), "tesseract")

        incompatible = build_parser().parse_args(
            ["fixture.pdf", "--fallback-engine", "none", "--structure-aware"]
        )
        with self.assertRaisesRegex(SystemExit, "structure-aware requires --fallback-engine surya"):
            # ``main`` validates after parsing; preserve that contract here.
            validate_args(incompatible)

    def test_structure_aware_rejects_primary_surya_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "structure_aware"):
            validate_pipeline_config(PipelineConfig(language="hin", structure_aware=True))

    def test_visual_only_surya_page_is_a_valid_quality_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "page-000001.png"
            pending = [
                _PendingSuryaPage(
                    page=1,
                    inspection={
                        "classification": "SCANNED",
                        "route": "tesseract",
                        "signals": {},
                        "native_text": "",
                    },
                    image_path=image_path,
                    quality={"accepted": False, "mean_word_confidence": 10.0},
                    raster={
                        "raster_width": 800,
                        "raster_height": 1_000,
                        "pdf_bbox": [0, 0, 595, 842],
                    },
                    tesseract_text="",
                    tesseract_words=[],
                    tesseract_blocks=[],
                    escalation_reason="quality",
                )
            ]
            visual_prediction = {
                "page-000001.png": {
                    "text": "",
                    "blocks": [
                        {
                            "block_type": "Figure",
                            "type": "figure",
                            "text": "",
                            "retain_empty": True,
                            "skipped": True,
                            "bbox": [100, 120, 700, 760],
                        }
                    ],
                },
            }
            records: dict[int, dict] = {}
            with patch("pdf_pipeline.surya_batch", return_value=visual_prediction):
                flush_surya(
                    pending,
                    batch_number=1,
                    job_dir=root,
                    config=self.config,
                    manifest_path=root / "pages.jsonl",
                    records=records,
                )
            self.assertEqual(records[1]["text"], "")
            self.assertEqual(records[1]["blocks"][0]["type"], "figure")
            summary = summarise_document(root / "source.pdf", 1, records)
            self.assertEqual(summary["surya_escalated_pages"], 1)
            self.assertEqual(summary["surya_quality_escalated_pages"], 1)
            self.assertEqual(summary["surya_structure_escalated_pages"], 0)

    def test_blank_surya_page_is_a_valid_quality_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pending = [
                _PendingSuryaPage(
                    page=1,
                    inspection={
                        "classification": "SCANNED",
                        "route": "tesseract",
                        "signals": {},
                        "native_text": "",
                    },
                    image_path=root / "page-000001.png",
                    quality={"accepted": False, "mean_word_confidence": 10.0},
                    raster={
                        "raster_width": 800,
                        "raster_height": 1_000,
                        "pdf_bbox": [0, 0, 595, 842],
                    },
                    tesseract_text="",
                    tesseract_words=[],
                    tesseract_blocks=[],
                    escalation_reason="quality",
                )
            ]
            records: dict[int, dict] = {}
            with patch(
                "pdf_pipeline.surya_batch",
                return_value={"page-000001.png": {"text": "", "blocks": []}},
            ):
                flush_surya(
                    pending,
                    batch_number=1,
                    job_dir=root,
                    config=self.config,
                    manifest_path=root / "pages.jsonl",
                    records=records,
                )
            self.assertEqual(records[1]["engine"], "surya")
            self.assertEqual(records[1]["text"], "")
            self.assertNotIn("blocks", records[1])


class ChandraAdapterUnitTests(unittest.TestCase):
    def test_html_page_and_block_text_are_normalized(self) -> None:
        result = {
            "json": {
                "children": [
                    {
                        "block_type": "Page",
                        "bbox": [0, 0, 600, 800],
                        "children": [
                            {
                                "block_type": "SectionHeader",
                                "bbox": [10, 10, 400, 30],
                                "html": "<h1>101 INTRODUCTION</h1>",
                            },
                            {
                                "block_type": "Text",
                                "bbox": [10, 40, 500, 80],
                                "html": "<p>Useful <b>body</b> text.</p>",
                            },
                            {
                                "block_type": "Table",
                                "bbox": [10, 90, 500, 150],
                                "html": "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>10</td></tr></table>",
                            },
                        ],
                    }
                ]
            }
        }

        pages = normalize_chandra(result)

        self.assertEqual(len(pages), 1)
        self.assertIn("101 INTRODUCTION", pages[0]["text"])
        self.assertIn("Useful body text.", pages[0]["text"])
        self.assertIn("Item Value", pages[0]["text"])
        self.assertEqual([block["reading_order"] for block in pages[0]["blocks"]], [1, 2, 3])
        self.assertEqual(pages[0]["blocks"][2]["type"], "table")
        self.assertIn("<table>", pages[0]["blocks"][2]["html"])

    def test_html_is_used_when_page_has_no_child_blocks(self) -> None:
        page = {"html": "<p>Fallback page text &amp; symbols.</p>"}
        self.assertEqual(first_text(page), "Fallback page text & symbols.")

    def test_saved_response_can_be_normalized_without_api_or_timing_output(self) -> None:
        result = {
            "status": "complete",
            "success": True,
            "runtime": 1.25,
            "markdown": "# Saved markdown\n",
            "json": {
                "children": [
                    {
                        "block_type": "Page",
                        "children": [{"block_type": "Text", "html": "<p>Offline text.</p>"}],
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_response = root / "report_datalab_response.json"
            raw_response.write_text(
                json.dumps({"submission": {"request_id": "saved"}, "result": result}),
                encoding="utf-8",
            )
            output_dir = root / "normalized"

            # Offline mode must not require credentials or construct a client.
            with (
                patch("run_chandra.require_api_key", side_effect=AssertionError("API key lookup")),
                patch("run_chandra.request_session", side_effect=AssertionError("network client")),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_chandra.py",
                        "--from-response",
                        str(raw_response),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
            ):
                self.assertEqual(chandra_main(), 0)

            normalized_path = output_dir / "report_chandra.json"
            normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
            self.assertEqual(normalized["source"], "report.pdf")
            self.assertEqual(normalized["pages"][0]["text"], "Offline text.")
            self.assertFalse((output_dir / "report_datalab_response.json").exists())
            self.assertFalse((output_dir / "report_timing.json").exists())
            self.assertFalse((output_dir / "report_chandra.md").exists())

            with patch.object(
                sys,
                "argv",
                [
                    "run_chandra.py",
                    "--from-response",
                    str(raw_response),
                    "--output-dir",
                    str(output_dir),
                    "--write-markdown",
                ],
            ):
                self.assertEqual(chandra_main(), 0)
            self.assertEqual(
                (output_dir / "report_chandra.md").read_text(encoding="utf-8"),
                "# Saved markdown\n",
            )


class CascadeEvaluationUnitTests(unittest.TestCase):
    def test_structure_scores_do_not_count_unrelated_same_type_blocks(self) -> None:
        reference_blocks = [
            {
                "type": "paragraph",
                "text": "First unique reference paragraph",
                "bbox": [0, 0, 100, 20],
            },
            {
                "type": "paragraph",
                "text": "Second unique reference paragraph",
                "bbox": [0, 30, 100, 50],
            },
        ]
        ocr_blocks = [
            {
                "label": "Text",
                "text": "Completely unrelated OCR noise",
                "bbox": [200, 0, 300, 20],
            },
            {
                "label": "Text",
                "text": "Another unrelated OCR region",
                "bbox": [200, 30, 300, 50],
            },
        ]

        scores = structure_scores(reference_blocks, ocr_blocks)
        paragraph = scores["per_type"]["paragraph"]

        # The old count-only metric would have reported F1=1.0 here.
        self.assertEqual(paragraph["matched"], 0)
        self.assertEqual(paragraph["precision"], 0.0)
        self.assertEqual(paragraph["recall"], 0.0)
        self.assertEqual(paragraph["f1"], 0.0)
        self.assertEqual(scores["matching"]["reference_match_coverage"], 0.0)
        self.assertEqual(scores["matching"]["ocr_match_coverage"], 0.0)

    def test_structure_scores_accept_text_or_geometry_evidence(self) -> None:
        page_geometry = {
            "coordinate_space": "pdf_points",
            "coordinate_frame": "pdf_page_points_top_left_unrotated",
            "page_bbox": [0, 0, 300, 100],
        }
        reference_blocks = [
            {
                "type": "section_heading",
                "text": "Correct heading",
                "bbox": [0, 0, 100, 20],
                "geometry": dict(page_geometry),
            },
            {
                "type": "paragraph",
                "text": "Text damaged in OCR",
                "bbox": [0, 30, 100, 50],
                "geometry": dict(page_geometry),
            },
        ]
        ocr_blocks = [
            {
                "label": "SectionHeader",
                "text": "Correct heading",
                "bbox": [200, 0, 300, 20],
                "geometry": dict(page_geometry),
            },
            {
                "label": "Text",
                "text": "garbled",
                "bbox": [0, 30, 100, 50],
                "geometry": dict(page_geometry),
            },
        ]

        scores = structure_scores(reference_blocks, ocr_blocks)
        matches = scores["matching"]["matches"]

        self.assertEqual(scores["per_type"]["section_heading"]["f1"], 1.0)
        self.assertEqual(scores["per_type"]["paragraph"]["f1"], 1.0)
        self.assertEqual(matches[0]["match_basis"], ["text_token_jaccard"])
        self.assertEqual(matches[1]["match_basis"], ["bbox_iou"])
        self.assertEqual(scores["matching"]["matched_blocks"], 2)

    def test_structure_scores_do_not_use_unlabeled_or_incompatible_geometry(self) -> None:
        reference = {
            "type": "paragraph",
            "text": "reference-only alpha",
            "bbox": [0, 0, 100, 20],
            "geometry": {
                "coordinate_space": "pdf_points",
                "coordinate_frame": "pdf_page_points_top_left_unrotated",
                "page_bbox": [0, 0, 300, 100],
            },
        }
        ocr = {
            "label": "Text",
            "text": "unrelated OCR beta",
            "bbox": [0, 0, 100, 20],
            "geometry": {
                "coordinate_space": "raster_pixels",
                "coordinate_frame": "surya_raster",
                "page_bbox": [0, 0, 300, 100],
            },
        }

        compatible, status = geometry_compatibility(reference, ocr)
        scores = structure_scores([reference], [ocr])

        self.assertFalse(compatible)
        self.assertEqual(status, "non_normalized_coordinate_space")
        self.assertEqual(scores["per_type"]["paragraph"]["f1"], 0.0)

    def test_structure_matching_maximizes_cardinality_before_evidence(self) -> None:
        # R0 has a perfect match with O0, but then R1 has no remaining match.
        # A maximum-cardinality matcher must instead use R0->O1 and R1->O0.
        reference_blocks = [
            {"type": "paragraph", "text": "a b c d e f g h i j"},
            {"type": "paragraph", "text": "a b c"},
        ]
        ocr_blocks = [
            {"label": "Text", "text": "a b c d e f g h i j"},
            {"label": "Text", "text": "d e f g h i j"},
        ]

        scores = structure_scores(reference_blocks, ocr_blocks)
        matches = scores["matching"]["matches"]

        self.assertEqual(scores["per_type"]["paragraph"]["matched"], 2)
        self.assertEqual(scores["per_type"]["paragraph"]["f1"], 1.0)
        self.assertEqual(
            {(match["reference_block_index"], match["ocr_block_index"]) for match in matches},
            {(0, 1), (1, 0)},
        )

    def test_reading_order_marks_low_match_coverage_inconclusive(self) -> None:
        reference_blocks = [
            {
                "type": "paragraph",
                "text": text,
                "bbox": [0, index * 20, 100, index * 20 + 10],
            }
            for index, text in enumerate(("alpha", "bravo", "charlie", "delta", "echo"))
        ]
        ocr_blocks = [
            {
                "label": "Text",
                "text": text,
                "bbox": [0, index * 20, 100, index * 20 + 10],
            }
            for index, text in enumerate(("alpha", "bravo"))
        ]

        details = order_similarity_details(reference_blocks, ocr_blocks)
        legacy_rate, legacy_matched, legacy_inversions = order_similarity(
            reference_blocks,
            ocr_blocks,
        )

        self.assertEqual(details["status"], "inconclusive")
        self.assertIn("low_reference_match_coverage", details["inconclusive_reasons"])
        self.assertEqual(details["reference_match_coverage"], 0.4)
        self.assertEqual(details["ocr_match_coverage"], 1.0)
        self.assertIsNone(details["inversion_rate"])
        self.assertEqual(details["observed_inversion_rate"], 0.0)
        # Existing callers can still use the historical tuple for diagnostics.
        self.assertEqual((legacy_rate, legacy_matched, legacy_inversions), (0.0, 2, 0))

    def test_reading_order_uses_declared_order_not_input_array_order(self) -> None:
        # The reference array is intentionally reversed. Explicit orders are
        # the source of truth, so this is still a correctly ordered OCR page.
        reference_blocks = [
            {"type": "paragraph", "text": "second", "reading_order": 2},
            {"type": "paragraph", "text": "first", "reading_order": 1},
        ]
        ocr_blocks = [
            {"label": "Text", "text": "first", "reading_order": 1},
            {"label": "Text", "text": "second", "reading_order": 2},
        ]

        details = order_similarity_details(reference_blocks, ocr_blocks)

        self.assertEqual(details["status"], "measured")
        self.assertEqual(details["inversions"], 0)
        self.assertEqual(details["inversion_rate"], 0.0)
        self.assertEqual(
            [match["reference_reading_order"] for match in details["matches"]],
            [1.0, 2.0],
        )

    def test_rich_cascade_uses_authoritative_blocks_and_semantic_block_types(self) -> None:
        rich_page = {
            "page": 1,
            "authoritative": {
                "engine": "surya",
                "text": "Heading Body Item Value",
                "blocks": [
                    {
                        "block_type": "Table",
                        "type": "text",
                        "text": "Item Value",
                        "reading_order": 3,
                    },
                    {
                        "block_type": "Section-Header",
                        "type": "text",
                        "text": "Heading",
                        "reading_order": 1,
                    },
                    {
                        "block_type": "Text",
                        "type": "text",
                        "text": "Body",
                        "reading_order": 2,
                    },
                    {
                        "block_type": "Figure",
                        "type": "text",
                        "text": "Do not score this visual description",
                        "reading_order": 4,
                    },
                    {
                        "block_type": "Chart",
                        "type": "text",
                        "text": "Do not score this chart description",
                        "reading_order": 5,
                    },
                    {
                        "block_type": "Image",
                        "type": "text",
                        "text": "Do not score this image description",
                        "reading_order": 6,
                    },
                ],
            },
            "layers": {
                "pymupdf": {"text": "DO NOT SCORE native layer"},
                "tesseract5": {"text": "DO NOT SCORE rejected layer"},
                "surya": {"text": "DO NOT SCORE duplicate layer"},
            },
        }

        blocks = page_ocr_blocks(rich_page, "cascade")

        self.assertEqual([block["text"] for block in blocks], ["Heading", "Body", "Item Value"])
        self.assertEqual(
            [canonical_label(block["label"]) for block in blocks],
            ["section_heading", "paragraph", "table"],
        )
        self.assertEqual(
            visual_description_labels(rich_page, "cascade"), ["figure", "chart", "image"]
        )
        only_visual = {
            "authoritative": {
                "text": "Do not fall back to this visual description",
                "blocks": [{"block_type": "Diagram", "text": "A diagram description"}],
            },
        }
        self.assertEqual(page_ocr_blocks(only_visual, "cascade"), [])
        self.assertEqual(
            page_ocr_blocks(
                {
                    "text": "Also exclude generic visual descriptions",
                    "blocks": [{"block_type": "Picture", "text": "image"}],
                },
                "chandra",
            ),
            [],
        )
        self.assertEqual(
            page_ocr_blocks(
                {
                    "authoritative": {"text": "Selected fallback", "blocks": []},
                    "layers": {"pymupdf": {"text": "wrong"}},
                },
                "cascade",
            ),
            [
                {
                    "text": "Selected fallback",
                    "label": None,
                    "bbox": None,
                    "reading_order": 1,
                    "geometry": None,
                }
            ],
        )

    def test_evaluator_joins_by_page_id_and_rejects_mismatches(self) -> None:
        reference = [{"page": 1}, {"page": 2}]
        structure = [{"page": 2}, {"page": 1}]
        ocr = [{"page": 2}, {"page": 1}]
        self.assertEqual(
            [row[0] for row in aligned_pages(reference, structure, ocr)],
            [1, 2],
        )
        with self.assertRaisesRegex(
            ValueError, r"OCR page IDs do not match reference: missing \[2\], unexpected \[3\]"
        ):
            aligned_pages(reference, structure, [{"page": 1}, {"page": 3}])

    def test_cascade_cli_handles_normalized_and_rich_artifacts(self) -> None:
        structure_page = {
            "page": 1,
            "blocks": [
                {"type": "page_header", "text": "Header"},
                {"type": "section_heading", "text": "Heading"},
                {"type": "paragraph", "text": "Body"},
                {"type": "table", "text": "Item Value"},
            ],
        }
        authoritative_blocks = [
            {"block_type": "PageHeader", "type": "text", "text": "Header", "reading_order": 1},
            {"block_type": "SectionHeader", "type": "text", "text": "Heading", "reading_order": 2},
            {"block_type": "Text", "type": "text", "text": "Body", "reading_order": 3},
            {"block_type": "Table", "type": "text", "text": "Item Value", "reading_order": 4},
            {
                "block_type": "Picture",
                "type": "text",
                "text": "Not in the silver text",
                "reading_order": 5,
            },
        ]
        rich = {
            "schema_version": "cascade-ocr/rich-v1",
            "engine": "cascade",
            "pages": [
                {
                    "page": 1,
                    "authoritative": {
                        "engine": "surya",
                        "text": "Header Heading Body Item Value",
                        "blocks": authoritative_blocks,
                    },
                    "layers": {
                        "pymupdf": {"text": "unselected native text"},
                        "tesseract5": {"text": "unselected OCR text"},
                    },
                }
            ],
        }
        normalized = {
            "schema_version": "cascade-ocr/v1",
            "engine": "cascade",
            "pages": [{"page": 2, "text": "second"}, {"page": 1, "blocks": authoritative_blocks}],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            structure_path = root / "structure.json"
            rich_path = root / "document_rich.json"
            normalized_path = root / "document_cascade.json"
            output_dir = root / "evaluation"
            reference_path.write_text(
                json.dumps({"pages": [{"page": 1, "text": "unused"}]}), encoding="utf-8"
            )
            structure_path.write_text(json.dumps({"pages": [structure_page]}), encoding="utf-8")
            rich_path.write_text(json.dumps(rich), encoding="utf-8")
            normalized_path.write_text(json.dumps(normalized), encoding="utf-8")

            # Normalized output has direct page blocks, and page numbers are
            # aligned before evaluation rather than relying on JSON order.
            normalized_pages = load_cascade(normalized_path)
            self.assertEqual([page["page"] for page in normalized_pages], [1, 2])
            self.assertEqual(
                [block["text"] for block in page_ocr_blocks(normalized_pages[0], "cascade")],
                ["Header", "Heading", "Body", "Item Value"],
            )

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
                        "cascade",
                        "--ocr-json",
                        str(rich_path),
                        "--out-dir",
                        str(output_dir),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertIsNone(evaluate_ocr_main())

            evaluation = json.loads(
                (output_dir / "cascade_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation["aggregate"]["weighted_CER"], 0.0)
            self.assertEqual(evaluation["aggregate"]["weighted_WER"], 0.0)
            self.assertEqual(
                evaluation["aggregate"]["method"],
                "exact per-page Levenshtein distances / total reference units",
            )
            self.assertEqual(
                evaluation["visual_description_blocks"],
                {
                    "text_scoring": "excluded",
                    "count": 1,
                    "by_type": {"picture": 1},
                },
            )
            self.assertEqual(evaluation["structure"][0]["per_type"]["table"]["f1"], 1.0)
            self.assertEqual(evaluation["structure"][0]["matching"]["matched_blocks"], 4)
            self.assertEqual(evaluation["reading_order"]["per_page"][0]["status"], "measured")
            self.assertEqual(
                evaluation["reading_order"]["per_page"][0]["reference_match_coverage"],
                1.0,
            )

    def test_rich_cascade_missing_authoritative_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid_rich.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "cascade-ocr/rich-v1",
                        "pages": [{"page": 1, "layers": {"pymupdf": {"text": "must not score"}}}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing authoritative"):
                load_cascade(path)


if __name__ == "__main__":
    unittest.main()
