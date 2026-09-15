#!/usr/bin/env python3
"""Focused regression coverage for Poppler reference-coordinate normalization."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf as fitz

from create_ground_truth import (
    PDF_POINT_FRAME,
    POPPLER_DISPLAY_FRAME,
    parse_pdf,
    pymupdf_unrotated_page_bbox,
)


@unittest.skipUnless(shutil.which("pdftotext"), "pdftotext is required for ground-truth coordinate tests")
class PopplerCoordinateNormalizationTests(unittest.TestCase):
    def test_rotated_poppler_words_align_with_pymupdf_unrotated_words(self) -> None:
        """Right-angle rotations must not leave reference boxes in display space."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "rotations.pdf"
            document = fitz.open()
            for rotation in (0, 90, 180, 270):
                page = document.new_page(width=200, height=100)
                page.insert_text((20, 30), f"ALPHA{rotation}")
                page.insert_text((125, 80), f"OMEGA{rotation}")
                page.set_rotation(rotation)
            document.save(pdf_path)
            document.close()

            reference_pages = parse_pdf(pdf_path, root / "layout.html")
            self.assertEqual(len(reference_pages), 4)

            document = fitz.open(pdf_path)
            try:
                for page_number, (reference, page) in enumerate(zip(reference_pages, document), start=1):
                    expected_bbox = pymupdf_unrotated_page_bbox(page)
                    self.assertEqual(reference["coordinate_frame"], PDF_POINT_FRAME)
                    self.assertEqual(reference["bbox"], expected_bbox)
                    self.assertEqual(reference["width"], 200.0)
                    self.assertEqual(reference["height"], 100.0)

                    native_words = {
                        word[4]: word[:4]
                        for word in page.get_text("words")
                        if word[4] in {f"ALPHA{page.rotation}", f"OMEGA{page.rotation}"}
                    }
                    reference_words = {word["text"]: word for word in reference["words"]}
                    self.assertEqual(set(native_words), set(reference_words))

                    for text, native_bbox in native_words.items():
                        reference_word = reference_words[text]
                        self.assertEqual(reference_word["coordinate_frame"], PDF_POINT_FRAME)
                        self.assertEqual(reference_word["source_coordinate_frame"], POPPLER_DISPLAY_FRAME)
                        self.assertEqual(len(reference_word["source_bbox"]), 4)
                        reference_center = (
                            (reference_word["x0"] + reference_word["x1"]) / 2,
                            (reference_word["y0"] + reference_word["y1"]) / 2,
                        )
                        native_center = (
                            (native_bbox[0] + native_bbox[2]) / 2,
                            (native_bbox[1] + native_bbox[3]) / 2,
                        )
                        # Poppler and MuPDF have slightly different glyph
                        # ascent/descent metrics, but a correctly derotated
                        # word remains within a few PDF points of its native
                        # PyMuPDF counterpart on both axes.
                        self.assertAlmostEqual(reference_center[0], native_center[0], delta=4.0)
                        self.assertAlmostEqual(reference_center[1], native_center[1], delta=4.0)
            finally:
                document.close()

    def test_rotated_cropbox_uses_visible_page_and_unrotated_output_frames(self) -> None:
        """Poppler's reported dimensions must not rescale a rotated CropBox."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "cropped-rotation.pdf"
            document = fitz.open()
            page = document.new_page(width=200, height=100)
            page.insert_text((30, 30), "CROP90")
            page.insert_text((120, 70), "TAIL90")
            page.set_cropbox(fitz.Rect(10, 10, 190, 90))
            page.set_rotation(90)
            document.save(pdf_path)
            document.close()

            reference = parse_pdf(pdf_path, root / "layout.html")[0]
            self.assertEqual(reference["bbox"], [0.0, 0.0, 180.0, 80.0])
            self.assertEqual(reference["poppler_display_bbox"], [0.0, 0.0, 80.0, 180.0])
            # Poppler reports the unrotated CropBox dimensions in the page
            # element even though its word bboxes are already rotated.
            self.assertEqual(reference["poppler_reported_page_bbox"], [0.0, 0.0, 180.0, 80.0])

            document = fitz.open(pdf_path)
            try:
                page = document[0]
                native_words = {word[4]: word[:4] for word in page.get_text("words")}
                reference_words = {word["text"]: word for word in reference["words"]}
                for text in ("CROP90", "TAIL90"):
                    native_bbox = native_words[text]
                    reference_word = reference_words[text]
                    self.assertAlmostEqual(
                        (reference_word["x0"] + reference_word["x1"]) / 2,
                        (native_bbox[0] + native_bbox[2]) / 2,
                        delta=4.0,
                    )
                    self.assertAlmostEqual(
                        (reference_word["y0"] + reference_word["y1"]) / 2,
                        (native_bbox[1] + native_bbox[3]) / 2,
                        delta=4.0,
                    )
            finally:
                document.close()


if __name__ == "__main__":
    unittest.main()
