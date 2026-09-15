#!/usr/bin/env python3
"""Resumable, cost-aware OCR cascade for one PDF or a directory of PDFs.

Pipeline: PyMuPDF native text -> Tesseract 5 -> quality / optional structure
gate -> Surya fallback.

Only the current page and one bounded Surya fallback batch are rendered.  Every
completed page is appended to a manifest, so a failed 500-page job resumes
without repeating completed work.

Examples:
  python pdf_pipeline.py incoming/ --dry-run
  python pdf_pipeline.py report.pdf --output-dir artifacts/production
  python pdf_pipeline.py incoming/ --shard-count 4 --shard-index 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterator, Sequence

try:
    import pymupdf as fitz
except ImportError as exc:
    raise SystemExit("PyMuPDF is required: ./sttl/bin/python -m pip install PyMuPDF") from exc


PIPELINE_VERSION = "1.4"
RICH_SCHEMA_VERSION = "cascade-ocr/rich-v1"
COMPLETE = "complete"
PAGE_STATES = ("DIGITAL", "SCANNED", "MIXED", "OCR_NEEDED")
VISUAL_BLOCK_TYPES = frozenset({"chart", "diagram", "figure", "image", "picture"})

# These geometry thresholds are intentionally expressed as page-relative
# values.  Tesseract receives a rendered image, so absolute pixels would make
# a policy tuned at 200 DPI behave differently at 150 or 300 DPI.
STRUCTURE_GAP_RATIO = 0.015
STRUCTURE_CLUSTER_TOLERANCE_RATIO = 0.02
STRUCTURE_MIN_ALIGNED_LINES = 4
STRUCTURE_MIN_ALIGNED_START_RATIO = 0.30
STRUCTURE_MIN_VERTICAL_SPAN_RATIO = 0.04
STRUCTURE_MIN_COLUMN_LINES = 5
STRUCTURE_MIN_COLUMN_WORDS = 40
STRUCTURE_MIN_COLUMN_GAP_RATIO = 0.08
STRUCTURE_MIN_COLUMN_WIDTH_RATIO = 0.20
STRUCTURE_MAX_COLUMN_WIDTH_RATIO = 0.60
STRUCTURE_MIN_COLUMN_HEIGHT_RATIO = 0.25
STRUCTURE_MIN_COLUMN_VERTICAL_OVERLAP = 0.40


@dataclass(frozen=True)
class PipelineConfig:
    """Every setting that can change a routed page's result."""

    language: str = "eng"
    render_dpi: int = 200
    tesseract_psm: int = 3
    tesseract_timeout_seconds: int = 180
    surya_timeout_seconds: int = 1_800
    surya_batch_size: int = 4
    surya_keep_server: bool = False
    # Opt in because a structurally complex page costs a full Surya pass even
    # when its plain Tesseract text is high-confidence.
    structure_aware: bool = False
    min_native_chars: int = 80
    min_native_words: int = 12
    max_native_garbage_ratio: float = 0.05
    dominant_image_ratio: float = 0.55
    min_tesseract_chars: int = 20
    min_tesseract_words: int = 4
    min_mean_confidence: float = 65.0
    min_confident_word_ratio: float = 0.70
    confident_word_threshold: float = 60.0
    max_tesseract_garbage_ratio: float = 0.08
    min_plausible_word_ratio: float = 0.70


@dataclass
class _PendingSuryaPage:
    """One rendered page retained until a bounded Surya batch is flushed."""

    page: int
    inspection: dict[str, Any]
    image_path: Path
    quality: dict[str, Any]
    raster: dict[str, Any]
    tesseract_text: str
    tesseract_words: list[dict[str, Any]]
    tesseract_blocks: list[dict[str, Any]]
    escalation_reason: str
    structure_gate: dict[str, Any] | None = None


class _HTMLTextExtractor(HTMLParser):
    """Dependency-free conversion of Surya block HTML to readable text."""

    BREAKS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    CELLS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BREAKS:
            self.parts.append("\n")
        elif tag.lower() in self.CELLS:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line).strip()


class _HTMLTableExtractor(HTMLParser):
    """Retain Surya table structure without adding an HTML dependency.

    Surya already emits useful table HTML, including ``th``, ``rowspan``, and
    ``colspan``.  Flattening it to text makes downstream table reconstruction
    needlessly lossy, so the rich artifact keeps both the original HTML and a
    compact, JSON-safe cell representation.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._table_depth = 0
        self._table: dict[str, Any] | None = None
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None

    @staticmethod
    def _positive_int(value: str | None) -> int:
        try:
            return max(1, int(value or "1"))
        except ValueError:
            return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table = {"rows": []}
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            attributes = dict(attrs)
            self._cell = {
                "text_parts": [],
                "is_header": tag == "th",
                "rowspan": self._positive_int(attributes.get("rowspan")),
                "colspan": self._positive_int(attributes.get("colspan")),
            }
        elif tag in {"br", "p", "div", "li"} and self._cell is not None:
            self._cell["text_parts"].append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._table_depth == 1 and self._table is not None:
                self.tables.append(self._table)
                self._table = None
                self._row = None
                self._cell = None
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            raw_text = html.unescape("".join(self._cell.pop("text_parts")))
            raw_text = re.sub(r"[ \t\f\v]+", " ", raw_text)
            raw_text = re.sub(r" *\n *", "\n", raw_text).strip()
            self._row.append({"text": raw_text, **self._cell})
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table["rows"].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text_parts"].append(data)


def html_to_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def html_to_table(value: Any) -> dict[str, Any] | None:
    """Return the first table in a Surya HTML block, preserving cell spans."""
    if not isinstance(value, str) or "<table" not in value.lower():
        return None
    parser = _HTMLTableExtractor()
    parser.feed(value)
    parser.close()
    return parser.tables[0] if parser.tables else None


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def garbage_ratio(value: str) -> float:
    characters = [char for char in value if not char.isspace()]
    if not characters:
        return 0.0
    garbage = sum(char == "\ufffd" or not char.isprintable() or ord(char) < 32 for char in characters)
    return garbage / len(characters)


def plausible_word_ratio(value: str) -> float:
    """Language-neutral sanity check; no English dictionary is assumed."""
    tokens = re.findall(r"\S+", value)
    if not tokens:
        return 0.0
    plausible = 0
    for token in tokens:
        symbols = sum(not (char.isalnum() or char in "'_-.,:/()[]{}%+*=#") for char in token)
        if any(char.isalnum() for char in token) and len(token) <= 64 and symbols / len(token) <= 0.30:
            plausible += 1
    return plausible / len(tokens)


def image_signals(page: fitz.Page) -> tuple[int, float]:
    """Cheap, conservative image coverage signal from PyMuPDF metadata."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0, 0.0
    image_count, image_area = 0, 0.0
    for image in page.get_image_info(hashes=False):
        bbox_data = image.get("bbox")
        if not bbox_data:
            continue
        bbox = fitz.Rect(bbox_data) & page.rect
        if bbox.is_empty:
            continue
        image_count += 1
        image_area += bbox.width * bbox.height
    # Images may overlap.  This is an upper-bound signal, not exact coverage.
    return image_count, min(1.0, image_area / page_area)


def classify_page_signals(
    *,
    text: str,
    text_block_count: int,
    image_count: int,
    image_area_ratio: float,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Classify without rendering the page.

    A page with usable native text and a dominant image is MIXED.  It receives
    cheap OCR because native extraction alone can omit content inside the image.
    """
    native_text = normalize_whitespace(text)
    text_chars = len(native_text)
    word_count = len(re.findall(r"\S+", native_text))
    native_garbage = garbage_ratio(native_text)
    usable = (
        text_chars >= config.min_native_chars
        and word_count >= config.min_native_words
        and native_garbage <= config.max_native_garbage_ratio
    )
    image_dominant = image_area_ratio >= config.dominant_image_ratio
    if usable and not image_dominant:
        classification, route = "DIGITAL", "native_text"
    elif usable:
        classification, route = "MIXED", "tesseract"
    elif image_count or image_dominant:
        classification, route = "SCANNED", "tesseract"
    else:
        # This includes vector/outlined pages and broken or sparse text layers.
        classification, route = "OCR_NEEDED", "tesseract"
    return {
        "classification": classification,
        "route": route,
        "native_text": native_text,
        "signals": {
            "native_text_chars": text_chars,
            "native_word_count": word_count,
            "native_text_block_count": text_block_count,
            "native_garbage_ratio": round(native_garbage, 6),
            "image_count": image_count,
            "image_area_ratio": round(image_area_ratio, 6),
        },
    }


def inspect_page(page: fitz.Page, config: PipelineConfig) -> dict[str, Any]:
    text = page.get_text("text")
    blocks = page.get_text("blocks")
    text_block_count = sum(1 for block in blocks if len(block) >= 5 and str(block[4]).strip())
    image_count, image_area_ratio = image_signals(page)
    return classify_page_signals(
        text=text,
        text_block_count=text_block_count,
        image_count=image_count,
        image_area_ratio=image_area_ratio,
        config=config,
    )


def render_page(page: fitz.Page, path: Path, dpi: int) -> dict[str, Any]:
    """Render a page and return the reversible raster-to-PDF transform.

    Tesseract and Surya report pixel coordinates in this PNG.  The exported
    document uses PDF points everywhere, while retaining this transform and
    each source bbox so consumers can return to the original OCR coordinate
    system if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    try:
        pixmap.save(str(path))
        rendered_rect = page.rect
        # PyMuPDF renders using the page's visible rotation, while native text
        # extraction uses its unrotated, crop-local coordinate system.  Store
        # both spaces and the inverse transform so OCR geometry can be mapped
        # into the same space as native spans below.
        target_rect = rendered_rect * page.derotation_matrix
        return {
            "source_coordinate_space": "raster_pixels",
            "coordinate_space": "pdf_points",
            "pdf_coordinate_convention": "pymupdf_unrotated_page_points",
            "dpi": dpi,
            "raster_width": pixmap.width,
            "raster_height": pixmap.height,
            "pdf_bbox": [float(target_rect.x0), float(target_rect.y0), float(target_rect.x1), float(target_rect.y1)],
            "rendered_pdf_bbox": [
                float(rendered_rect.x0), float(rendered_rect.y0),
                float(rendered_rect.x1), float(rendered_rect.y1),
            ],
            "derotation_matrix": [float(value) for value in page.derotation_matrix],
            "page_rotation": page.rotation,
        }
    finally:
        del pixmap


def _float_bbox(value: Any) -> list[float] | None:
    if isinstance(value, fitz.Rect):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None


def _derotation_matrix(raster: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    """Return a JSON-safe PyMuPDF affine matrix, if this raster recorded one."""
    raw = raster.get("derotation_matrix")
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        return tuple(float(value) for value in raw)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _raster_point_to_pdf_point(x: Any, y: Any, raster: dict[str, Any]) -> list[float] | None:
    """Map one PNG point to PyMuPDF's unrotated PDF-point space.

    A rendered page may be rotated.  First scale pixels into the rotated
    ``page.rect`` space, then apply the stored derotation matrix.  Older job
    records did not persist a matrix, so they retain their legacy identity
    behavior rather than being silently guessed at.
    """
    rendered_bbox = _float_bbox(raster.get("rendered_pdf_bbox")) or _float_bbox(raster.get("pdf_bbox"))
    try:
        raster_width = float(raster["raster_width"])
        raster_height = float(raster["raster_height"])
        source_x = float(x)
        source_y = float(y)
    except (KeyError, TypeError, ValueError):
        return None
    if rendered_bbox is None or raster_width <= 0 or raster_height <= 0:
        return None
    point_x = rendered_bbox[0] + source_x * (rendered_bbox[2] - rendered_bbox[0]) / raster_width
    point_y = rendered_bbox[1] + source_y * (rendered_bbox[3] - rendered_bbox[1]) / raster_height
    matrix = _derotation_matrix(raster)
    if matrix is None:
        return [point_x, point_y]
    a, b, c, d, e, f = matrix
    return [a * point_x + c * point_y + e, b * point_x + d * point_y + f]


def scale_bbox_to_pdf_points(bbox: Any, raster: dict[str, Any]) -> list[float] | None:
    """Map an OCR bbox from the rendered PNG into PDF-point coordinates.

    All four corners are transformed because derotation can swap or reverse
    axes (notably at 90 and 270 degrees).
    """
    source = _float_bbox(bbox)
    if source is None:
        return None
    points = [
        _raster_point_to_pdf_point(source[0], source[1], raster),
        _raster_point_to_pdf_point(source[2], source[1], raster),
        _raster_point_to_pdf_point(source[2], source[3], raster),
        _raster_point_to_pdf_point(source[0], source[3], raster),
    ]
    if any(point is None for point in points):
        return None
    mapped = [point for point in points if point is not None]
    return [
        min(point[0] for point in mapped), min(point[1] for point in mapped),
        max(point[0] for point in mapped), max(point[1] for point in mapped),
    ]


def scale_polygon_to_pdf_points(value: Any, raster: dict[str, Any]) -> list[list[float]] | None:
    """Map a polygon from raster pixels into PDF points, retaining its shape."""
    if not isinstance(value, (list, tuple)):
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        mapped = _raster_point_to_pdf_point(point[0], point[1], raster)
        if mapped is None:
            return None
        points.append(mapped)
    return points


def scale_ocr_items_to_pdf_points(items: Sequence[dict[str, Any]], raster: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy OCR items into a common PDF-point coordinate system.

    ``source_bbox`` / ``source_polygon`` retain the exact engine payload;
    ``bbox`` / ``polygon`` become interoperable with PyMuPDF's native output.
    """
    scaled: list[dict[str, Any]] = []
    for item in items:
        output = dict(item)
        bbox = _float_bbox(item.get("bbox"))
        if bbox is not None:
            output["source_bbox"] = bbox
            mapped_bbox = scale_bbox_to_pdf_points(bbox, raster)
            if mapped_bbox is not None:
                output["bbox"] = mapped_bbox
        polygon = item.get("polygon")
        mapped_polygon = scale_polygon_to_pdf_points(polygon, raster)
        if mapped_polygon is not None:
            output["source_polygon"] = [list(point[:2]) for point in polygon]
            output["polygon"] = mapped_polygon
        output["coordinate_space"] = "pdf_points"
        scaled.append(output)
    return scaled


def executable_for(name: str) -> str | None:
    """Find a system executable or the one installed in this Python venv."""
    return shutil.which(name) or (str(Path(sys.executable).parent / name) if (Path(sys.executable).parent / name).is_file() else None)


def parse_tesseract_tsv(tsv: str) -> tuple[str, list[float]]:
    words: list[str] = []
    confidences: list[float] = []
    # Tesseract emits TSV, not RFC-4180 CSV.  In particular, it can emit a
    # literal double quote in a recognised word without escaping it.  The CSV
    # parser then treats the rest of the page as a quoted field and corrupts
    # output.  Split fields directly, retaining any further tabs in the text.
    for line in tsv.splitlines()[1:]:
        fields = line.split("\t", 11)
        if len(fields) != 12:
            continue
        word = fields[11].strip()
        if not word:
            continue
        words.append(word)
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        if confidence >= 0:
            confidences.append(confidence)
    return " ".join(words).strip(), confidences


def _as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_from_tsv(fields: Sequence[str]) -> list[float] | None:
    """Convert Tesseract's left/top/width/height fields into x0/y0/x1/y1."""
    if len(fields) < 10:
        return None
    left, top, width, height = (_as_float(fields[index]) for index in range(6, 10))
    if None in (left, top, width, height):
        return None
    return [left, top, left + max(0.0, width), top + max(0.0, height)]


def canonical_blocks(raw_blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give all engines the same ordered block schema.

    Every exported block has a contiguous ``reading_order``.  When an engine
    supplies an order (Surya does), that original value is retained separately
    as ``source_reading_order`` for traceability.
    """
    prepared: list[tuple[int, int, dict[str, Any]]] = []
    for original_index, raw in enumerate(raw_blocks, start=1):
        text = str(raw.get("text", "")).strip()
        # A Surya picture / diagram can intentionally have no OCR text.  It is
        # still meaningful rich output when its type and geometry survive.
        if not text and not raw.get("retain_empty"):
            continue
        source_order = raw.get("reading_order")
        order_number = _as_int(source_order, default=10**9)
        block: dict[str, Any] = {
            "block_type": str(raw.get("block_type") or raw.get("label") or "Text"),
            "type": str(raw.get("type") or "text").lower(),
            "text": text,
        }
        bbox = raw.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                block["bbox"] = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                pass
        if source_order is not None:
            block["source_reading_order"] = source_order
        # Keep documented engine evidence rather than flattening it away.  We
        # deliberately whitelist fields so the normalized artifact remains
        # stable even when an engine adds unrelated response fields.
        for key in ("raw_label", "html", "source", "coordinate_space", "skipped", "error", "retain_empty"):
            if key in raw:
                block[key] = raw[key]
        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)):
            block["confidence"] = float(confidence)
        polygon = raw.get("polygon")
        if isinstance(polygon, (list, tuple)):
            block["polygon"] = [list(point[:2]) for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
        source_bbox = raw.get("source_bbox")
        if isinstance(source_bbox, (list, tuple)) and len(source_bbox) >= 4:
            try:
                block["source_bbox"] = [float(value) for value in source_bbox[:4]]
            except (TypeError, ValueError):
                pass
        source_polygon = raw.get("source_polygon")
        if isinstance(source_polygon, (list, tuple)):
            block["source_polygon"] = [list(point[:2]) for point in source_polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
        if isinstance(raw.get("table"), dict):
            block["table"] = raw["table"]
        if isinstance(raw.get("lines"), list):
            block["lines"] = raw["lines"]
        if isinstance(raw.get("source_tsv"), dict):
            block["source_tsv"] = raw["source_tsv"]
        prepared.append((order_number, original_index, block))

    # Stable ordering preserves an engine's output sequence when it does not
    # expose explicit reading-order metadata (notably Tesseract TSV).
    prepared.sort(key=lambda item: (item[0], item[1]))
    blocks: list[dict[str, Any]] = []
    for reading_order, (_, _, block) in enumerate(prepared, start=1):
        block["reading_order"] = reading_order
        blocks.append(block)
    return blocks


def parse_tesseract_tsv_layout(tsv: str) -> tuple[str, list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse Tesseract TSV into text, confidence values, words, and line blocks.

    Tesseract emits words in its inferred reading sequence.  We retain that
    sequence at word level and aggregate words with the same block/paragraph/
    line IDs into line blocks.  This makes the final JSON useful for layout and
    reading-order evaluation without changing the quality gate's text result.
    """
    words: list[dict[str, Any]] = []
    confidences: list[float] = []
    line_order: list[tuple[int, int, int]] = []
    lines: dict[tuple[int, int, int], dict[str, Any]] = {}

    for line in tsv.splitlines()[1:]:
        fields = line.split("\t", 11)
        if len(fields) != 12 or _as_int(fields[0], default=-1) != 5:
            continue
        text = fields[11].strip()
        if not text:
            continue
        confidence = _as_float(fields[10])
        bbox = _bbox_from_tsv(fields)
        source_tsv = {
            "page": _as_int(fields[1]),
            "block": _as_int(fields[2]),
            "paragraph": _as_int(fields[3]),
            "line": _as_int(fields[4]),
            "word": _as_int(fields[5]),
        }
        word: dict[str, Any] = {
            "text": text,
            "reading_order": len(words) + 1,
            "source_tsv": source_tsv,
        }
        if bbox is not None:
            word["bbox"] = bbox
        if confidence is not None and confidence >= 0:
            word["confidence"] = confidence
            confidences.append(confidence)
        words.append(word)

        line_key = (_as_int(fields[2]), _as_int(fields[3]), _as_int(fields[4]))
        if line_key not in lines:
            lines[line_key] = {
                "block_type": "Text",
                "type": "text",
                "text_parts": [],
                "bboxes": [],
                "source_tsv": {
                    "page": _as_int(fields[1]),
                    "block": line_key[0],
                    "paragraph": line_key[1],
                    "line": line_key[2],
                },
            }
            line_order.append(line_key)
        lines[line_key]["text_parts"].append(text)
        if bbox is not None:
            lines[line_key]["bboxes"].append(bbox)

    raw_blocks: list[dict[str, Any]] = []
    for source_order, line_key in enumerate(line_order, start=1):
        line = lines[line_key]
        block: dict[str, Any] = {
            "block_type": "Text",
            "type": "text",
            "text": " ".join(line["text_parts"]),
            "reading_order": source_order,
            "source_tsv": line["source_tsv"],
        }
        if line["bboxes"]:
            boxes = line["bboxes"]
            block["bbox"] = [
                min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes),
            ]
        raw_blocks.append(block)
    return " ".join(word["text"] for word in words).strip(), confidences, words, canonical_blocks(raw_blocks)


def _tesseract_line_words(words: Sequence[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    """Group reliable word geometry by Tesseract's block/paragraph/line IDs."""
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        source = word.get("source_tsv")
        bbox = _float_bbox(word.get("bbox"))
        if not isinstance(source, dict) or bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        key = (
            _as_int(source.get("block")),
            _as_int(source.get("paragraph")),
            _as_int(source.get("line")),
        )
        if key[0] <= 0 or key[2] <= 0:
            continue
        grouped.setdefault(key, []).append({"bbox": bbox, "word": word})
    for line_words in grouped.values():
        line_words.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return grouped


def _raster_size(raster: dict[str, Any]) -> tuple[float, float] | None:
    try:
        width = float(raster["raster_width"])
        height = float(raster["raster_height"])
    except (KeyError, TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _bbox_union(items: Sequence[dict[str, Any]]) -> list[float] | None:
    boxes = [item["bbox"] for item in items if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    ]


def _aligned_column_evidence(
    lines: dict[tuple[int, int, int], list[dict[str, Any]]],
    raster_width: float,
    raster_height: float,
) -> list[dict[str, Any]]:
    """Find repeated large in-line gaps, a cheap table / form signal.

    Normal wrapped prose has a stable *left* edge but does not repeatedly
    start another text island at the same x coordinate.  In contrast, a
    table's value column commonly produces exactly that pattern in Tesseract
    TSV, even when the page has no extractable PDF ruling lines.
    """
    gap_threshold = raster_width * STRUCTURE_GAP_RATIO
    tolerance = raster_width * STRUCTURE_CLUSTER_TOLERANCE_RATIO
    points: list[dict[str, Any]] = []
    line_heights: list[float] = []
    for line_key, line_words in lines.items():
        line_bbox = _bbox_union(line_words)
        if line_bbox is None:
            continue
        line_heights.append(line_bbox[3] - line_bbox[1])
        for previous, current in zip(line_words, line_words[1:]):
            gap = current["bbox"][0] - previous["bbox"][2]
            if gap >= gap_threshold:
                points.append({
                    "x": current["bbox"][0],
                    "y": line_bbox[1],
                    "line": line_key,
                    "gap": gap,
                })

    # Keep clustering deterministic and use the nearest cluster when two
    # candidate starts are close enough.  This tolerates minor OCR skew while
    # retaining distinct table columns.
    clusters: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: (item["x"], item["y"])):
        candidates = [
            cluster for cluster in clusters
            if abs(point["x"] - cluster["mean_x"]) <= tolerance
        ]
        cluster = min(candidates, key=lambda item: abs(point["x"] - item["mean_x"])) if candidates else None
        if cluster is None:
            cluster = {"mean_x": point["x"], "points": []}
            clusters.append(cluster)
        cluster["points"].append(point)
        cluster["mean_x"] = fmean(item["x"] for item in cluster["points"])

    median_line_height = median(line_heights) if line_heights else 0.0
    min_vertical_span = max(
        raster_height * STRUCTURE_MIN_VERTICAL_SPAN_RATIO,
        median_line_height * max(1, STRUCTURE_MIN_ALIGNED_LINES - 1),
    )
    evidence: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_points = cluster["points"]
        lines_seen = {point["line"] for point in cluster_points}
        ys = [point["y"] for point in cluster_points]
        vertical_span = max(ys) - min(ys) if ys else 0.0
        start_ratio = cluster["mean_x"] / raster_width
        if (
            len(lines_seen) < STRUCTURE_MIN_ALIGNED_LINES
            or start_ratio < STRUCTURE_MIN_ALIGNED_START_RATIO
            or vertical_span < min_vertical_span
        ):
            continue
        evidence.append({
            "start_x": round(cluster["mean_x"], 3),
            "start_ratio": round(start_ratio, 6),
            "line_count": len(lines_seen),
            "vertical_span": round(vertical_span, 3),
            "vertical_span_ratio": round(vertical_span / raster_height, 6),
            "mean_gap": round(fmean(point["gap"] for point in cluster_points), 3),
            "mean_gap_ratio": round(fmean(point["gap"] for point in cluster_points) / raster_width, 6),
        })
    return sorted(evidence, key=lambda item: (-item["line_count"], item["start_x"]))


def _multi_column_evidence(
    lines: dict[tuple[int, int, int], list[dict[str, Any]]],
    raster_width: float,
    raster_height: float,
) -> list[dict[str, Any]]:
    """Detect two substantial, side-by-side Tesseract text blocks."""
    blocks: dict[int, list[dict[str, Any]]] = {}
    line_counts: dict[int, int] = {}
    for (block_number, _, _), line_words in lines.items():
        blocks.setdefault(block_number, []).extend(line_words)
        line_counts[block_number] = line_counts.get(block_number, 0) + 1

    candidates: list[dict[str, Any]] = []
    for block_number, block_words in blocks.items():
        bbox = _bbox_union(block_words)
        if bbox is None:
            continue
        width_ratio = (bbox[2] - bbox[0]) / raster_width
        height_ratio = (bbox[3] - bbox[1]) / raster_height
        if (
            line_counts.get(block_number, 0) < STRUCTURE_MIN_COLUMN_LINES
            or len(block_words) < STRUCTURE_MIN_COLUMN_WORDS
            or not STRUCTURE_MIN_COLUMN_WIDTH_RATIO <= width_ratio <= STRUCTURE_MAX_COLUMN_WIDTH_RATIO
            or height_ratio < STRUCTURE_MIN_COLUMN_HEIGHT_RATIO
        ):
            continue
        candidates.append({
            "block": block_number,
            "bbox": bbox,
            "line_count": line_counts[block_number],
            "word_count": len(block_words),
        })

    evidence: list[dict[str, Any]] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            left, right = sorted((first, second), key=lambda item: item["bbox"][0])
            horizontal_gap = right["bbox"][0] - left["bbox"][2]
            if horizontal_gap < raster_width * STRUCTURE_MIN_COLUMN_GAP_RATIO:
                continue
            overlap = max(
                0.0,
                min(left["bbox"][3], right["bbox"][3]) - max(left["bbox"][1], right["bbox"][1]),
            )
            shorter_height = min(
                left["bbox"][3] - left["bbox"][1],
                right["bbox"][3] - right["bbox"][1],
            )
            if shorter_height <= 0 or overlap / shorter_height < STRUCTURE_MIN_COLUMN_VERTICAL_OVERLAP:
                continue
            evidence.append({
                "left_block": left["block"],
                "right_block": right["block"],
                "horizontal_gap": round(horizontal_gap, 3),
                "horizontal_gap_ratio": round(horizontal_gap / raster_width, 6),
                "vertical_overlap_ratio": round(overlap / shorter_height, 6),
                "left_line_count": left["line_count"],
                "right_line_count": right["line_count"],
                "left_word_count": left["word_count"],
                "right_word_count": right["word_count"],
            })
    return evidence


def _strict_native_table_evidence(page: fitz.Page | None) -> list[dict[str, Any]]:
    """Use only strict vector rulings as a high-precision extra signal.

    PyMuPDF's text strategy can turn ordinary aligned prose into a giant false
    table.  Strict line detection deliberately avoids that failure mode and is
    advisory: a PyMuPDF exception must never make an OCR page fail.
    """
    if page is None:
        return []
    try:
        tables = page.find_tables(
            vertical_strategy="lines_strict",
            horizontal_strategy="lines_strict",
        ).tables
    except Exception:
        return []
    evidence: list[dict[str, Any]] = []
    for table in tables:
        row_count = _as_int(getattr(table, "row_count", 0))
        column_count = _as_int(getattr(table, "col_count", 0))
        if row_count < 2 or column_count < 2:
            continue
        item: dict[str, Any] = {"row_count": row_count, "column_count": column_count}
        bbox = _float_bbox(getattr(table, "bbox", None))
        if bbox is not None:
            item["bbox"] = bbox
            item["coordinate_space"] = "pdf_points"
        evidence.append(item)
    return evidence


def tesseract_structure_gate(
    words: Sequence[dict[str, Any]],
    raster: dict[str, Any],
    page: fitz.Page | None = None,
) -> dict[str, Any]:
    """Return auditable, conservative structure-risk evidence for one page.

    The gate does not try to reconstruct a table or claim that every positive
    is a table.  It asks whether the page is structurally rich enough that a
    full-page Surya result is worth preserving as the authoritative layer.
    """
    lines = _tesseract_line_words(words)
    size = _raster_size(raster)
    aligned_columns: list[dict[str, Any]] = []
    multi_columns: list[dict[str, Any]] = []
    geometry_available = size is not None
    if size is not None:
        raster_width, raster_height = size
        aligned_columns = _aligned_column_evidence(lines, raster_width, raster_height)
        multi_columns = _multi_column_evidence(lines, raster_width, raster_height)
    native_tables = _strict_native_table_evidence(page)
    reasons: list[str] = []
    if aligned_columns:
        reasons.append("repeated_aligned_columns")
    if multi_columns:
        reasons.append("multiple_text_columns")
    if native_tables:
        reasons.append("strict_native_table")
    return {
        "enabled": True,
        "evaluated": True,
        "escalate": bool(reasons),
        "reasons": reasons,
        "signals": {
            "geometry_available": geometry_available,
            "tesseract_geometry_coordinate_space": "raster_pixels" if geometry_available else None,
            "tesseract_word_count": len(words),
            "tesseract_line_count": len(lines),
            "aligned_columns": aligned_columns,
            "multi_column_pairs": multi_columns,
            "strict_native_tables": native_tables,
        },
    }


def tesseract_quality(text: str, confidences: Sequence[float], config: PipelineConfig) -> dict[str, Any]:
    """Decide whether cheap OCR is safe enough to accept.

    A page must have enough material to be meaningful, reliable word-level
    confidence, low corruption, and plausible tokens.  Any failed condition is
    saved as a rejection reason and routes the page to Surya; this makes the
    fallback decision reviewable and tuneable rather than a black box.
    """
    text = normalize_whitespace(text)
    character_count = len(text)
    word_count = len(re.findall(r"\S+", text))
    mean_confidence = fmean(confidences) if confidences else None
    confident_ratio = (
        sum(value >= config.confident_word_threshold for value in confidences) / len(confidences)
        if confidences else 0.0
    )
    ocr_garbage = garbage_ratio(text)
    plausibility = plausible_word_ratio(text)
    rejected: list[str] = []
    # Sparse output is commonly a blank/failed recognition result, even if its
    # few detected words look confident.
    if character_count < config.min_tesseract_chars:
        rejected.append("too_few_characters")
    if word_count < config.min_tesseract_words:
        rejected.append("too_few_words")
    # Tesseract's TSV confidence is our cheapest quality signal.  Both its
    # mean and the share of individually confident words must pass.
    if not confidences:
        rejected.append("no_word_confidences")
    elif mean_confidence is not None and mean_confidence < config.min_mean_confidence:
        rejected.append("low_mean_confidence")
    if confident_ratio < config.min_confident_word_ratio:
        rejected.append("low_confident_word_ratio")
    # Confidence alone can miss encoding noise or symbol-heavy gibberish.
    if ocr_garbage > config.max_tesseract_garbage_ratio:
        rejected.append("high_garbage_ratio")
    if plausibility < config.min_plausible_word_ratio:
        rejected.append("low_word_plausibility")
    return {
        "accepted": not rejected,
        "rejection_reasons": rejected,
        "text_chars": character_count,
        "word_count": word_count,
        "word_confidence_count": len(confidences),
        "mean_word_confidence": round(mean_confidence, 3) if mean_confidence is not None else None,
        "confident_word_ratio": round(confident_ratio, 6),
        "garbage_ratio": round(ocr_garbage, 6),
        "plausible_word_ratio": round(plausibility, 6),
    }


def tesseract_page(image_path: Path, config: PipelineConfig) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the cheap first-pass OCR engine and return its ordered layout.

    The quality gate below—not a file-name rule or a second OCR run—decides
    whether this output is accepted.  Pages rejected by the gate are queued
    for the more capable, slower Surya fallback.
    """
    executable = executable_for("tesseract")
    if executable is None:
        raise RuntimeError("Tesseract 5 executable was not found on PATH")
    started = time.perf_counter()
    result = subprocess.run(
        [executable, str(image_path), "stdout", "-l", config.language, "--psm", str(config.tesseract_psm), "tsv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=config.tesseract_timeout_seconds,
    )
    text, confidences, words, blocks = parse_tesseract_tsv_layout(result.stdout)
    quality = tesseract_quality(text, confidences, config)
    quality["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return text, quality, words, blocks


def extract_surya_text(prediction: Any) -> str:
    if not isinstance(prediction, dict):
        return ""
    return "\n".join(
        text for block in prediction.get("blocks", [])
        if isinstance(block, dict) and not block.get("error")
        if (text := html_to_text(block.get("html")))
    ).strip()


def bbox_from_polygon(value: Any) -> list[float] | None:
    """Accept Surya's bbox or either common polygon representation."""
    if isinstance(value, (list, tuple)) and len(value) >= 4 and all(isinstance(v, (int, float)) for v in value[:4]):
        return [float(value[index]) for index in range(4)]
    if isinstance(value, (list, tuple)):
        points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
        if points:
            try:
                return [
                    min(float(point[0]) for point in points), min(float(point[1]) for point in points),
                    max(float(point[0]) for point in points), max(float(point[1]) for point in points),
                ]
            except (TypeError, ValueError):
                return None
    return None


def extract_surya_layout(prediction: Any) -> tuple[str, list[dict[str, Any]]]:
    """Normalize Surya blocks without discarding semantic or table evidence."""
    if not isinstance(prediction, dict):
        return "", []
    raw_blocks: list[dict[str, Any]] = []
    for original_index, block in enumerate(prediction.get("blocks", []), start=1):
        if not isinstance(block, dict):
            continue
        html_value = block.get("html")
        text = html_to_text(html_value)
        # Keep blank visual / failed blocks: their labels and geometry are the
        # only reliable way for downstream consumers to know content exists.
        label = str(block.get("label") or block.get("raw_label") or "Text")
        visual_or_failed = (
            bool(block.get("skipped") or block.get("error"))
            or label.strip().lower() in VISUAL_BLOCK_TYPES
        )
        if not text and not visual_or_failed:
            continue
        raw_block: dict[str, Any] = {
            "block_type": label,
            "type": label.lower(),
            "text": text,
            # Use original sequence only when Surya did not supply an order.
            "reading_order": block.get("reading_order", original_index),
        }
        if isinstance(html_value, str):
            raw_block["html"] = html_value
            table = html_to_table(html_value)
            if table is not None:
                raw_block["table"] = table
        if isinstance(block.get("raw_label"), str):
            raw_block["raw_label"] = block["raw_label"]
        if isinstance(block.get("confidence"), (int, float)):
            raw_block["confidence"] = float(block["confidence"])
        if "skipped" in block:
            raw_block["skipped"] = bool(block["skipped"])
        if "error" in block:
            raw_block["error"] = bool(block["error"])
        if visual_or_failed:
            raw_block["retain_empty"] = True
        bbox = bbox_from_polygon(block.get("bbox")) or bbox_from_polygon(block.get("polygon"))
        if bbox is not None:
            raw_block["bbox"] = bbox
        polygon = block.get("polygon")
        if isinstance(polygon, (list, tuple)):
            raw_block["polygon"] = polygon
        raw_blocks.append(raw_block)
    blocks = canonical_blocks(raw_blocks)
    return "\n".join(block["text"] for block in blocks if block.get("text")).strip(), blocks


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def surya_batch(input_dir: Path, output_dir: Path, config: PipelineConfig) -> dict[str, dict[str, Any]]:
    executable = executable_for("surya_ocr")
    if executable is None:
        raise RuntimeError("surya_ocr executable was not found on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = [executable, str(input_dir), "--output_dir", str(output_dir)]
    if config.surya_keep_server:
        # Surya's CLI has no safe server-owner-aware stop command.  This is
        # deliberately opt-in: a caller that selects it must manage the
        # host-wide inference server lifecycle.
        command.append("--keep_server")
    subprocess.run(
        command,
        check=True,
        timeout=config.surya_timeout_seconds,
    )
    # Surya's CLI deliberately nests output as
    # ``<output_dir>/<input-folder-name>/results.json``.
    result_dir = output_dir / input_dir.name
    results_path = result_dir / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"Surya did not produce {results_path}")
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("Surya results.json is not a JSON object")
    output: dict[str, dict[str, Any]] = {}
    for image_name, predictions in raw.items():
        if isinstance(predictions, list):
            page_texts: list[str] = []
            page_blocks: list[dict[str, Any]] = []
            source_image_bbox: list[float] | None = None
            for prediction in predictions:
                text, blocks = extract_surya_layout(prediction)
                if text:
                    page_texts.append(text)
                page_blocks.extend(blocks)
                if source_image_bbox is None and isinstance(prediction, dict):
                    source_image_bbox = bbox_from_polygon(prediction.get("image_bbox"))
            # Some APIs can return multiple predictions for one source image.
            # Re-index their concatenated blocks once, so exported order stays
            # contiguous and can be validated page by page.
            page = {"text": "\n".join(page_texts).strip(), "blocks": canonical_blocks(page_blocks)}
            if source_image_bbox is not None:
                page["source_image_bbox"] = source_image_bbox
            # Folder input names are Surya's extension-free stems, while PDF
            # input names can retain another representation.  Store both.
            output[Path(str(image_name)).name] = page
            output[Path(str(image_name)).stem] = page
    atomic_write_json(result_dir / "timing.json", {
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "input_images": len(list(input_dir.glob("*.png"))),
    })
    return output


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed_pages(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                page_number = int(record["page"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid manifest at {path}:{line_number}") from exc
            if record.get("status") == COMPLETE:
                completed[page_number] = record
    return completed


def source_identity(pdf_path: Path) -> dict[str, Any]:
    """Return a path-free, content-based resume identity for a source PDF."""
    stat = pdf_path.stat()
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return {
        "file_name": pdf_path.name,
        "size_bytes": stat.st_size,
        "content_sha256": digest.hexdigest(),
    }


def job_output_dir(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        relative = pdf_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(pdf_path.name)
    return output_root / relative.with_suffix("")


def config_matches(existing: Any, expected: dict[str, Any]) -> bool:
    """Compare persisted policy while allowing newly-added default fields.

    A version change still requires ``--reprocess-ocr``.  This compatibility
    step merely lets an older job reach that explicit upgrade path instead of
    failing early because a harmless new default is absent from its JSON.
    """
    if not isinstance(existing, dict) or set(existing) - set(expected):
        return False
    defaults = asdict(PipelineConfig())
    normalized = {key: existing.get(key, defaults[key]) for key in expected}
    return normalized == expected


def check_or_create_job(
    job_dir: Path,
    source: dict[str, Any],
    config: PipelineConfig,
    resume: bool,
    allow_pipeline_upgrade: bool = False,
) -> None:
    expected = {"pipeline_version": PIPELINE_VERSION, "source": source, "config": asdict(config)}
    job_path = job_dir / "job.json"
    if job_path.exists():
        if not resume:
            raise RuntimeError(f"Job already exists at {job_dir}; remove --no-resume or use a new output directory")
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        if existing.get("source") != expected["source"]:
            raise RuntimeError(
                f"Existing job at {job_dir} has a different source; use a new output directory so results are not mixed."
            )
        if not config_matches(existing.get("config"), expected["config"]):
            raise RuntimeError(
                f"Existing job at {job_dir} has a different config; use a new output directory so results are not mixed."
            )
        if existing.get("pipeline_version") != PIPELINE_VERSION:
            if not allow_pipeline_upgrade:
                raise RuntimeError(
                    f"Existing job at {job_dir} uses pipeline {existing.get('pipeline_version')}; "
                    "use --reprocess-ocr to repair OCR records or choose a new output directory."
                )
            atomic_write_json(job_path, {
                **existing,
                "config": expected["config"],
                "pipeline_version": PIPELINE_VERSION,
                "upgraded_at_epoch": time.time(),
            })
        return
    job_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path, {**expected, "created_at_epoch": time.time()})


def make_record(
    page: int,
    inspection: dict[str, Any],
    *,
    engine: str,
    outcome: str,
    text: str,
    quality: dict[str, Any] | None = None,
    surya_batch_number: int | None = None,
    words: list[dict[str, Any]] | None = None,
    blocks: list[dict[str, Any]] | None = None,
    raster: dict[str, Any] | None = None,
    tesseract_candidate: dict[str, Any] | None = None,
    escalation_reason: str | None = None,
    structure_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "page": page,
        "status": COMPLETE,
        "classification": inspection["classification"],
        "route": inspection["route"],
        "engine": engine,
        "outcome": outcome,
        "signals": inspection["signals"],
        "native_text": inspection["native_text"],
        "text": text,
        "completed_at_epoch": time.time(),
    }
    if quality is not None:
        record["tesseract_quality"] = quality
    if surya_batch_number is not None:
        record["surya_batch"] = surya_batch_number
    if words:
        record["words"] = words
    if blocks:
        record["blocks"] = blocks
    if raster is not None:
        record["raster"] = raster
    if escalation_reason is not None:
        record["escalation_reason"] = escalation_reason
    if structure_gate is not None:
        record["structure_gate"] = structure_gate
    # A rejected Tesseract pass is valuable audit evidence after Surya wins.
    # Retain it only for fallback pages so accepted pages do not duplicate
    # their already-authoritative words and blocks in the JSONL manifest.
    if tesseract_candidate is not None:
        record["tesseract_candidate"] = tesseract_candidate
    return record


def summarise_document(
    pdf_path: Path,
    page_count: int,
    records: dict[int, dict[str, Any]],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = source or {"file_name": pdf_path.name, "content_sha256": None}
    states = {state: 0 for state in PAGE_STATES}
    outcomes: dict[str, int] = {}
    confidences: list[float] = []
    quality_escalations = structure_escalations = 0
    for record in records.values():
        if record.get("classification") in states:
            states[record["classification"]] += 1
        outcome = str(record.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        quality = record.get("tesseract_quality")
        if isinstance(quality, dict) and isinstance(quality.get("mean_word_confidence"), (int, float)):
            confidences.append(float(quality["mean_word_confidence"]))
        # Older manifests did not distinguish fallback reasons.  Their sole
        # historical Surya route was the quality gate, so classify them there
        # while retaining the long-standing total below.
        if record.get("outcome") == "surya_escalated":
            if record.get("escalation_reason") == "structure":
                structure_escalations += 1
            else:
                quality_escalations += 1
    return {
        "pipeline_version": PIPELINE_VERSION,
        "source": source["file_name"],
        "source_sha256": source.get("content_sha256"),
        "source_page_count": page_count,
        "completed_pages": len(records),
        "pending_pages": max(0, page_count - len(records)),
        "state": "complete" if len(records) == page_count and page_count else "incomplete",
        "classification_counts": states,
        "outcome_counts": outcomes,
        "native_pages": outcomes.get("native_text_accepted", 0),
        "tesseract_accepted_pages": outcomes.get("tesseract_accepted", 0),
        "surya_escalated_pages": outcomes.get("surya_escalated", 0),
        "surya_quality_escalated_pages": quality_escalations,
        "surya_structure_escalated_pages": structure_escalations,
        "mean_tesseract_confidence": round(fmean(confidences), 3) if confidences else None,
        "updated_at_epoch": time.time(),
    }


def write_combined_text(path: Path, records: dict[int, dict[str, Any]], page_count: int) -> None:
    if len(records) != page_count:
        return
    lines: list[str] = []
    for page in range(1, page_count + 1):
        lines += [f"--- Page {page} ---", str(records[page].get("text", "")).strip(), ""]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def page_bbox(page: fitz.Page) -> list[float]:
    """Return the unrotated page envelope used by native PyMuPDF geometry."""
    rect = page.rect * page.derotation_matrix
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def native_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """Build ordered native blocks for the normalized result JSON.

    ``sort=True`` asks PyMuPDF to return its geometry-aware reading sequence.
    This is intentionally used for exported layout metadata; it does not make
    an unsupported claim that multi-column reading order is always perfect.
    """
    raw_blocks: list[dict[str, Any]] = []
    for source_order, block in enumerate(page.get_text("blocks", sort=True), start=1):
        if len(block) < 5 or not str(block[4]).strip():
            continue
        raw_blocks.append({
            "block_type": "Text",
            "type": "text",
            "text": str(block[4]).strip(),
            "bbox": [block[0], block[1], block[2], block[3]],
            "reading_order": source_order,
        })
    return canonical_blocks(raw_blocks)


def _native_span(span: dict[str, Any]) -> dict[str, Any] | None:
    text = str(span.get("text", ""))
    if not text:
        return None
    output: dict[str, Any] = {"text": text}
    bbox = _float_bbox(span.get("bbox"))
    if bbox is not None:
        output["bbox"] = bbox
    origin = span.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
        try:
            output["origin"] = [float(origin[0]), float(origin[1])]
        except (TypeError, ValueError):
            pass
    style: dict[str, Any] = {}
    if isinstance(span.get("font"), str):
        style["font"] = span["font"]
    for key in ("size", "flags", "char_flags", "ascender", "descender", "bidi"):
        value = span.get(key)
        if isinstance(value, (int, float)):
            style[key] = value
    color = span.get("color")
    if isinstance(color, int):
        style["color"] = f"#{color & 0xFFFFFF:06x}"
    if style:
        output["style"] = style
    return output


def native_rich_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """Extract native text with spans, styling, direction, and PDF bboxes."""
    raw_blocks: list[dict[str, Any]] = []
    data = page.get_text("dict", sort=True)
    for source_order, raw_block in enumerate(data.get("blocks", []), start=1):
        if not isinstance(raw_block, dict) or raw_block.get("type") != 0:
            continue
        lines: list[dict[str, Any]] = []
        text_lines: list[str] = []
        for raw_line in raw_block.get("lines", []):
            if not isinstance(raw_line, dict):
                continue
            spans = [
                output for span in raw_line.get("spans", [])
                if isinstance(span, dict)
                if (output := _native_span(span)) is not None
            ]
            line_text = "".join(span["text"] for span in spans).strip()
            if not line_text:
                continue
            line: dict[str, Any] = {"text": line_text, "spans": spans}
            bbox = _float_bbox(raw_line.get("bbox"))
            if bbox is not None:
                line["bbox"] = bbox
            direction = raw_line.get("dir")
            if isinstance(direction, (list, tuple)) and len(direction) >= 2:
                try:
                    line["direction"] = [float(direction[0]), float(direction[1])]
                except (TypeError, ValueError):
                    pass
            if isinstance(raw_line.get("wmode"), int):
                line["writing_mode"] = raw_line["wmode"]
            lines.append(line)
            text_lines.append(line_text)
        if not lines:
            continue
        block: dict[str, Any] = {
            "block_type": "Text",
            "type": "text",
            "text": "\n".join(text_lines),
            "reading_order": source_order,
            "coordinate_space": "pdf_points",
            "source": "pymupdf",
            "lines": lines,
        }
        bbox = _float_bbox(raw_block.get("bbox"))
        if bbox is not None:
            block["bbox"] = bbox
        raw_blocks.append(block)
    return canonical_blocks(raw_blocks)


def native_images(page: fitz.Page) -> list[dict[str, Any]]:
    """Describe placed PDF images without embedding their bytes in the JSON."""
    images: list[dict[str, Any]] = []
    for info in page.get_image_info(hashes=True, xrefs=True):
        if not isinstance(info, dict):
            continue
        bbox = _float_bbox(info.get("bbox"))
        if bbox is None:
            continue
        image: dict[str, Any] = {"bbox": bbox, "coordinate_space": "pdf_points"}
        for key in ("xref", "width", "height", "bpc", "colorspace", "size"):
            value = info.get(key)
            if isinstance(value, (int, float, str)):
                image[key] = value
        transform = info.get("transform")
        if isinstance(transform, (list, tuple)):
            try:
                image["transform"] = [float(value) for value in transform]
            except (TypeError, ValueError):
                pass
        digest = info.get("digest")
        if isinstance(digest, bytes):
            image["digest"] = digest.hex()
        images.append(image)
    return images


def native_links(page: fitz.Page) -> list[dict[str, Any]]:
    """Preserve navigable PDF links with their placement geometry."""
    links: list[dict[str, Any]] = []
    for raw_link in page.get_links():
        if not isinstance(raw_link, dict):
            continue
        link: dict[str, Any] = {}
        bbox = _float_bbox(raw_link.get("from"))
        if bbox is not None:
            link["bbox"] = bbox
            link["coordinate_space"] = "pdf_points"
        for key in ("kind", "page", "uri", "xref", "id", "zoom"):
            value = raw_link.get(key)
            if isinstance(value, (int, float, str)):
                link[key] = value
        if link:
            links.append(link)
    return links


def native_page_layer(page: fitz.Page) -> dict[str, Any]:
    """The lossless-enough native layer used alongside OCR-derived layers."""
    return {
        "available": True,
        "coordinate_space": "pdf_points",
        "text": page.get_text("text", sort=True).strip(),
        "blocks": native_rich_blocks(page),
        "images": native_images(page),
        "links": native_links(page),
    }


def reading_order_check(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate explicit order fields and flag—not silently fix—geometry risks.

    A sequence of ``1..N`` proves that our exported order is internally
    consistent.  It cannot prove semantic order in a multi-column page, so a
    vertical backtrack is reported as a visual-review flag rather than counted
    as a failure.  This distinction keeps the report honest.
    """
    orders = [block.get("reading_order") for block in blocks]
    contiguous = orders == list(range(1, len(blocks) + 1))
    boxes = [block["bbox"] for block in blocks if isinstance(block.get("bbox"), list) and len(block["bbox"]) == 4]
    heights = [max(0.0, box[3] - box[1]) for box in boxes if box[3] >= box[1]]
    tolerance = max(4.0, median(heights) * 0.75) if heights else None
    vertical_backtracks = 0
    previous_y0: float | None = None
    for block in blocks:
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if previous_y0 is not None and tolerance is not None and bbox[1] < previous_y0 - tolerance:
            vertical_backtracks += 1
        previous_y0 = bbox[1]
    return {
        "status": "maintained" if contiguous else "invalid",
        "block_count": len(blocks),
        "contiguous_block_order": contiguous,
        "blocks_with_bbox": len(boxes),
        "vertical_backtrack_warnings": vertical_backtracks,
        "visual_review_recommended": vertical_backtracks > 0,
    }


def normalized_page(record: dict[str, Any], page: fitz.Page) -> dict[str, Any]:
    """Create one Chandra-shaped page entry without hiding cascade decisions."""
    blocks = record.get("blocks")
    if not isinstance(blocks, list):
        blocks = native_blocks(page) if record.get("engine") == "pymupdf" else canonical_blocks([
            {"block_type": "Text", "type": "text", "text": record.get("text", ""), "reading_order": 1}
        ])
    blocks = canonical_blocks(blocks)
    entry: dict[str, Any] = {
        "page": record["page"],
        "text": record.get("text", ""),
        "bbox": page_bbox(page),
        "coordinate_space": "pdf_points",
        "block_type": "Page",
        "reading_order": record["page"],
        "blocks": blocks,
        "reading_order_check": reading_order_check(blocks),
        "metadata": {
            "engine": record.get("engine"),
            "classification": record.get("classification"),
            "route": record.get("route"),
            "outcome": record.get("outcome"),
            "signals": record.get("signals"),
        },
    }
    if record.get("words"):
        entry["words"] = record["words"]
    if record.get("tesseract_quality"):
        entry["metadata"]["tesseract_quality"] = record["tesseract_quality"]
    if isinstance(record.get("raster"), dict):
        entry["metadata"]["raster"] = record["raster"]
    if isinstance(record.get("structure_gate"), dict):
        entry["metadata"]["structure_gate"] = record["structure_gate"]
    if isinstance(record.get("escalation_reason"), str):
        entry["metadata"]["escalation_reason"] = record["escalation_reason"]
    return entry


def write_normalized_json(
    job_dir: Path,
    pdf_path: Path,
    records: dict[int, dict[str, Any]],
    summary: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Write a normalized document plus order diagnostics.

    The top-level fields are shaped like the normalized artifact produced by
    ``run_chandra.py``. Extra metadata makes this format suitable for cascade
    audits: consumers can see which engine won each page and why, while generic
    consumers can simply read ``pages[].text``.
    """
    with fitz.open(pdf_path) as pdf:
        pages = [normalized_page(records[number], pdf[number - 1]) for number in range(1, pdf.page_count + 1)]
    page_numbers = [entry["page"] for entry in pages]
    expected_pages = list(range(1, len(pages) + 1))
    order_report = {
        "status": "maintained" if page_numbers == expected_pages else "invalid",
        "page_count": len(pages),
        "contiguous_page_order": page_numbers == expected_pages,
        "pages_with_block_order_error": [
            entry["page"] for entry in pages if entry["reading_order_check"]["status"] != "maintained"
        ],
        "pages_recommended_for_visual_review": [
            entry["page"] for entry in pages if entry["reading_order_check"]["visual_review_recommended"]
        ],
    }
    normalized = {
        "schema_version": "cascade-ocr/v1",
        "engine": "cascade",
        "source": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
        "metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "pipeline": "PyMuPDF -> Tesseract 5 -> quality / optional structure gate -> Surya fallback",
            "routing": {
                "classification_counts": summary["classification_counts"],
                "outcome_counts": summary["outcome_counts"],
                "surya_quality_escalated_pages": summary["surya_quality_escalated_pages"],
                "surya_structure_escalated_pages": summary["surya_structure_escalated_pages"],
            },
            "tesseract_quality_gate": {
                "min_mean_confidence": config.min_mean_confidence,
                "min_confident_word_ratio": config.min_confident_word_ratio,
                "confident_word_threshold": config.confident_word_threshold,
                "max_garbage_ratio": config.max_tesseract_garbage_ratio,
                "min_plausible_word_ratio": config.min_plausible_word_ratio,
                "min_characters": config.min_tesseract_chars,
                "min_words": config.min_tesseract_words,
            },
            "structure_gate": {"enabled": config.structure_aware},
            "reading_order": order_report,
        },
    }
    atomic_write_json(job_dir / f"{pdf_path.stem}_cascade.json", normalized)
    atomic_write_json(job_dir / "reading_order.json", order_report)
    return order_report


def tesseract_layer(record: dict[str, Any]) -> dict[str, Any]:
    """Expose Tesseract evidence even when Surya became authoritative."""
    candidate = record.get("tesseract_candidate")
    if isinstance(candidate, dict):
        return {
            "attempted": True,
            "selected": False,
            "text": candidate.get("text", ""),
            "quality": candidate.get("quality"),
            "words": candidate.get("words", []),
            "blocks": candidate.get("blocks", []),
        }
    if record.get("engine") == "tesseract5":
        return {
            "attempted": True,
            "selected": True,
            "text": record.get("text", ""),
            "quality": record.get("tesseract_quality"),
            "words": record.get("words", []),
            "blocks": record.get("blocks", []),
        }
    return {"attempted": False, "selected": False}


def surya_layer(record: dict[str, Any]) -> dict[str, Any]:
    """Expose Surya's semantic blocks, including its original HTML tables."""
    selected = record.get("engine") == "surya"
    return {
        "attempted": selected,
        "selected": selected,
        "text": record.get("text", "") if selected else "",
        "blocks": record.get("blocks", []) if selected else [],
    }


def rich_page(record: dict[str, Any], page: fitz.Page) -> dict[str, Any]:
    """Build a multi-engine page without pretending that evidence was fused.

    One engine remains authoritative for final text and ordered blocks.  The
    other layers are retained separately with provenance, letting a consumer
    compare, repair, or selectively merge them without losing where a value
    originated.
    """
    authoritative = normalized_page(record, page)
    routing: dict[str, Any] = {
        "classification": record.get("classification"),
        "route": record.get("route"),
        "signals": record.get("signals"),
        "tesseract_quality": record.get("tesseract_quality"),
        "raster": record.get("raster"),
    }
    if isinstance(record.get("structure_gate"), dict):
        routing["structure_gate"] = record["structure_gate"]
    if isinstance(record.get("escalation_reason"), str):
        routing["escalation_reason"] = record["escalation_reason"]
    return {
        "page": record["page"],
        "bbox": page_bbox(page),
        "coordinate_space": "pdf_points",
        "authoritative": {
            "engine": record.get("engine"),
            "outcome": record.get("outcome"),
            "text": authoritative["text"],
            "blocks": authoritative["blocks"],
            "reading_order_check": authoritative["reading_order_check"],
        },
        "layers": {
            "pymupdf": native_page_layer(page),
            "tesseract5": tesseract_layer(record),
            "surya": surya_layer(record),
        },
        "routing": routing,
    }


def write_rich_output(
    job_dir: Path,
    pdf_path: Path,
    records: dict[int, dict[str, Any]],
    summary: dict[str, Any],
    config: PipelineConfig,
) -> Path:
    """Write the multi-layer, coordinate-normalized document artifact."""
    with fitz.open(pdf_path) as pdf:
        pages = [rich_page(records[number], pdf[number - 1]) for number in range(1, pdf.page_count + 1)]
    output_path = job_dir / f"{pdf_path.stem}_rich.json"
    atomic_write_json(output_path, {
        "schema_version": RICH_SCHEMA_VERSION,
        "engine": "cascade",
        "source": pdf_path.name,
        "page_count": len(pages),
        "coordinate_space": "pdf_points",
        "pages": pages,
        "metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "pipeline": "PyMuPDF -> Tesseract 5 -> quality / optional structure gate -> Surya fallback",
            "authoritative_selection": "one engine per page; non-winning evidence is retained in layers",
            "ocr_coordinate_transform": "source_bbox/source_polygon are raster pixels; bbox/polygon are PDF points",
            "routing": {
                "classification_counts": summary["classification_counts"],
                "outcome_counts": summary["outcome_counts"],
                "surya_quality_escalated_pages": summary["surya_quality_escalated_pages"],
                "surya_structure_escalated_pages": summary["surya_structure_escalated_pages"],
            },
            "tesseract_quality_gate": {
                "min_mean_confidence": config.min_mean_confidence,
                "min_confident_word_ratio": config.min_confident_word_ratio,
                "confident_word_threshold": config.confident_word_threshold,
            },
            "structure_gate": {"enabled": config.structure_aware},
        },
    })
    return output_path


def flush_surya(
    pending: list[_PendingSuryaPage],
    *,
    batch_number: int,
    job_dir: Path,
    config: PipelineConfig,
    manifest_path: Path,
    records: dict[int, dict[str, Any]],
) -> None:
    if not pending:
        return
    predictions = surya_batch(pending[0].image_path.parent, job_dir / "surya" / f"batch-{batch_number:06d}", config)
    for candidate in pending:
        prediction = predictions.get(candidate.image_path.name) or predictions.get(candidate.image_path.stem)
        # A Figure / Diagram can be a valuable Surya result even when all of
        # its text blocks were intentionally skipped. A blank scanned page is
        # valid too: reject only a missing or malformed response, never an
        # empty but well-formed Surya page.
        if (
            not isinstance(prediction, dict)
            or not isinstance(prediction.get("blocks"), list)
        ):
            raise RuntimeError(f"Surya returned a malformed result for page {candidate.page} ({candidate.image_path.name})")
        surya_blocks = scale_ocr_items_to_pdf_points(prediction["blocks"], candidate.raster)
        record_raster = dict(candidate.raster)
        source_image_bbox = prediction.get("source_image_bbox")
        if source_image_bbox is not None:
            record_raster["surya_source_image_bbox"] = source_image_bbox
        record = make_record(
            candidate.page, candidate.inspection, engine="surya", outcome="surya_escalated", text=prediction.get("text", ""),
            quality=candidate.quality, surya_batch_number=batch_number, blocks=surya_blocks, raster=record_raster,
            escalation_reason=candidate.escalation_reason, structure_gate=candidate.structure_gate,
            tesseract_candidate={
                "text": candidate.tesseract_text,
                "quality": candidate.quality,
                "words": candidate.tesseract_words,
                "blocks": candidate.tesseract_blocks,
            },
        )
        append_jsonl(manifest_path, record)
        records[candidate.page] = record


def process_document(
    pdf_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    config: PipelineConfig,
    resume: bool = True,
    dry_run: bool = False,
    reprocess_ocr: bool = False,
) -> dict[str, Any]:
    """Process a document using no more than one Surya batch of page images."""
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF: {pdf_path}")
    if dry_run:
        states = {state: 0 for state in PAGE_STATES}
        with fitz.open(pdf_path) as pdf:
            for index in range(pdf.page_count):
                states[inspect_page(pdf[index], config)["classification"]] += 1
            return {
                "source": pdf_path.name, "source_page_count": pdf.page_count,
                "state": "dry_run", "classification_counts": states,
                "ocr_candidate_pages": states["SCANNED"] + states["MIXED"] + states["OCR_NEEDED"],
            }

    job_dir = job_output_dir(pdf_path, input_root, output_root)
    source = source_identity(pdf_path)
    check_or_create_job(job_dir, source, config, resume, allow_pipeline_upgrade=reprocess_ocr)
    manifest_path = job_dir / "pages.jsonl"
    records = load_completed_pages(manifest_path)
    with fitz.open(pdf_path) as pdf, tempfile.TemporaryDirectory(prefix=".sttl-work-", dir=job_dir) as work:
        work_root = Path(work)
        pending: list[_PendingSuryaPage] = []
        batch_number = 0

        def flush_pending() -> None:
            nonlocal batch_number, pending
            if pending:
                batch_number += 1
                flush_surya(
                    pending, batch_number=batch_number, job_dir=job_dir, config=config,
                    manifest_path=manifest_path, records=records,
                )
                pending = []

        for index in range(pdf.page_count):
            page_number = index + 1
            if page_number in records and not (
                reprocess_ocr and records[page_number].get("engine") != "pymupdf"
            ):
                continue
            inspection = inspect_page(pdf[index], config)
            if inspection["route"] == "native_text":
                record = make_record(
                    page_number, inspection, engine="pymupdf", outcome="native_text_accepted",
                    text=inspection["native_text"],
                )
                append_jsonl(manifest_path, record)
                records[page_number] = record
                continue
            # A unique input directory prevents Surya from accidentally seeing
            # subsequent pages: its CLI processes every image in the directory.
            image_dir = work_root / f"surya-input-{batch_number + 1:06d}"
            image_path = image_dir / f"page-{page_number:06d}.png"
            raster = render_page(pdf[index], image_path, config.render_dpi)
            # Tesseract is always attempted before Surya.  Its TSV layout is
            # retained if the quality gate accepts the page; an opt-in
            # structure gate can still select Surya when the plain text is
            # reliable but its table / column semantics are not.
            tesseract_text, quality, tesseract_words, tesseract_blocks = tesseract_page(image_path, config)
            structure_gate = (
                tesseract_structure_gate(tesseract_words, raster, pdf[index])
                if config.structure_aware and quality["accepted"]
                else None
            )
            tesseract_words = scale_ocr_items_to_pdf_points(tesseract_words, raster)
            tesseract_blocks = scale_ocr_items_to_pdf_points(tesseract_blocks, raster)
            if quality["accepted"] and not (structure_gate and structure_gate["escalate"]):
                record = make_record(
                    page_number, inspection, engine="tesseract5", outcome="tesseract_accepted",
                    text=tesseract_text, quality=quality, words=tesseract_words, blocks=tesseract_blocks, raster=raster,
                    structure_gate=structure_gate,
                )
                append_jsonl(manifest_path, record)
                records[page_number] = record
                image_path.unlink(missing_ok=True)
            else:
                pending.append(_PendingSuryaPage(
                    page=page_number,
                    inspection=inspection,
                    image_path=image_path,
                    quality=quality,
                    raster=raster,
                    tesseract_text=tesseract_text,
                    tesseract_words=tesseract_words,
                    tesseract_blocks=tesseract_blocks,
                    escalation_reason="structure" if quality["accepted"] else "quality",
                    structure_gate=structure_gate,
                ))
                if len(pending) >= config.surya_batch_size:
                    flush_pending()
        flush_pending()
        summary = summarise_document(pdf_path, pdf.page_count, records, source=source)
    if summary["state"] == "complete":
        write_combined_text(job_dir / "combined.txt", records, summary["source_page_count"])
        # The normalized JSON is deliberately written only after every page is
        # present, so consumers never mistake a partial document for a complete
        # normalized result.
        summary["reading_order"] = write_normalized_json(job_dir, pdf_path, records, summary, config)
        summary["normalized_output"] = f"{pdf_path.stem}_cascade.json"
        write_rich_output(job_dir, pdf_path, records, summary, config)
        summary["rich_output"] = f"{pdf_path.stem}_rich.json"
    atomic_write_json(job_dir / "document.json", summary)
    return {**summary, "job_dir": str(job_dir.relative_to(output_root))}


def stable_path_key(path: Path) -> int:
    return int(hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:16], 16)


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def iter_pdfs(input_path: Path, output_root: Path) -> Iterator[Path]:
    if input_path.is_file():
        yield input_path
        return
    for candidate in input_path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() == ".pdf" and not is_within(candidate, output_root):
            yield candidate


def write_batch_reports(output_root: Path) -> dict[str, Any]:
    aggregate = {
        "documents": 0, "complete_documents": 0, "incomplete_documents": 0,
        "source_pages": 0, "native_pages": 0, "tesseract_accepted_pages": 0, "surya_escalated_pages": 0,
        "surya_quality_escalated_pages": 0, "surya_structure_escalated_pages": 0,
    }
    csv_path = output_root / "batch_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fields = [
        "source", "state", "source_page_count", "native_pages", "tesseract_accepted_pages", "surya_escalated_pages",
        "surya_quality_escalated_pages", "surya_structure_escalated_pages", "mean_tesseract_confidence",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        # Stream report rows rather than retaining one Python object per PDF.
        for path in output_root.rglob("document.json"):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            aggregate["documents"] += 1
            aggregate["source_pages"] += int(summary.get("source_page_count", 0))
            for field in (
                "native_pages", "tesseract_accepted_pages", "surya_escalated_pages",
                "surya_quality_escalated_pages", "surya_structure_escalated_pages",
            ):
                aggregate[field] += int(summary.get(field, 0))
            aggregate["complete_documents" if summary.get("state") == "complete" else "incomplete_documents"] += 1
            writer.writerow({
                "source": summary.get("source", ""), "state": summary.get("state", ""),
                "source_page_count": summary.get("source_page_count", 0), "native_pages": summary.get("native_pages", 0),
                "tesseract_accepted_pages": summary.get("tesseract_accepted_pages", 0),
                "surya_escalated_pages": summary.get("surya_escalated_pages", 0),
                "surya_quality_escalated_pages": summary.get("surya_quality_escalated_pages", 0),
                "surya_structure_escalated_pages": summary.get("surya_structure_escalated_pages", 0),
                "mean_tesseract_confidence": summary.get("mean_tesseract_confidence", ""),
            })
    os.replace(temporary, csv_path)
    atomic_write_json(output_root / "batch_summary.json", aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="PDF or directory tree of PDFs")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ocr"))
    parser.add_argument("--dry-run", action="store_true", help="Inspect pages only; neither render nor write output")
    parser.add_argument("--no-resume", action="store_true", help="Refuse an existing matching job directory")
    parser.add_argument(
        "--reprocess-ocr", action="store_true",
        help="Repair existing Tesseract/Surya records while retaining native-text pages",
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--tesseract-psm", type=int, default=3)
    parser.add_argument("--tesseract-timeout", type=int, default=180, metavar="SECONDS")
    parser.add_argument("--surya-timeout", type=int, default=1_800, metavar="SECONDS")
    parser.add_argument("--surya-batch-size", type=int, default=4)
    parser.add_argument(
        "--surya-keep-server", action="store_true",
        help="Reuse Surya's host-wide inference server across batches; you must manage its shutdown",
    )
    parser.add_argument(
        "--structure-aware", action="store_true",
        help="Escalate Tesseract-accepted pages with table-like alignment or multiple text columns to Surya",
    )
    parser.add_argument("--min-native-chars", type=int, default=80)
    parser.add_argument("--min-native-words", type=int, default=12)
    parser.add_argument("--max-native-garbage-ratio", type=float, default=0.05)
    parser.add_argument("--dominant-image-ratio", type=float, default=0.55)
    parser.add_argument("--min-tesseract-chars", type=int, default=20)
    parser.add_argument("--min-tesseract-words", type=int, default=4)
    parser.add_argument("--min-mean-confidence", type=float, default=65.0)
    parser.add_argument("--min-confident-word-ratio", type=float, default=0.70)
    parser.add_argument("--confident-word-threshold", type=float, default=60.0)
    parser.add_argument("--max-tesseract-garbage-ratio", type=float, default=0.08)
    parser.add_argument("--min-plausible-word-ratio", type=float, default=0.70)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        language=args.language, render_dpi=args.dpi, tesseract_psm=args.tesseract_psm,
        tesseract_timeout_seconds=args.tesseract_timeout, surya_timeout_seconds=args.surya_timeout,
        surya_batch_size=args.surya_batch_size, surya_keep_server=args.surya_keep_server,
        structure_aware=args.structure_aware,
        min_native_chars=args.min_native_chars,
        min_native_words=args.min_native_words, max_native_garbage_ratio=args.max_native_garbage_ratio,
        dominant_image_ratio=args.dominant_image_ratio, min_tesseract_chars=args.min_tesseract_chars,
        min_tesseract_words=args.min_tesseract_words, min_mean_confidence=args.min_mean_confidence,
        min_confident_word_ratio=args.min_confident_word_ratio, confident_word_threshold=args.confident_word_threshold,
        max_tesseract_garbage_ratio=args.max_tesseract_garbage_ratio,
        min_plausible_word_ratio=args.min_plausible_word_ratio,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in the range [0, --shard-count)")
    if args.max_documents is not None and args.max_documents <= 0:
        raise SystemExit("--max-documents must be positive")
    if args.dpi <= 0 or args.surya_batch_size <= 0:
        raise SystemExit("--dpi and --surya-batch-size must be positive")
    for name in ("max_native_garbage_ratio", "dominant_image_ratio", "min_confident_word_ratio", "max_tesseract_garbage_ratio", "min_plausible_word_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if input_path.is_file() and input_path.suffix.lower() != ".pdf":
        raise SystemExit("Input file must be a PDF")
    config = config_from_args(args)
    input_root = input_path.parent if input_path.is_file() else input_path
    output_root = args.output_dir.resolve()
    selected = failures = 0
    dry_runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for pdf_path in iter_pdfs(input_path, output_root):
        if stable_path_key(pdf_path.resolve()) % args.shard_count != args.shard_index:
            continue
        if args.max_documents is not None and selected >= args.max_documents:
            break
        selected += 1
        print(f"[{selected}] {pdf_path}", file=sys.stderr, flush=True)
        try:
            summary = process_document(
                pdf_path, input_root=input_root, output_root=output_root, config=config,
                resume=not args.no_resume, dry_run=args.dry_run, reprocess_ocr=args.reprocess_ocr,
            )
            if args.dry_run:
                dry_runs.append(summary)
            else:
                print(
                    f"    pages={summary['source_page_count']} native={summary['native_pages']} "
                    f"tesseract={summary['tesseract_accepted_pages']} surya={summary['surya_escalated_pages']} "
                    f"state={summary['state']}", file=sys.stderr, flush=True,
                )
        except Exception as exc:  # a corpus worker continues after one bad PDF
            failures += 1
            print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    elapsed = round(time.perf_counter() - started, 3)
    if args.dry_run:
        print(json.dumps({"documents": dry_runs, "failures": failures, "runtime_seconds": elapsed}, indent=2))
    else:
        print(json.dumps({**write_batch_reports(output_root), "selected_documents": selected, "failures": failures, "runtime_seconds": elapsed}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
