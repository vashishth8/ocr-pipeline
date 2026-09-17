#!/usr/bin/env python3
"""Resumable, cost-aware OCR cascade for one PDF or a directory of PDFs.

Pipeline: PyMuPDF native text -> Tesseract 5 -> quality / optional structure
gate -> optional Surya fallback (or a compact-only stop).

Only the current page and one bounded optional fallback batch are rendered.
Every completed page is appended to a manifest, so a failed 500-page job
resumes without repeating completed work.

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
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from statistics import fmean, median
from typing import Any

from sttl.gates import (
    PipelineConfig,
    _unicode_word_char,
    _unicode_word_tokens,
    classify_page_signals,
    garbage_ratio,
    language_requests_hindi,
    pipeline_description,
    plausible_word_ratio,
    primary_ocr_engine,
    private_use_character_count,
    tesseract_quality,
)
from sttl.geometry import (
    PDF_POINT_FRAME,
    coerce_bbox,
    scale_bbox_to_pdf_points,
    scale_ocr_items_to_pdf_points,
    scale_polygon_to_pdf_points,
)
from sttl.reporting import (
    COMPLETE,
    append_jsonl,
    atomic_write_json,
    job_output_dir,
    load_completed_pages,
    source_identity,
)
from sttl.text import html_to_text, normalize_whitespace
from sttl.version import PIPELINE_VERSION

__all__ = [
    "_unicode_word_char",
    "classify_page_signals",
    "garbage_ratio",
    "pipeline_description",
    "plausible_word_ratio",
    "private_use_character_count",
    "scale_bbox_to_pdf_points",
    "scale_polygon_to_pdf_points",
    "summarise_document",
    "summarize_document",
    "tesseract_quality",
]

try:
    import pymupdf as fitz
except ImportError as exc:
    raise SystemExit("PyMuPDF is required: python -m pip install PyMuPDF") from exc


RICH_SCHEMA_VERSION = "cascade-ocr/rich-v1"
PAGE_STATES = ("DIGITAL", "SCANNED", "MIXED", "OCR_NEEDED")
VISUAL_BLOCK_TYPES = frozenset({"chart", "diagram", "figure", "image", "picture"})
MISSING_SOURCE_READING_ORDER = 10**9

LOGGER = logging.getLogger(__name__)

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

# Hybrid text is intentionally conservative. It is only useful if it leaves
# each textual region with one unambiguous owner and two engines substantially
# agree about its content.
HYBRID_MIN_TOKEN_F1 = 1.0
HYBRID_MIN_SEQUENCE_RATIO = 1.0
# Table cells are order-sensitive. Tesseract's global order may be
# column-major, so a table hybrid is allowed only after spatial row ordering
# produces the exact token sequence Surya identified.
HYBRID_TABLE_SEQUENCE_RATIO = 1.0
HYBRID_MAX_REGION_OVERLAP_RATIO = 0.15
HYBRID_MAX_UNASSIGNED_WORD_RATIO = 0.0
CRITICAL_VALUE_RE = re.compile(
    r"(?:\(\s*(?:[$€£₹]\s*)?[+\-−]?\d[\d,._:/\-]*\s*%?\s*\)|(?:[$€£₹]\s*)?[+\-−]?\d[\d,._:/\-]*\s*%?)",
    re.UNICODE,
)


@dataclass
class _PendingSuryaPage:
    """One rendered page retained until a bounded Surya batch is flushed."""

    page: int
    inspection: dict[str, Any]
    image_path: Path
    quality: dict[str, Any] | None
    raster: dict[str, Any]
    tesseract_text: str
    tesseract_words: list[dict[str, Any]]
    tesseract_blocks: list[dict[str, Any]]
    escalation_reason: str
    structure_gate: dict[str, Any] | None = None


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


def html_to_table(value: Any) -> dict[str, Any] | None:
    """Return the first table in a Surya HTML block, preserving cell spans."""
    if not isinstance(value, str) or "<table" not in value.lower():
        return None
    parser = _HTMLTableExtractor()
    parser.feed(value)
    parser.close()
    return parser.tables[0] if parser.tables else None


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
            "coordinate_frame": PDF_POINT_FRAME,
            "pdf_coordinate_convention": "pymupdf_unrotated_page_points",
            "dpi": dpi,
            "raster_width": pixmap.width,
            "raster_height": pixmap.height,
            "pdf_bbox": [
                float(target_rect.x0),
                float(target_rect.y0),
                float(target_rect.x1),
                float(target_rect.y1),
            ],
            "rendered_pdf_bbox": [
                float(rendered_rect.x0),
                float(rendered_rect.y0),
                float(rendered_rect.x1),
                float(rendered_rect.y1),
            ],
            "derotation_matrix": [float(value) for value in page.derotation_matrix],
            "page_rotation": page.rotation,
        }
    finally:
        del pixmap


def _float_bbox(value: Any) -> list[float] | None:
    if isinstance(value, fitz.Rect):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    return coerce_bbox(value)


def executable_for(name: str) -> str | None:
    """Prefer the pinned current-venv executable, then fall back to PATH."""
    venv_executable = Path(sys.executable).parent / name
    if venv_executable.is_file():
        return str(venv_executable)
    return shutil.which(name)


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
        order_number = _as_int(source_order, default=MISSING_SOURCE_READING_ORDER)
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
        for key in (
            "raw_label",
            "html",
            "source",
            "coordinate_space",
            "coordinate_frame",
            "skipped",
            "error",
            "retain_empty",
            "layout_engine",
            "text_engine",
            "text_provenance",
            "surya_structure",
        ):
            if key in raw:
                block[key] = raw[key]
        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)):
            block["confidence"] = float(confidence)
        polygon = raw.get("polygon")
        if isinstance(polygon, (list, tuple)):
            block["polygon"] = [
                list(point[:2])
                for point in polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        source_bbox = raw.get("source_bbox")
        if isinstance(source_bbox, (list, tuple)) and len(source_bbox) >= 4:
            try:
                block["source_bbox"] = [float(value) for value in source_bbox[:4]]
            except (TypeError, ValueError):
                pass
        source_polygon = raw.get("source_polygon")
        if isinstance(source_polygon, (list, tuple)):
            block["source_polygon"] = [
                list(point[:2])
                for point in source_polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
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


def parse_tesseract_tsv_layout(
    tsv: str,
) -> tuple[str, list[float], list[dict[str, Any]], list[dict[str, Any]]]:
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
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
        raw_blocks.append(block)
    return (
        " ".join(word["text"] for word in words).strip(),
        confidences,
        words,
        canonical_blocks(raw_blocks),
    )


def _tesseract_line_words(
    words: Sequence[dict[str, Any]],
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
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
    boxes = [
        item["bbox"]
        for item in items
        if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4
    ]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
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
        for previous, current in zip(line_words, line_words[1:], strict=False):
            gap = current["bbox"][0] - previous["bbox"][2]
            if gap >= gap_threshold:
                points.append(
                    {
                        "x": current["bbox"][0],
                        "y": line_bbox[1],
                        "line": line_key,
                        "gap": gap,
                    }
                )

    # Keep clustering deterministic and use the nearest cluster when two
    # candidate starts are close enough.  This tolerates minor OCR skew while
    # retaining distinct table columns.
    clusters: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: (item["x"], item["y"])):
        candidates = [
            cluster for cluster in clusters if abs(point["x"] - cluster["mean_x"]) <= tolerance
        ]
        cluster = (
            min(candidates, key=lambda item: abs(point["x"] - item["mean_x"]))
            if candidates
            else None
        )
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
        evidence.append(
            {
                "start_x": round(cluster["mean_x"], 3),
                "start_ratio": round(start_ratio, 6),
                "line_count": len(lines_seen),
                "vertical_span": round(vertical_span, 3),
                "vertical_span_ratio": round(vertical_span / raster_height, 6),
                "mean_gap": round(fmean(point["gap"] for point in cluster_points), 3),
                "mean_gap_ratio": round(
                    fmean(point["gap"] for point in cluster_points) / raster_width, 6
                ),
            }
        )
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
            or not STRUCTURE_MIN_COLUMN_WIDTH_RATIO
            <= width_ratio
            <= STRUCTURE_MAX_COLUMN_WIDTH_RATIO
            or height_ratio < STRUCTURE_MIN_COLUMN_HEIGHT_RATIO
        ):
            continue
        candidates.append(
            {
                "block": block_number,
                "bbox": bbox,
                "line_count": line_counts[block_number],
                "word_count": len(block_words),
            }
        )

    evidence: list[dict[str, Any]] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
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
            if (
                shorter_height <= 0
                or overlap / shorter_height < STRUCTURE_MIN_COLUMN_VERTICAL_OVERLAP
            ):
                continue
            evidence.append(
                {
                    "left_block": left["block"],
                    "right_block": right["block"],
                    "horizontal_gap": round(horizontal_gap, 3),
                    "horizontal_gap_ratio": round(horizontal_gap / raster_width, 6),
                    "vertical_overlap_ratio": round(overlap / shorter_height, 6),
                    "left_line_count": left["line_count"],
                    "right_line_count": right["line_count"],
                    "left_word_count": left["word_count"],
                    "right_word_count": right["word_count"],
                }
            )
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
        LOGGER.debug(
            "PyMuPDF table detection failed; continuing without table evidence", exc_info=True
        )
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


def tesseract_page(
    image_path: Path, config: PipelineConfig
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the cheap first-pass OCR engine and return its ordered layout.

    The quality gate below—not a file-name rule or a second OCR run—decides
    whether this output is accepted. Rejected pages either queue for the
    optional Surya fallback or remain unselected audit evidence in compact-only
    mode.
    """
    executable = executable_for("tesseract")
    if executable is None:
        raise RuntimeError("Tesseract 5 executable was not found on PATH")
    started = time.perf_counter()
    try:
        command = [executable, str(image_path), "stdout"]
        if config.tessdata_dir:
            command.extend(["--tessdata-dir", config.tessdata_dir])
        # Use the built-in TSV switch rather than the ``tsv`` config file.
        # A workspace-local --tessdata-dir contains language data only, while
        # the config-file form would incorrectly look for ``configs/tsv`` in
        # that same directory.
        command.extend(
            [
                "-l",
                config.language,
                "--psm",
                str(config.tesseract_psm),
                "-c",
                "tessedit_create_tsv=1",
            ]
        )
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.tesseract_timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if "Failed loading language" in stderr or "Could not initialize tesseract" in stderr:
            verification_command = "tesseract --list-langs"
            if config.tessdata_dir:
                verification_command += f" --tessdata-dir {shlex.quote(config.tessdata_dir)}"
            raise RuntimeError(
                f"Tesseract language data for {config.language!r} is unavailable. "
                "Install the matching traineddata file, then verify it with "
                f"`{verification_command}`."
            ) from exc
        raise
    text, confidences, words, blocks = parse_tesseract_tsv_layout(result.stdout)
    quality = tesseract_quality(text, confidences, config)
    quality["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return text, quality, words, blocks


def extract_surya_text(prediction: Any) -> str:
    if not isinstance(prediction, dict):
        return ""
    return "\n".join(
        text
        for block in prediction.get("blocks", [])
        if isinstance(block, dict) and not block.get("error")
        if (text := html_to_text(block.get("html")))
    ).strip()


def bbox_from_polygon(value: Any) -> list[float] | None:
    """Accept Surya's bbox or either common polygon representation."""
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 4
        and all(isinstance(v, (int, float)) for v in value[:4])
    ):
        return [float(value[index]) for index in range(4)]
    if isinstance(value, (list, tuple)):
        points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
        if points:
            try:
                return [
                    min(float(point[0]) for point in points),
                    min(float(point[1]) for point in points),
                    max(float(point[0]) for point in points),
                    max(float(point[1]) for point in points),
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


def _bbox_area(bbox: Any) -> float:
    value = _float_bbox(bbox)
    if value is None:
        return 0.0
    return max(0.0, value[2] - value[0]) * max(0.0, value[3] - value[1])


def _bbox_intersection_area(left: Any, right: Any) -> float:
    first = _float_bbox(left)
    second = _float_bbox(right)
    if first is None or second is None:
        return 0.0
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )


def _word_region_overlap_score(word_bbox: Any, region_bbox: Any) -> float | None:
    """Score a Tesseract word's membership in one Surya region.

    Small coordinate drift is normal between two OCR engines. A word is
    therefore accepted when its center falls in the region or when at least
    half of its area overlaps it. The score prefers center containment, then
    overlap coverage. Hybridization rejects pages with overlapping text
    regions before invoking this function, so an accepted word has one clear
    authoritative owner.
    """
    word = _float_bbox(word_bbox)
    region = _float_bbox(region_bbox)
    if word is None or region is None:
        return None
    intersection_width = max(0.0, min(word[2], region[2]) - max(word[0], region[0]))
    intersection_height = max(0.0, min(word[3], region[3]) - max(word[1], region[1]))
    intersection = intersection_width * intersection_height
    word_area = _bbox_area(word)
    if intersection <= 0.0 or word_area <= 0.0:
        return None
    coverage = intersection / word_area
    center_x = (word[0] + word[2]) / 2.0
    center_y = (word[1] + word[3]) / 2.0
    center_inside = region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]
    if not center_inside and coverage < 0.5:
        return None
    return (2.0 if center_inside else 0.0) + coverage


def _surya_text_region(block: dict[str, Any]) -> bool:
    """Whether a Surya region is safe to receive OCR text from Tesseract."""
    if block.get("skipped") or block.get("error"):
        return False
    if not str(block.get("text", "")).strip():
        return False
    label = str(block.get("block_type") or block.get("type") or "").strip().lower()
    return label not in VISUAL_BLOCK_TYPES


def _is_table_region(block: dict[str, Any]) -> bool:
    """Recognize the canonical and raw spellings of Surya table blocks."""
    label = str(block.get("block_type") or block.get("type") or "")
    return re.sub(r"[\W_]+", "", label.casefold()) == "table"


def _reading_ordered_words(
    assigned: Sequence[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    return sorted(
        assigned,
        key=lambda item: (
            0 if isinstance(item[1].get("reading_order"), (int, float)) else 1,
            float(item[1].get("reading_order", 0))
            if isinstance(item[1].get("reading_order"), (int, float))
            else 0.0,
            item[0],
        ),
    )


def _spatial_row_ordered_table_words(
    assigned: Sequence[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    """Recover row-major table order from word geometry when it is available.

    Tesseract can enumerate a multi-column table column-by-column even when
    its word boxes clearly encode rows. Group nearby vertical centers into
    visual rows and read each row left-to-right. The hybrid still requires an
    exact Surya token sequence afterward, so uncertain row grouping declines
    safely rather than changing the table's authoritative order.
    """
    fallback = _reading_ordered_words(assigned)
    positioned: list[tuple[int, dict[str, Any], list[float], float]] = []
    for index, word in fallback:
        bbox = _float_bbox(word.get("bbox"))
        if bbox is None:
            return fallback
        positioned.append((index, word, bbox, (bbox[1] + bbox[3]) / 2.0))
    if not positioned:
        return fallback

    typical_height = median(max(1.0, bbox[3] - bbox[1]) for _, _, bbox, _ in positioned)
    line_tolerance = max(1.0, typical_height * 0.75)
    rows: list[dict[str, Any]] = []
    for item in sorted(positioned, key=lambda value: (value[3], value[2][0], value[0])):
        if not rows or item[3] - rows[-1]["center_y"] > line_tolerance:
            rows.append({"center_y": item[3], "items": [item]})
            continue
        row = rows[-1]
        row["items"].append(item)
        row["center_y"] = sum(value[3] for value in row["items"]) / len(row["items"])

    return [
        (index, word)
        for row in rows
        for index, word, _bbox, _center_y in sorted(
            row["items"],
            key=lambda value: (value[2][0], value[2][1], value[0]),
        )
    ]


def _ordered_words_for_surya_region(
    block: dict[str, Any],
    assigned: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any]]], str]:
    if _is_table_region(block):
        return _spatial_row_ordered_table_words(assigned), "spatial_row_major"
    return _reading_ordered_words(assigned), "tesseract_reading_order"


def _overlapping_surya_text_regions(
    blocks: Sequence[dict[str, Any]],
    eligible: Sequence[int],
) -> list[dict[str, Any]]:
    """Return meaningful textual-region overlaps that make ownership unsafe."""
    overlaps: list[dict[str, Any]] = []
    for position, left_index in enumerate(eligible):
        left = blocks[left_index]
        left_area = _bbox_area(left.get("bbox"))
        if not left_area:
            continue
        for right_index in eligible[position + 1 :]:
            right_area = _bbox_area(blocks[right_index].get("bbox"))
            if not right_area:
                continue
            intersection = _bbox_intersection_area(
                left.get("bbox"), blocks[right_index].get("bbox")
            )
            overlap_ratio = intersection / min(left_area, right_area) if intersection else 0.0
            if overlap_ratio >= HYBRID_MAX_REGION_OVERLAP_RATIO:
                overlaps.append(
                    {
                        "first_reading_order": left.get("reading_order"),
                        "second_reading_order": blocks[right_index].get("reading_order"),
                        "overlap_ratio_of_smaller_region": round(overlap_ratio, 6),
                    }
                )
    return overlaps


def _text_tokens_for_agreement(text: str) -> list[str]:
    return _unicode_word_tokens(normalize_whitespace(text).casefold())


def _critical_value_tokens(text: str) -> list[str]:
    """Keep sign-, currency-, decimal-, and percentage-bearing values intact."""
    return [
        re.sub(r"\s+", "", token).replace("−", "-").casefold()
        for token in CRITICAL_VALUE_RE.findall(normalize_whitespace(text))
    ]


def text_agreement(left: str, right: str) -> dict[str, float | int | bool]:
    """Report token-content and token-sequence agreement transparently.

    Bag agreement alone cannot distinguish a row-major table from a
    column-major transcription of the same cells.  The sequence metric is a
    token-level ``SequenceMatcher`` ratio, recorded alongside the less strict
    bag metrics so a hybrid decision can be audited without calling it CER.
    """
    left_tokens = _text_tokens_for_agreement(left)
    right_tokens = _text_tokens_for_agreement(right)
    left_critical_values = _critical_value_tokens(left)
    right_critical_values = _critical_value_tokens(right)
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    shared = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    precision = shared / len(right_tokens) if right_tokens else 0.0
    recall = shared / len(left_tokens) if left_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sequence_matches = sum(
        block.size
        for block in SequenceMatcher(
            a=left_tokens,
            b=right_tokens,
            autojunk=False,
        ).get_matching_blocks()
    )
    sequence_ratio = (
        2 * sequence_matches / (len(left_tokens) + len(right_tokens))
        if left_tokens or right_tokens
        else 0.0
    )
    return {
        "surya_tokens": len(left_tokens),
        "tesseract_tokens": len(right_tokens),
        "shared_tokens": shared,
        "token_precision": round(precision, 6),
        "token_recall": round(recall, 6),
        "token_f1": round(f1, 6),
        "token_jaccard": round(shared / union, 6) if union else 0.0,
        "sequence_matches": sequence_matches,
        "sequence_ratio": round(sequence_ratio, 6),
        # Tokenization above intentionally ignores ordinary punctuation for
        # OCR tolerance, but never allow it to erase a financial sign,
        # decimal, date separator, currency marker, or percentage.
        "surya_critical_value_tokens": len(left_critical_values),
        "tesseract_critical_value_tokens": len(right_critical_values),
        "critical_value_tokens_match": left_critical_values == right_critical_values,
    }


def hybridize_surya_layout_with_tesseract(
    surya_blocks: Sequence[dict[str, Any]],
    tesseract_words: Sequence[dict[str, Any]],
    *,
    confident_word_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use Tesseract text only when the complete Surya text layout is safe.

    A partial region replacement creates ambiguous ownership: unassigned Surya
    text can be duplicated or silently lost, especially with nested layout
    blocks. This function therefore applies a hybrid only when every
    text-bearing, non-visual Surya region has high-confidence overlapping
    Tesseract words with the same canonical token sequence, and no competing
    or unassigned text. Otherwise the caller receives unchanged Surya output.
    """
    source_blocks = [dict(block) for block in surya_blocks if isinstance(block, dict)]
    eligible = [index for index, block in enumerate(source_blocks) if _surya_text_region(block)]
    base_details: dict[str, Any] = {
        "applied": False,
        "strategy": "all_or_nothing_exact_tesseract_token_sequence_in_surya_regions",
        "layout_engine": "surya",
        "text_engine": "tesseract5",
        "confident_word_threshold": confident_word_threshold,
        "minimum_token_f1": HYBRID_MIN_TOKEN_F1,
        "minimum_sequence_ratio": HYBRID_MIN_SEQUENCE_RATIO,
        "table_sequence_requirement": "exact_token_sequence_after_spatial_row_ordering",
        "all_high_confidence_tesseract_words_must_be_assigned": True,
        "maximum_region_overlap_ratio": HYBRID_MAX_REGION_OVERLAP_RATIO,
        "maximum_unassigned_word_ratio": HYBRID_MAX_UNASSIGNED_WORD_RATIO,
        "eligible_surya_text_regions": len(eligible),
    }
    non_hybrid_text_regions = [
        {
            "reading_order": block.get("reading_order"),
            "block_type": block.get("block_type"),
        }
        for block in source_blocks
        if str(block.get("text", "")).strip() and not _surya_text_region(block)
    ]
    # A visual description or failed region with text would remain Surya-owned
    # in final page text. Do not claim a wholly Tesseract text page in that
    # mixed case; retain the unmodified Surya result instead.
    if non_hybrid_text_regions:
        return canonical_blocks(source_blocks), {
            **base_details,
            "skip_reason": "non_tesseract_textual_surya_regions",
            "non_hybrid_text_regions": non_hybrid_text_regions,
            "hybridized_regions": 0,
            "selected_tesseract_words": 0,
        }
    if not eligible:
        return canonical_blocks(source_blocks), {
            **base_details,
            "skip_reason": "no_textual_surya_regions",
            "hybridized_regions": 0,
            "selected_tesseract_words": 0,
        }

    overlapping_regions = _overlapping_surya_text_regions(source_blocks, eligible)
    if overlapping_regions:
        return canonical_blocks(source_blocks), {
            **base_details,
            "skip_reason": "overlapping_surya_text_regions",
            "overlapping_surya_text_regions": overlapping_regions,
            "hybridized_regions": 0,
            "selected_tesseract_words": 0,
        }

    assignments: dict[int, list[tuple[int, dict[str, Any]]]] = {index: [] for index in eligible}
    high_confidence_words = 0
    no_geometry_words = 0
    low_confidence_words = 0
    unassigned_high_confidence_words = 0

    for word_index, word in enumerate(tesseract_words):
        if not isinstance(word, dict) or not str(word.get("text", "")).strip():
            continue
        confidence = word.get("confidence")
        if not isinstance(confidence, (int, float)) or float(confidence) < confident_word_threshold:
            low_confidence_words += 1
            continue
        high_confidence_words += 1
        if _float_bbox(word.get("bbox")) is None:
            no_geometry_words += 1
            unassigned_high_confidence_words += 1
            continue
        choices: list[tuple[float, int]] = []
        for block_index in eligible:
            score = _word_region_overlap_score(
                word.get("bbox"), source_blocks[block_index].get("bbox")
            )
            if score is not None:
                choices.append((score, block_index))
        if not choices:
            unassigned_high_confidence_words += 1
            continue
        _, selected = max(choices, key=lambda item: (item[0], -item[1]))
        assignments[selected].append((word_index, word))

    region_texts: dict[int, dict[str, Any]] = {}
    rejected_regions: list[dict[str, Any]] = []
    for block_index, assigned in assignments.items():
        block = source_blocks[block_index]
        ordered_assigned, tesseract_ordering = _ordered_words_for_surya_region(block, assigned)
        selected_text = normalize_whitespace(
            " ".join(str(word.get("text", "")).strip() for _, word in ordered_assigned)
        )
        surya_text = str(block.get("text", "")).strip()
        agreement = text_agreement(surya_text, selected_text)
        required_sequence_ratio = (
            HYBRID_TABLE_SEQUENCE_RATIO if _is_table_region(block) else HYBRID_MIN_SEQUENCE_RATIO
        )
        if (
            not selected_text
            or agreement["token_f1"] < HYBRID_MIN_TOKEN_F1
            or agreement["sequence_ratio"] < required_sequence_ratio
            or not agreement["critical_value_tokens_match"]
        ):
            reasons = []
            if not selected_text:
                reasons.append("no_high_confidence_tesseract_text")
            if agreement["token_f1"] < HYBRID_MIN_TOKEN_F1:
                reasons.append("low_token_f1")
            if agreement["sequence_ratio"] < required_sequence_ratio:
                reasons.append("low_token_sequence_ratio")
            if not agreement["critical_value_tokens_match"]:
                reasons.append("critical_value_token_mismatch")
            rejected_regions.append(
                {
                    "reading_order": block.get("reading_order"),
                    "tesseract_word_count": len(assigned),
                    "token_f1": agreement["token_f1"],
                    "sequence_ratio": agreement["sequence_ratio"],
                    "required_sequence_ratio": required_sequence_ratio,
                    "critical_value_tokens_match": agreement["critical_value_tokens_match"],
                    "rejection_reasons": reasons,
                }
            )
            continue
        region_texts[block_index] = {
            "text": selected_text,
            "surya_text": surya_text,
            "agreement": agreement,
            "required_sequence_ratio": required_sequence_ratio,
            "tesseract_ordering": tesseract_ordering,
            "reading_orders": [
                word.get("reading_order")
                for _, word in ordered_assigned
                if word.get("reading_order") is not None
            ],
            "word_count": len(assigned),
        }

    unassigned_ratio = (
        unassigned_high_confidence_words / high_confidence_words if high_confidence_words else 1.0
    )
    if rejected_regions or unassigned_ratio > HYBRID_MAX_UNASSIGNED_WORD_RATIO:
        skip_reason = (
            "insufficient_per_region_agreement"
            if rejected_regions
            else "too_many_unassigned_tesseract_words"
        )
        return canonical_blocks(source_blocks), {
            **base_details,
            "skip_reason": skip_reason,
            "hybridized_regions": 0,
            "selected_tesseract_words": 0,
            "high_confidence_tesseract_words": high_confidence_words,
            "low_confidence_tesseract_words": low_confidence_words,
            "tesseract_words_without_geometry": no_geometry_words,
            "unassigned_high_confidence_tesseract_words": unassigned_high_confidence_words,
            "unassigned_high_confidence_tesseract_word_ratio": round(unassigned_ratio, 6),
            "rejected_regions": rejected_regions,
        }

    # Each textual Surya region is now covered, so page-level engine ownership
    # is honest: all final text originates from Tesseract while Surya supplies
    # semantic type and geometry. Move raw table/HTML text out of authoritative
    # fields to prevent a consumer from reading two contradictory values.
    selected_words = 0
    for block_index, region in region_texts.items():
        block = source_blocks[block_index]
        surya_structure = {key: block.pop(key) for key in ("html", "table") if key in block}
        if surya_structure:
            block["surya_structure"] = surya_structure
        block["text"] = region["text"]
        block["layout_engine"] = "surya"
        block["text_engine"] = "tesseract5"
        block["text_provenance"] = {
            "strategy": base_details["strategy"],
            "layout_engine": "surya",
            "text_engine": "tesseract5",
            "surya_text": region["surya_text"],
            "tesseract_confidence_threshold": confident_word_threshold,
            "tesseract_word_count": region["word_count"],
            "tesseract_word_reading_orders": region["reading_orders"],
            "minimum_token_f1": HYBRID_MIN_TOKEN_F1,
            "minimum_sequence_ratio": region["required_sequence_ratio"],
            "tesseract_word_ordering": region["tesseract_ordering"],
            "agreement": region["agreement"],
        }
        selected_words += region["word_count"]

    return canonical_blocks(source_blocks), {
        **base_details,
        "applied": True,
        "hybridized_regions": len(region_texts),
        "selected_tesseract_words": selected_words,
        "high_confidence_tesseract_words": high_confidence_words,
        "low_confidence_tesseract_words": low_confidence_words,
        "tesseract_words_without_geometry": no_geometry_words,
        "unassigned_high_confidence_tesseract_words": unassigned_high_confidence_words,
        "unassigned_high_confidence_tesseract_word_ratio": round(unassigned_ratio, 6),
        "all_eligible_surya_regions_hybridized": len(region_texts) == len(eligible),
    }


def surya_batch(
    input_dir: Path, output_dir: Path, config: PipelineConfig
) -> dict[str, dict[str, Any]]:
    """Run one isolated, bounded Surya batch and normalize its raw CLI payload.

    This function retains raw model output but does not select authoritative
    text. ``flush_surya`` makes that route-specific decision once the evidence
    is available for a primary, quality-fallback, or structure-fallback page.
    """
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
    atomic_write_json(
        result_dir / "timing.json",
        {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "input_images": len(list(input_dir.glob("*.png"))),
        },
    )
    return output


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
            raise RuntimeError(
                f"Job already exists at {job_dir}; remove --no-resume or use a new output directory"
            )
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
            atomic_write_json(
                job_path,
                {
                    **existing,
                    "config": expected["config"],
                    "pipeline_version": PIPELINE_VERSION,
                    "upgraded_at_epoch": time.time(),
                },
            )
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
    surya_candidate: dict[str, Any] | None = None,
    hybrid_text: dict[str, Any] | None = None,
    escalation_reason: str | None = None,
    structure_gate: dict[str, Any] | None = None,
    executed_route: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "page": page,
        "status": COMPLETE,
        "classification": inspection["classification"],
        "route": executed_route or inspection["route"],
        "engine": engine,
        "outcome": outcome,
        "signals": inspection["signals"],
        "native_text": inspection["native_text"],
        "text": text,
        "completed_at_epoch": time.time(),
    }
    # The page classifier historically calls every OCR-needed page a
    # ``tesseract`` route. Preserve that recommendation when Hindi primary
    # routing deliberately selects Surya instead, while making the executed
    # route unambiguous to downstream audit consumers.
    if executed_route is not None and executed_route != inspection["route"]:
        record["classifier_route"] = inspection["route"]
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
    if surya_candidate is not None:
        record["surya_candidate"] = surya_candidate
    if hybrid_text is not None:
        record["hybrid_text"] = hybrid_text
    return record


def summarize_document(
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
    primary_surya_pages = quality_escalations = structure_escalations = hybrid_text_pages = 0
    for record in records.values():
        if record.get("classification") in states:
            states[record["classification"]] += 1
        outcome = str(record.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        quality = record.get("tesseract_quality")
        if isinstance(quality, dict) and isinstance(
            quality.get("mean_word_confidence"), (int, float)
        ):
            confidences.append(float(quality["mean_word_confidence"]))
        # Older manifests did not distinguish fallback reasons.  Their sole
        # historical Surya route was the quality gate, so classify them there
        # while retaining the long-standing total below.
        if record.get("outcome") == "surya_escalated":
            if record.get("escalation_reason") == "structure":
                structure_escalations += 1
            else:
                quality_escalations += 1
        if record.get("outcome") == "surya_primary":
            primary_surya_pages += 1
        if isinstance(record.get("hybrid_text"), dict) and record["hybrid_text"].get("applied"):
            hybrid_text_pages += 1
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
        "tesseract_rejected_no_fallback_pages": outcomes.get("tesseract_rejected_no_fallback", 0),
        "surya_primary_pages": primary_surya_pages,
        "surya_escalated_pages": outcomes.get("surya_escalated", 0),
        "surya_quality_escalated_pages": quality_escalations,
        "surya_structure_escalated_pages": structure_escalations,
        "surya_hybrid_text_pages": hybrid_text_pages,
        "mean_tesseract_confidence": round(fmean(confidences), 3) if confidences else None,
        "updated_at_epoch": time.time(),
    }


# Retain the established British spelling for downstream scripts while making
# the primary API consistent with the rest of the project naming convention.
summarise_document = summarize_document


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
        raw_blocks.append(
            {
                "block_type": "Text",
                "type": "text",
                "text": str(block[4]).strip(),
                "bbox": [block[0], block[1], block[2], block[3]],
                "reading_order": source_order,
            }
        )
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
                output
                for span in raw_line.get("spans", [])
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
        "coordinate_frame": PDF_POINT_FRAME,
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
    boxes = [
        block["bbox"]
        for block in blocks
        if isinstance(block.get("bbox"), list) and len(block["bbox"]) == 4
    ]
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
        blocks = (
            native_blocks(page)
            if record.get("engine") == "pymupdf"
            else canonical_blocks(
                [
                    {
                        "block_type": "Text",
                        "type": "text",
                        "text": record.get("text", ""),
                        "reading_order": 1,
                    }
                ]
            )
        )
    blocks = canonical_blocks(blocks)
    hybrid = record.get("hybrid_text")
    hybrid_applied = isinstance(hybrid, dict) and bool(hybrid.get("applied"))
    text_engine = "tesseract5" if hybrid_applied else record.get("engine")
    entry: dict[str, Any] = {
        "page": record["page"],
        "text": record.get("text", ""),
        "bbox": page_bbox(page),
        "coordinate_space": "pdf_points",
        "coordinate_frame": PDF_POINT_FRAME,
        "block_type": "Page",
        "reading_order": record["page"],
        "blocks": blocks,
        "reading_order_check": reading_order_check(blocks),
        "metadata": {
            "engine": record.get("engine"),
            "layout_engine": "surya" if hybrid_applied else record.get("engine"),
            "text_engine": text_engine,
            "classification": record.get("classification"),
            "route": record.get("route"),
            "outcome": record.get("outcome"),
            "signals": record.get("signals"),
        },
    }
    if isinstance(record.get("classifier_route"), str):
        entry["metadata"]["classifier_route"] = record["classifier_route"]
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
    if isinstance(record.get("hybrid_text"), dict):
        entry["metadata"]["hybrid_text"] = record["hybrid_text"]
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
        pages = [
            normalized_page(records[number], pdf[number - 1])
            for number in range(1, pdf.page_count + 1)
        ]
    page_numbers = [entry["page"] for entry in pages]
    expected_pages = list(range(1, len(pages) + 1))
    order_report = {
        "status": "maintained" if page_numbers == expected_pages else "invalid",
        "page_count": len(pages),
        "contiguous_page_order": page_numbers == expected_pages,
        "pages_with_block_order_error": [
            entry["page"]
            for entry in pages
            if entry["reading_order_check"]["status"] != "maintained"
        ],
        "pages_recommended_for_visual_review": [
            entry["page"]
            for entry in pages
            if entry["reading_order_check"]["visual_review_recommended"]
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
            "pipeline": pipeline_description(config),
            "routing": {
                "classification_counts": summary["classification_counts"],
                "outcome_counts": summary["outcome_counts"],
                "primary_ocr_engine": primary_ocr_engine(config),
                "fallback_engine": config.fallback_engine,
                "tesseract_rejected_no_fallback_pages": summary[
                    "tesseract_rejected_no_fallback_pages"
                ],
                "surya_primary_pages": summary["surya_primary_pages"],
                "surya_quality_escalated_pages": summary["surya_quality_escalated_pages"],
                "surya_structure_escalated_pages": summary["surya_structure_escalated_pages"],
                "surya_hybrid_text_pages": summary["surya_hybrid_text_pages"],
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
            "structure_gate": {
                "enabled": config.structure_aware,
                "hybrid_text_enabled": config.structure_aware and config.structure_hybrid_text,
            },
            "reading_order": order_report,
        },
    }
    atomic_write_json(job_dir / f"{pdf_path.stem}_cascade.json", normalized)
    atomic_write_json(job_dir / "reading_order.json", order_report)
    return order_report


def tesseract_layer(record: dict[str, Any]) -> dict[str, Any]:
    """Expose raw Tesseract evidence without mislabeling it as final text."""
    candidate = record.get("tesseract_candidate")
    if isinstance(candidate, dict):
        hybrid = record.get("hybrid_text")
        hybrid_applied = isinstance(hybrid, dict) and bool(hybrid.get("applied"))
        return {
            "attempted": True,
            # The raw candidate can differ from the reconstructed hybrid text
            # after region assignment / table row ordering. Only the top-level
            # authoritative layer is selected in that case.
            "selected": False,
            "selected_layout": False,
            "selected_text": False,
            "contributes_to_authoritative_layout": False,
            "contributes_to_authoritative_text": hybrid_applied,
            "text": candidate.get("text", ""),
            "quality": candidate.get("quality"),
            "words": candidate.get("words", []),
            "blocks": candidate.get("blocks", []),
            "hybrid_text": hybrid,
        }
    if record.get("engine") == "tesseract5":
        return {
            "attempted": True,
            "selected": True,
            "selected_layout": True,
            "selected_text": True,
            "text": record.get("text", ""),
            "quality": record.get("tesseract_quality"),
            "words": record.get("words", []),
            "blocks": record.get("blocks", []),
        }
    return {"attempted": False, "selected": False}


def surya_layer(record: dict[str, Any]) -> dict[str, Any]:
    """Expose Surya's semantic blocks, including its original HTML tables."""
    selected = record.get("engine") == "surya"
    candidate = record.get("surya_candidate")
    hybrid = record.get("hybrid_text")
    if isinstance(candidate, dict):
        return {
            "attempted": True,
            # This is the raw Surya candidate. The selected layout is the
            # top-level authoritative hybrid, which may omit raw HTML fields
            # to avoid conflicting text values.
            "selected": False,
            "selected_layout": False,
            "selected_text": False,
            "contributes_to_authoritative_layout": True,
            "contributes_to_authoritative_text": False,
            "text": candidate.get("text", ""),
            "blocks": candidate.get("blocks", []),
            "hybrid_text": hybrid,
        }
    return {
        "attempted": selected,
        "selected": selected,
        "selected_layout": selected,
        "selected_text": selected,
        "text": record.get("text", "") if selected else "",
        "blocks": record.get("blocks", []) if selected else [],
        "hybrid_text": hybrid if isinstance(hybrid, dict) else None,
    }


def rich_page(record: dict[str, Any], page: fitz.Page) -> dict[str, Any]:
    """Build a multi-engine page with one selected authoritative result.

    A validated hybrid can combine Surya layout with region-assigned Tesseract
    text. Raw engine layers remain evidence rather than selected output, and
    explicitly state which part of the authoritative result they contributed.
    """
    authoritative = normalized_page(record, page)
    routing: dict[str, Any] = {
        "classification": record.get("classification"),
        "route": record.get("route"),
        "signals": record.get("signals"),
        "tesseract_quality": record.get("tesseract_quality"),
        "raster": record.get("raster"),
    }
    if isinstance(record.get("classifier_route"), str):
        routing["classifier_route"] = record["classifier_route"]
    if isinstance(record.get("structure_gate"), dict):
        routing["structure_gate"] = record["structure_gate"]
    if isinstance(record.get("escalation_reason"), str):
        routing["escalation_reason"] = record["escalation_reason"]
    hybrid = record.get("hybrid_text")
    if isinstance(hybrid, dict):
        routing["hybrid_text"] = hybrid
    hybrid_applied = isinstance(hybrid, dict) and bool(hybrid.get("applied"))
    text_engine = "tesseract5" if hybrid_applied else record.get("engine")
    return {
        "page": record["page"],
        "bbox": page_bbox(page),
        "coordinate_space": "pdf_points",
        "coordinate_frame": PDF_POINT_FRAME,
        "authoritative": {
            "engine": record.get("engine"),
            "layout_engine": "surya" if hybrid_applied else record.get("engine"),
            "text_engine": text_engine,
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
        pages = [
            rich_page(records[number], pdf[number - 1]) for number in range(1, pdf.page_count + 1)
        ]
    output_path = job_dir / f"{pdf_path.stem}_rich.json"
    atomic_write_json(
        output_path,
        {
            "schema_version": RICH_SCHEMA_VERSION,
            "engine": "cascade",
            "source": pdf_path.name,
            "page_count": len(pages),
            "coordinate_space": "pdf_points",
            "coordinate_frame": PDF_POINT_FRAME,
            "pages": pages,
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "pipeline": pipeline_description(config),
                "authoritative_selection": "one engine per page, except accepted Tesseract text may populate Surya regions after a structure escalation; raw evidence is retained in layers",
                "ocr_coordinate_transform": "source_bbox/source_polygon are raster pixels; bbox/polygon are PDF points",
                "routing": {
                    "classification_counts": summary["classification_counts"],
                    "outcome_counts": summary["outcome_counts"],
                    "primary_ocr_engine": primary_ocr_engine(config),
                    "fallback_engine": config.fallback_engine,
                    "tesseract_rejected_no_fallback_pages": summary[
                        "tesseract_rejected_no_fallback_pages"
                    ],
                    "surya_primary_pages": summary["surya_primary_pages"],
                    "surya_quality_escalated_pages": summary["surya_quality_escalated_pages"],
                    "surya_structure_escalated_pages": summary["surya_structure_escalated_pages"],
                    "surya_hybrid_text_pages": summary["surya_hybrid_text_pages"],
                },
                "tesseract_quality_gate": {
                    "min_mean_confidence": config.min_mean_confidence,
                    "min_confident_word_ratio": config.min_confident_word_ratio,
                    "confident_word_threshold": config.confident_word_threshold,
                },
                "structure_gate": {
                    "enabled": config.structure_aware,
                    "hybrid_text_enabled": config.structure_aware and config.structure_hybrid_text,
                },
            },
        },
    )
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
    """Finalize one bounded Surya batch into route-specific manifest records.

    ``primary`` bypasses Tesseract, ``quality`` replaces a rejected Tesseract
    candidate, and ``structure`` may hybridize only after Tesseract's quality
    gate accepted the page. Those modes intentionally have different evidence
    contracts and must not be conflated in downstream evaluation.
    """
    if not pending:
        return
    predictions = surya_batch(
        pending[0].image_path.parent, job_dir / "surya" / f"batch-{batch_number:06d}", config
    )
    for candidate in pending:
        prediction = predictions.get(candidate.image_path.name) or predictions.get(
            candidate.image_path.stem
        )
        # A Figure / Diagram can be a valuable Surya result even when all of
        # its text blocks were intentionally skipped. A blank scanned page is
        # valid too: reject only a missing or malformed response, never an
        # empty but well-formed Surya page.
        if not isinstance(prediction, dict) or not isinstance(prediction.get("blocks"), list):
            raise RuntimeError(
                f"Surya returned a malformed result for page {candidate.page} ({candidate.image_path.name})"
            )
        surya_blocks = scale_ocr_items_to_pdf_points(prediction["blocks"], candidate.raster)
        raw_surya_text = str(prediction.get("text", "")).strip()
        if not raw_surya_text:
            raw_surya_text = "\n".join(
                str(block.get("text", "")).strip()
                for block in surya_blocks
                if isinstance(block, dict) and str(block.get("text", "")).strip()
            ).strip()
        authoritative_blocks = surya_blocks
        authoritative_text = raw_surya_text
        primary_route = candidate.escalation_reason == "primary"
        hybrid_text: dict[str, Any] | None = None
        surya_candidate: dict[str, Any] | None = None
        # Structure escalation is only reached after the inexpensive OCR pass
        # passed its text-quality gate.  It is consequently safe to let Surya
        # provide semantic layout while high-confidence Tesseract words supply
        # final text inside those regions. A quality rejection remains a pure
        # Surya result because its Tesseract candidate is not trustworthy.
        if (
            not primary_route
            and config.structure_hybrid_text
            and candidate.escalation_reason == "structure"
            and bool(candidate.quality and candidate.quality.get("accepted"))
        ):
            hybrid_blocks, details = hybridize_surya_layout_with_tesseract(
                surya_blocks,
                candidate.tesseract_words,
                confident_word_threshold=config.confident_word_threshold,
            )
            # Persist declined attempts too: a structure route that remains
            # pure Surya must be distinguishable from a page where hybridizing
            # was disabled or never considered.
            hybrid_text = details
            if details["applied"]:
                authoritative_blocks = hybrid_blocks
                authoritative_text = "\n".join(
                    str(block.get("text", "")).strip()
                    for block in hybrid_blocks
                    if str(block.get("text", "")).strip()
                ).strip()
                hybrid_text = details
                surya_candidate = {"text": raw_surya_text, "blocks": surya_blocks}
        record_raster = dict(candidate.raster)
        source_image_bbox = prediction.get("source_image_bbox")
        if source_image_bbox is not None:
            record_raster["surya_source_image_bbox"] = source_image_bbox
        if primary_route:
            # Hindi primary mode deliberately bypasses the cheap engine and
            # its confidence gate. The Surya result is the sole OCR evidence.
            record = make_record(
                candidate.page,
                candidate.inspection,
                engine="surya",
                outcome="surya_primary",
                text=authoritative_text,
                surya_batch_number=batch_number,
                blocks=authoritative_blocks,
                raster=record_raster,
                escalation_reason="primary",
                executed_route="surya",
            )
        else:
            record = make_record(
                candidate.page,
                candidate.inspection,
                engine="surya",
                outcome="surya_escalated",
                text=authoritative_text,
                quality=candidate.quality,
                surya_batch_number=batch_number,
                blocks=authoritative_blocks,
                raster=record_raster,
                escalation_reason=candidate.escalation_reason,
                structure_gate=candidate.structure_gate,
                tesseract_candidate={
                    "text": candidate.tesseract_text,
                    "quality": candidate.quality,
                    "words": candidate.tesseract_words,
                    "blocks": candidate.tesseract_blocks,
                },
                surya_candidate=surya_candidate,
                hybrid_text=hybrid_text,
            )
        append_jsonl(manifest_path, record)
        records[candidate.page] = record


def validate_pipeline_config(config: PipelineConfig) -> None:
    """Reject unsafe or ambiguous fallback combinations for library callers."""
    if config.fallback_engine not in {"surya", "none"}:
        raise ValueError("fallback_engine must be 'surya' or 'none'")
    if config.ocr_engine not in {"auto", "tesseract", "surya"}:
        raise ValueError("ocr_engine must be 'auto', 'tesseract', or 'surya'")
    # Structure escalation derives its evidence from Tesseract TSV geometry;
    # it has no meaningful direct-Surya semantics.
    if config.structure_aware and primary_ocr_engine(config) != "tesseract":
        raise ValueError("structure_aware requires primary OCR engine 'tesseract'")
    if config.structure_aware and config.fallback_engine != "surya":
        raise ValueError("structure_aware requires fallback_engine='surya'")


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
    """Process a document using bounded batches for any Surya route."""
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF: {pdf_path}")
    validate_pipeline_config(config)
    if dry_run:
        states = {state: 0 for state in PAGE_STATES}
        with fitz.open(pdf_path) as pdf:
            for index in range(pdf.page_count):
                states[inspect_page(pdf[index], config)["classification"]] += 1
            ocr_candidate_pages = states["SCANNED"] + states["MIXED"] + states["OCR_NEEDED"]
            resolved_primary_engine = primary_ocr_engine(config)
            return {
                "source": pdf_path.name,
                "source_page_count": pdf.page_count,
                "state": "dry_run",
                "classification_counts": states,
                "ocr_candidate_pages": ocr_candidate_pages,
                # This is a plan, not an inference result: native pages are
                # excluded and no Tesseract quality gate or Surya model is run.
                "primary_ocr_engine": resolved_primary_engine,
                "planned_surya_primary_pages": ocr_candidate_pages
                if resolved_primary_engine == "surya"
                else 0,
                "planned_tesseract_primary_pages": ocr_candidate_pages
                if resolved_primary_engine == "tesseract"
                else 0,
            }

    job_dir = job_output_dir(pdf_path, input_root, output_root)
    source = source_identity(pdf_path)
    check_or_create_job(job_dir, source, config, resume, allow_pipeline_upgrade=reprocess_ocr)
    manifest_path = job_dir / "pages.jsonl"
    records = load_completed_pages(manifest_path)
    with (
        fitz.open(pdf_path) as pdf,
        tempfile.TemporaryDirectory(prefix=".sttl-work-", dir=job_dir) as work,
    ):
        work_root = Path(work)
        pending: list[_PendingSuryaPage] = []
        # A resumed job must not rewrite an existing raw Surya batch.  The
        # page manifest persists each selected batch number, so resume from
        # the largest one rather than beginning again at zero.
        completed_batches = [
            value
            for record in records.values()
            if isinstance((value := record.get("surya_batch")), int) and value >= 0
        ]
        batch_number = max(completed_batches, default=0)

        def flush_pending() -> None:
            nonlocal batch_number, pending
            if pending:
                batch_number += 1
                flush_surya(
                    pending,
                    batch_number=batch_number,
                    job_dir=job_dir,
                    config=config,
                    manifest_path=manifest_path,
                    records=records,
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
                    page_number,
                    inspection,
                    engine="pymupdf",
                    outcome="native_text_accepted",
                    text=inspection["native_text"],
                )
                append_jsonl(manifest_path, record)
                records[page_number] = record
                continue
            # A unique input directory prevents a Surya fallback from
            # accidentally seeing subsequent pages: its CLI processes every
            # image in the directory.
            image_dir = work_root / f"surya-input-{batch_number + 1:06d}"
            image_path = image_dir / f"page-{page_number:06d}.png"
            raster = render_page(pdf[index], image_path, config.render_dpi)
            if primary_ocr_engine(config) == "surya":
                # Hindi/Devanagari's accuracy-first route deliberately skips
                # the cheap engine and its gate. Native pages above remain
                # trusted native text; only OCR-needed pages reach Surya.
                pending.append(
                    _PendingSuryaPage(
                        page=page_number,
                        inspection=inspection,
                        image_path=image_path,
                        quality=None,
                        raster=raster,
                        tesseract_text="",
                        tesseract_words=[],
                        tesseract_blocks=[],
                        escalation_reason="primary",
                    )
                )
                if len(pending) >= config.surya_batch_size:
                    flush_pending()
                continue
            # Tesseract is always attempted before any fallback. Its TSV
            # layout is retained if the quality gate accepts the page; an
            # opt-in structure gate can still select Surya when the plain text
            # is reliable but its table / column semantics are not.
            tesseract_text, quality, tesseract_words, tesseract_blocks = tesseract_page(
                image_path, config
            )
            structure_gate = (
                tesseract_structure_gate(tesseract_words, raster, pdf[index])
                if config.structure_aware and quality["accepted"]
                else None
            )
            tesseract_words = scale_ocr_items_to_pdf_points(tesseract_words, raster)
            tesseract_blocks = scale_ocr_items_to_pdf_points(tesseract_blocks, raster)
            if quality["accepted"] and not (structure_gate and structure_gate["escalate"]):
                record = make_record(
                    page_number,
                    inspection,
                    engine="tesseract5",
                    outcome="tesseract_accepted",
                    text=tesseract_text,
                    quality=quality,
                    words=tesseract_words,
                    blocks=tesseract_blocks,
                    raster=raster,
                    structure_gate=structure_gate,
                )
                append_jsonl(manifest_path, record)
                records[page_number] = record
                image_path.unlink(missing_ok=True)
            elif config.fallback_engine == "none":
                # Do not present known low-quality text as authoritative when
                # compact-only mode declines a heavyweight fallback. Keep the
                # full candidate in the evidence layer so it can be reviewed
                # or reprocessed later with a chosen engine.
                record = make_record(
                    page_number,
                    inspection,
                    engine="none",
                    outcome="tesseract_rejected_no_fallback",
                    text="",
                    quality=quality,
                    raster=raster,
                    escalation_reason="quality",
                    tesseract_candidate={
                        "text": tesseract_text,
                        "quality": quality,
                        "words": tesseract_words,
                        "blocks": tesseract_blocks,
                    },
                )
                append_jsonl(manifest_path, record)
                records[page_number] = record
                image_path.unlink(missing_ok=True)
            else:
                pending.append(
                    _PendingSuryaPage(
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
                    )
                )
                if len(pending) >= config.surya_batch_size:
                    flush_pending()
        flush_pending()
        summary = summarize_document(pdf_path, pdf.page_count, records, source=source)
    if summary["state"] == "complete":
        write_combined_text(job_dir / "combined.txt", records, summary["source_page_count"])
        # The normalized JSON is deliberately written only after every page is
        # present, so consumers never mistake a partial document for a complete
        # normalized result.
        summary["reading_order"] = write_normalized_json(
            job_dir, pdf_path, records, summary, config
        )
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
        if (
            candidate.is_file()
            and candidate.suffix.lower() == ".pdf"
            and not is_within(candidate, output_root)
        ):
            yield candidate


def write_batch_reports(output_root: Path) -> dict[str, Any]:
    aggregate = {
        "documents": 0,
        "complete_documents": 0,
        "incomplete_documents": 0,
        "source_pages": 0,
        "native_pages": 0,
        "tesseract_accepted_pages": 0,
        "tesseract_rejected_no_fallback_pages": 0,
        "surya_primary_pages": 0,
        "surya_escalated_pages": 0,
        "surya_quality_escalated_pages": 0,
        "surya_structure_escalated_pages": 0,
        "surya_hybrid_text_pages": 0,
    }
    csv_path = output_root / "batch_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fields = [
        "source",
        "state",
        "source_page_count",
        "native_pages",
        "tesseract_accepted_pages",
        "tesseract_rejected_no_fallback_pages",
        "surya_primary_pages",
        "surya_escalated_pages",
        "surya_quality_escalated_pages",
        "surya_structure_escalated_pages",
        "surya_hybrid_text_pages",
        "mean_tesseract_confidence",
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
                "native_pages",
                "tesseract_accepted_pages",
                "tesseract_rejected_no_fallback_pages",
                "surya_primary_pages",
                "surya_escalated_pages",
                "surya_quality_escalated_pages",
                "surya_structure_escalated_pages",
                "surya_hybrid_text_pages",
            ):
                aggregate[field] += int(summary.get(field, 0))
            aggregate[
                "complete_documents"
                if summary.get("state") == "complete"
                else "incomplete_documents"
            ] += 1
            writer.writerow(
                {
                    "source": summary.get("source", ""),
                    "state": summary.get("state", ""),
                    "source_page_count": summary.get("source_page_count", 0),
                    "native_pages": summary.get("native_pages", 0),
                    "tesseract_accepted_pages": summary.get("tesseract_accepted_pages", 0),
                    "tesseract_rejected_no_fallback_pages": summary.get(
                        "tesseract_rejected_no_fallback_pages", 0
                    ),
                    "surya_primary_pages": summary.get("surya_primary_pages", 0),
                    "surya_escalated_pages": summary.get("surya_escalated_pages", 0),
                    "surya_quality_escalated_pages": summary.get(
                        "surya_quality_escalated_pages", 0
                    ),
                    "surya_structure_escalated_pages": summary.get(
                        "surya_structure_escalated_pages", 0
                    ),
                    "surya_hybrid_text_pages": summary.get("surya_hybrid_text_pages", 0),
                    "mean_tesseract_confidence": summary.get("mean_tesseract_confidence", ""),
                }
            )
    os.replace(temporary, csv_path)
    atomic_write_json(output_root / "batch_summary.json", aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="PDF or directory tree of PDFs")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ocr"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Inspect pages only; neither render nor write output"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Refuse an existing matching job directory"
    )
    parser.add_argument(
        "--reprocess-ocr",
        action="store_true",
        help="Repair existing Tesseract/Surya records while retaining native-text pages",
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--language",
        default="eng",
        help="Tesseract language(s), for example hin or script/Devanagari for Hindi plus English",
    )
    parser.add_argument(
        "--tessdata-dir",
        help="Directory containing local Tesseract .traineddata files, such as models/tessdata",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--tesseract-psm", type=int, default=3)
    parser.add_argument("--tesseract-timeout", type=int, default=180, metavar="SECONDS")
    parser.add_argument("--surya-timeout", type=int, default=1_800, metavar="SECONDS")
    parser.add_argument("--surya-batch-size", type=int, default=4)
    parser.add_argument(
        "--ocr-engine",
        choices=("auto", "tesseract", "surya"),
        default="auto",
        help=(
            "Primary engine for OCR-needed pages; auto prefers Surya for Hindi/Devanagari "
            "when the Surya route is enabled, otherwise Tesseract"
        ),
    )
    parser.add_argument(
        "--fallback-engine",
        choices=("surya", "none"),
        default="surya",
        help="Fallback after a rejected Tesseract page; use none with --ocr-engine tesseract for compact CPU-only OCR",
    )
    parser.add_argument(
        "--surya-keep-server",
        action="store_true",
        help="Reuse Surya's host-wide inference server across batches; you must manage its shutdown",
    )
    parser.add_argument(
        "--structure-aware",
        action="store_true",
        help="Escalate Tesseract-accepted pages with table-like alignment or multiple text columns to Surya",
    )
    parser.add_argument(
        "--no-structure-hybrid-text",
        action="store_false",
        dest="structure_hybrid_text",
        default=True,
        help="Keep Surya text on structure escalations instead of replacing region text with accepted Tesseract words",
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
        language=args.language,
        tessdata_dir=args.tessdata_dir,
        render_dpi=args.dpi,
        tesseract_psm=args.tesseract_psm,
        tesseract_timeout_seconds=args.tesseract_timeout,
        surya_timeout_seconds=args.surya_timeout,
        surya_batch_size=args.surya_batch_size,
        surya_keep_server=args.surya_keep_server,
        fallback_engine=args.fallback_engine,
        ocr_engine=args.ocr_engine,
        structure_aware=args.structure_aware,
        structure_hybrid_text=args.structure_hybrid_text,
        min_native_chars=args.min_native_chars,
        min_native_words=args.min_native_words,
        max_native_garbage_ratio=args.max_native_garbage_ratio,
        dominant_image_ratio=args.dominant_image_ratio,
        min_tesseract_chars=args.min_tesseract_chars,
        min_tesseract_words=args.min_tesseract_words,
        min_mean_confidence=args.min_mean_confidence,
        min_confident_word_ratio=args.min_confident_word_ratio,
        confident_word_threshold=args.confident_word_threshold,
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
    if args.tessdata_dir is not None and not Path(args.tessdata_dir).is_dir():
        raise SystemExit(
            f"--tessdata-dir does not exist or is not a directory: {args.tessdata_dir}"
        )
    if args.fallback_engine == "none" and args.structure_aware:
        raise SystemExit("--structure-aware requires --fallback-engine surya")
    primary_is_surya = args.ocr_engine == "surya" or (
        args.ocr_engine == "auto"
        and args.fallback_engine == "surya"
        and language_requests_hindi(args.language)
    )
    if args.structure_aware and primary_is_surya:
        raise SystemExit(
            "--structure-aware requires a Tesseract primary route; add --ocr-engine tesseract"
        )
    for name in (
        "max_native_garbage_ratio",
        "dominant_image_ratio",
        "min_confident_word_ratio",
        "max_tesseract_garbage_ratio",
        "min_plausible_word_ratio",
    ):
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
                pdf_path,
                input_root=input_root,
                output_root=output_root,
                config=config,
                resume=not args.no_resume,
                dry_run=args.dry_run,
                reprocess_ocr=args.reprocess_ocr,
            )
            if args.dry_run:
                dry_runs.append(summary)
            else:
                print(
                    f"    pages={summary['source_page_count']} native={summary['native_pages']} "
                    f"tesseract={summary['tesseract_accepted_pages']} "
                    f"rejected_no_fallback={summary['tesseract_rejected_no_fallback_pages']} "
                    f"surya_primary={summary['surya_primary_pages']} "
                    f"surya={summary['surya_escalated_pages']} "
                    f"state={summary['state']}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:  # a corpus worker continues after one bad PDF
            failures += 1
            print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    elapsed = round(time.perf_counter() - started, 3)
    if args.dry_run:
        print(
            json.dumps(
                {"documents": dry_runs, "failures": failures, "runtime_seconds": elapsed}, indent=2
            )
        )
    else:
        print(
            json.dumps(
                {
                    **write_batch_reports(output_root),
                    "selected_documents": selected,
                    "failures": failures,
                    "runtime_seconds": elapsed,
                },
                indent=2,
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
