#!/usr/bin/env python3
"""
Evaluate OCR output against STTL silver references.

Important:
- CER/WER use reference.json, but only for text that belongs to
  trusted structural regions when structure.json is available.
- Reading order is evaluated at block level, not word-level bbox guesses.
- Structure classification is reported as per-type precision/recall/F1 from
  one-to-one, same-type block matches. A matching label count alone is not a
  true positive: blocks also need text-token overlap or spatial overlap.
- Performance comes from the existing benchmark summary JSON.

Supported Surya input:
  {"document-name": [{"page": 1, "blocks": [...]}]}

Surya block text is extracted from block["html"].

Supported cascade input:
  <stem>_cascade.json (normalized pages with ``blocks``), or
  <stem>_rich.json (pages with ``authoritative.blocks``).

The rich adapter deliberately evaluates only the page's authoritative output;
the non-winning PyMuPDF, Tesseract, and Surya evidence layers are not
concatenated into the OCR hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import re
from collections import Counter, deque
from pathlib import Path


WORD_RE = re.compile(r"\S+", re.UNICODE)

# A structural true positive needs meaningful evidence beyond a matching
# semantic label. Text is generally the strongest signal; bbox IoU is a useful
# alternative when OCR text is damaged but the detected region is correctly
# located. Keep these explicit in the output so evaluations are reproducible.
TEXT_TOKEN_JACCARD_THRESHOLD = 0.25
BBOX_IOU_THRESHOLD = 0.25
NORMALIZED_GEOMETRY_SPACE = "pdf_points"

# An inversion rate based on one or two incidental matches says little about
# whole-page reading order. Retain it for diagnostics, but only call it
# measured when both sides have enough coverage.
READING_ORDER_MIN_MATCHES = 2
READING_ORDER_MIN_COVERAGE = 0.50

REFERENCE_TEXT_TYPES = {
    "page_header",
    "page_footer",
    "section_heading",
    "heading",
    "paragraph",
    "table",
    "table_caption",
    "figure_caption",
}

# These blocks describe or delimit a visual region, rather than supplying the
# text that belongs in the conservative silver reference.  Their labels remain
# available in the source artifacts, but their generated descriptions must not
# be treated as OCR insertions in CER/WER.
VISUAL_DESCRIPTION_TYPES = {
    "chart",
    "figure",
    "image",
    "diagram",
    "picture",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    text = text or ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_token(token):
    token = token.lower()
    token = token.replace("\u00ad", "")
    token = re.sub(r"[^\w]+", "", token, flags=re.UNICODE)
    return token


def tokens(text):
    out = []
    for raw in WORD_RE.findall(normalize_text(text)):
        t = normalize_token(raw)
        if t:
            out.append(t)
    return out


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current

    return previous[-1]


def cer(reference, hypothesis):
    distance, reference_length, hypothesis_length = character_error_stats(
        reference,
        hypothesis,
    )
    return error_rate(distance, reference_length, hypothesis_length)


def wer(reference, hypothesis):
    distance, reference_length, hypothesis_length = word_error_stats(
        reference,
        hypothesis,
    )
    return error_rate(distance, reference_length, hypothesis_length)


def error_rate(distance, reference_length, hypothesis_length):
    """Return a page-level error rate with explicit empty-reference behavior."""
    if not reference_length:
        return 0.0 if not hypothesis_length else 1.0
    return distance / reference_length


def character_error_stats(reference, hypothesis):
    """Return exact character edit distance and normalized input lengths."""
    r = normalize_text(reference)
    h = normalize_text(hypothesis)
    return levenshtein(list(r), list(h)), len(r), len(h)


def word_error_stats(reference, hypothesis):
    """Return exact token edit distance and normalized token counts."""
    r = tokens(reference)
    h = tokens(hypothesis)
    return levenshtein(r, h), len(r), len(h)


def html_to_text(value):
    if not isinstance(value, str):
        return ""

    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    text = re.sub(r"</(?:p|div|tr|table|h[1-6])>", "\n", text, flags=re.I)
    text = re.sub(r"</td>|</th>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    return normalize_text(text)


def bbox_of(block):
    if not isinstance(block, dict):
        return None

    value = block.get("bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [float(x) for x in value]
        except Exception:
            pass

    return None


def geometry_for_block(page, source_page, block):
    """Retain only an explicit, page-normalized geometry frame for matching.

    Raw Surya/Chandra payloads may use model or raster coordinates. Numerical
    bboxes are not comparable merely because they have the same four values;
    IoU is enabled later only when both sides explicitly identify the same
    normalized PDF-point frame and page envelope.
    """
    if not isinstance(block, dict):
        return None
    coordinate_space = (
        block.get("coordinate_space")
        or (source_page.get("coordinate_space") if isinstance(source_page, dict) else None)
        or (page.get("coordinate_space") if isinstance(page, dict) else None)
    )
    coordinate_frame = (
        block.get("coordinate_frame")
        or (source_page.get("coordinate_frame") if isinstance(source_page, dict) else None)
        or (page.get("coordinate_frame") if isinstance(page, dict) else None)
    )
    page_bbox = bbox_of(page)
    return {
        "coordinate_space": coordinate_space,
        "coordinate_frame": coordinate_frame,
        "page_bbox": page_bbox,
    }


def surya_block_text(block):
    if not isinstance(block, dict):
        return ""

    if isinstance(block.get("html"), str):
        return html_to_text(block["html"])

    for key in ("text", "content", "raw_text"):
        if isinstance(block.get(key), str):
            return normalize_text(block[key])

    return ""


def load_reference(path):
    data = load_json(path)
    pages = data.get("pages", [])
    return sorted(pages, key=lambda p: p.get("page", 0))


def load_structure(path):
    data = load_json(path)
    pages = data.get("pages", [])
    return sorted(pages, key=lambda p: p.get("page", 0))


def load_surya(path):
    data = load_json(path)

    if isinstance(data, dict):
        if isinstance(data.get("pages"), list):
            return data["pages"]

        for value in data.values():
            if isinstance(value, list):
                return value

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported OCR JSON format: {path}")


def ordered_pages(pages):
    """Sort numbered pages without losing input order for unnumbered entries."""
    prepared = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        try:
            prepared.append((0, float(page.get("page")), index, page))
        except (TypeError, ValueError):
            prepared.append((1, 0.0, index, page))
    prepared.sort(key=lambda item: item[:3])
    return [item[3] for item in prepared]


def pages_by_number(pages, source_name):
    """Index pages by their declared number, rejecting ambiguous comparisons.

    Legacy OCR arrays sometimes omit ``page``; in that narrow case their
    ordered position is the only available identifier.  Once any artifact
    declares numbered pages, duplicated, non-positive, or mismatched numbers
    must fail rather than silently scoring one page against another.
    """
    indexed = {}
    for position, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ValueError(f"{source_name} page {position} is not an object")
        raw_number = page.get("page", position)
        try:
            number = int(raw_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_name} page {position} has an invalid page number: {raw_number!r}") from exc
        if number <= 0:
            raise ValueError(f"{source_name} page {position} has a non-positive page number: {number}")
        if number in indexed:
            raise ValueError(f"{source_name} contains duplicate page number {number}")
        indexed[number] = page
    return indexed


def aligned_pages(reference_pages, structure_pages, ocr_pages):
    """Return exact page-ID joins; do not let mismatched artifacts mis-score."""
    reference = pages_by_number(reference_pages, "reference")
    structure = pages_by_number(structure_pages, "structure")
    ocr = pages_by_number(ocr_pages, "OCR")
    expected = set(reference)
    for name, pages in (("structure", structure), ("OCR", ocr)):
        actual = set(pages)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ValueError(f"{name} page IDs do not match reference: {', '.join(details)}")
    return [(number, reference[number], structure[number], ocr[number]) for number in sorted(expected)]


def load_cascade(path):
    """Load either normalized or rich output written by ``pdf_pipeline.py``.

    Both artifacts are document objects with a ``pages`` list.  The rich form
    keeps its final page content under ``authoritative``; selection happens in
    ``cascade_page_ocr_blocks`` so page-level metadata remains available to the
    caller.
    """
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ValueError(
            "Unsupported cascade JSON format: expected an object with a pages list: "
            f"{path}"
        )
    pages = data["pages"]
    if data.get("schema_version") == "cascade-ocr/rich-v1":
        malformed = [
            str(page.get("page", index + 1))
            for index, page in enumerate(pages)
            if not isinstance(page, dict) or not isinstance(page.get("authoritative"), dict)
        ]
        if malformed:
            raise ValueError(
                "Invalid rich cascade JSON: page(s) missing authoritative output: "
                + ", ".join(malformed)
            )
    return ordered_pages(pages)


def trusted_reference_text(structure_page):
    """
    Return text only from trusted semantic structures. This avoids scoring
    diagram-internal garbage extracted by pdftotext as if it were ground truth.
    """
    parts = []

    for block in structure_page.get("blocks", []):
        if block.get("type") in REFERENCE_TEXT_TYPES:
            text = normalize_text(block.get("text", ""))
            if text:
                parts.append(text)

    return "\n".join(parts)


def generic_page_text(page):
    if isinstance(page, dict):
        for key in ("text", "markdown", "content", "raw"):
            if isinstance(page.get(key), str):
                return normalize_text(page[key])

    return ""


def semantic_block_label(block):
    """Return the richest available semantic label for an OCR block."""
    if not isinstance(block, dict):
        return None
    return (
        block.get("block_type")
        or block.get("label")
        or block.get("raw_label")
        or block.get("type")
    )


def normalized_label_key(label):
    if not label:
        return ""
    return re.sub(r"[\W_]+", "", str(label).strip().casefold())


def is_visual_description_block(block):
    """Whether a semantic visual description must be excluded from text OCR."""
    return normalized_label_key(semantic_block_label(block)) in VISUAL_DESCRIPTION_TYPES


def ordered_blocks(raw_blocks):
    """Sort supplied blocks by explicit reading order while keeping ties stable."""
    if not isinstance(raw_blocks, list):
        return []

    prepared = []
    for index, block in enumerate(raw_blocks):
        if not isinstance(block, dict):
            continue
        value = block.get("reading_order")
        try:
            prepared.append((0, float(value), index, block))
        except (TypeError, ValueError):
            prepared.append((1, 0.0, index, block))

    prepared.sort(key=lambda item: item[:3])
    return [item[3] for item in prepared]


def cascade_block_text(block):
    """Read the canonical text first, with HTML retained as a safe fallback."""
    if not isinstance(block, dict):
        return ""

    for key in ("text", "content", "raw_text"):
        if isinstance(block.get(key), str):
            text = normalize_text(block[key])
            if text:
                return text

    if isinstance(block.get("html"), str):
        return html_to_text(block["html"])

    return ""


def cascade_page_ocr_blocks(page):
    """Adapt normalized and rich cascade pages without mixing evidence layers."""
    if not isinstance(page, dict):
        return []

    # Rich output has a top-level page envelope and a final, explicitly chosen
    # result.  Never fall through to ``layers``: those values are provenance,
    # not additional OCR text.
    authoritative = page.get("authoritative")
    source_page = authoritative if isinstance(authoritative, dict) else page

    raw_blocks = source_page.get("blocks", [])
    result = []
    for block in ordered_blocks(raw_blocks):
        if block.get("skipped") or block.get("error"):
            continue
        if is_visual_description_block(block):
            continue

        text = cascade_block_text(block)
        label = semantic_block_label(block)
        bbox = bbox_of(block)
        if not text and not (label and bbox):
            continue

        result.append(
            {
                "text": text,
                # ``block_type`` is the semantic label preserved by the
                # normalized schema.  ``type`` is often merely "text".
                "label": label,
                "bbox": bbox,
                "reading_order": block.get("reading_order"),
                "geometry": geometry_for_block(page, source_page, block),
            }
        )

    if result:
        return result

    # A populated block array is the authoritative segmentation.  Falling back
    # to page text after its entries were skipped would reintroduce visual
    # descriptions (or failed OCR) that this adapter intentionally excluded.
    if isinstance(raw_blocks, list) and raw_blocks:
        return []

    text = generic_page_text(source_page)
    return (
        [{"text": text, "label": None, "bbox": None, "reading_order": 1, "geometry": None}]
        if text
        else []
    )


def visual_description_labels(page, engine):
    """Report visual-region markers without feeding their descriptions to OCR metrics."""
    if not isinstance(page, dict):
        return []
    authoritative = page.get("authoritative") if engine == "cascade" else None
    source_page = authoritative if isinstance(authoritative, dict) else page
    return [
        normalized_label_key(semantic_block_label(block))
        for block in ordered_blocks(source_page.get("blocks", []))
        if is_visual_description_block(block)
    ]


def page_ocr_blocks(page, engine):
    if engine == "cascade":
        return cascade_page_ocr_blocks(page)

    if engine == "surya":
        raw = page.get("blocks", [])
        result = []

        for block in ordered_blocks(raw):
            if block.get("skipped") or block.get("error"):
                continue
            if is_visual_description_block(block):
                continue

            text = surya_block_text(block)
            label = semantic_block_label(block)
            bbox = bbox_of(block)
            if not text and not (label and bbox):
                continue

            result.append(
                {
                    "text": text,
                    "label": label,
                    "bbox": bbox,
                    "reading_order": block.get("reading_order"),
                    "geometry": geometry_for_block(page, page, block),
                }
            )

        return result

    # Generic adapter for normalized Chandra output.
    raw = page.get("blocks", [])
    if isinstance(raw, list):
        result = []
        for block in ordered_blocks(raw):
            if block.get("skipped") or block.get("error"):
                continue
            if is_visual_description_block(block):
                continue
            text = normalize_text(
                block.get("text")
                or block.get("content")
                or block.get("raw_text")
                or ""
            )
            label = semantic_block_label(block)
            bbox = bbox_of(block)
            if text or (label and bbox):
                result.append(
                    {
                        "text": text,
                        "label": label,
                        "bbox": bbox,
                        "reading_order": block.get("reading_order"),
                        "geometry": geometry_for_block(page, page, block),
                    }
                )
        if result:
            return result
        if raw:
            return []

    text = generic_page_text(page)
    return (
        [{"text": text, "label": None, "bbox": None, "reading_order": 1, "geometry": None}]
        if text
        else []
    )


def canonical_label(label):
    if not label:
        return None

    key = normalized_label_key(label)

    mapping = {
        "pageheader": "page_header",
        "pagefooter": "page_footer",
        "sectionheader": "section_heading",
        "sectionheading": "section_heading",
        "heading": "heading",
        "title": "heading",
        "text": "paragraph",
        "paragraph": "paragraph",
        "body": "paragraph",
        "bodytext": "paragraph",
        "caption": "figure_caption",
        "figurecaption": "figure_caption",
        "imagecaption": "figure_caption",
        "table": "table",
        "tablecaption": "table_caption",
        "listgroup": "list",
        "listitem": "list",
        "list": "list",
    }

    return mapping.get(key, key)


def reference_blocks_for_page(structure_page):
    blocks = []

    # Structure JSON is normally emitted in reading order, but a consumer can
    # reorder its array while preserving the explicit reading-order field.
    # The evaluator should honor the declared order rather than JSON order.
    for block in ordered_blocks(structure_page.get("blocks", [])):
        if block.get("type") in REFERENCE_TEXT_TYPES:
            blocks.append(
                {
                    "type": block.get("type"),
                    "text": normalize_text(block.get("text", "")),
                    "bbox": bbox_of(block),
                    "reading_order": block.get("reading_order"),
                    "geometry": geometry_for_block(structure_page, structure_page, block),
                }
            )

    return blocks


def token_jaccard(left, right):
    """Return multiset token Jaccard, or zero when either side has no text."""
    left_tokens = Counter(tokens(left))
    right_tokens = Counter(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = sum((left_tokens & right_tokens).values())
    union = sum((left_tokens | right_tokens).values())
    return intersection / union if union else 0.0


def bbox_iou(left, right):
    """Return IoU for two positive-area boxes, otherwise ``None``."""
    if not (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
        and len(left) == 4
        and len(right) == 4
    ):
        return None

    try:
        lx0, ly0, lx1, ly1 = (float(value) for value in left)
        rx0, ry0, rx1, ry1 = (float(value) for value in right)
    except (TypeError, ValueError):
        return None

    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    if not left_area or not right_area:
        return None

    intersection_width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    intersection_height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = intersection_width * intersection_height
    union = left_area + right_area - intersection
    return intersection / union if union else None


def canonical_block_type(block):
    """Read a reference ``type`` or OCR ``label`` through one label mapping."""
    if not isinstance(block, dict):
        return None
    return canonical_label(block.get("label") or block.get("type"))


def declared_reading_order(block, fallback_index):
    """Use an explicit block order when supplied, otherwise preserve input order."""
    if isinstance(block, dict):
        try:
            return float(block.get("reading_order"))
        except (TypeError, ValueError):
            pass
    return float(fallback_index)


def _same_page_bbox(left, right):
    if not (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
        and len(left) == 4
        and len(right) == 4
    ):
        return False
    try:
        return all(abs(float(a) - float(b)) <= 0.01 for a, b in zip(left, right))
    except (TypeError, ValueError):
        return False


def geometry_compatibility(left, right):
    """Explain whether two block bboxes share a safe normalized frame."""
    left_geometry = left.get("geometry") if isinstance(left, dict) else None
    right_geometry = right.get("geometry") if isinstance(right, dict) else None
    if not isinstance(left_geometry, dict) or not isinstance(right_geometry, dict):
        return False, "missing_geometry_metadata"
    if (
        left_geometry.get("coordinate_space") != NORMALIZED_GEOMETRY_SPACE
        or right_geometry.get("coordinate_space") != NORMALIZED_GEOMETRY_SPACE
    ):
        return False, "non_normalized_coordinate_space"
    left_frame = left_geometry.get("coordinate_frame")
    right_frame = right_geometry.get("coordinate_frame")
    if not left_frame or not right_frame:
        return False, "missing_coordinate_frame"
    if left_frame != right_frame:
        return False, "different_coordinate_frames"
    if not _same_page_bbox(left_geometry.get("page_bbox"), right_geometry.get("page_bbox")):
        return False, "different_or_missing_page_bbox"
    if bbox_of(left) is None or bbox_of(right) is None:
        return False, "missing_block_bbox"
    return True, "compatible_normalized_pdf_points"


def maximum_cardinality_weight_matching(candidates):
    """Return a maximum-cardinality matching, then maximize total evidence.

    A greedy edge choice can consume the only partner for a second block. A
    small min-cost flow graph avoids that undercount while keeping the matching
    criterion itself simple and inspectable. Every valid edge has capacity one;
    augmenting until no path remains guarantees maximum cardinality, and the
    negative edge cost selects the highest total evidence among those matchings.
    """
    if not candidates:
        return []
    reference_indices = sorted({match["reference_index"] for match in candidates})
    ocr_indices = sorted({match["ocr_index"] for match in candidates})
    reference_nodes = {index: offset + 1 for offset, index in enumerate(reference_indices)}
    ocr_offset = 1 + len(reference_indices)
    ocr_nodes = {index: ocr_offset + offset for offset, index in enumerate(ocr_indices)}
    source = 0
    sink = ocr_offset + len(ocr_indices)
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity, cost, candidate=None):
        forward = {
            "to": end,
            "reverse": len(graph[end]),
            "capacity": capacity,
            "cost": cost,
            "candidate": candidate,
        }
        backward = {
            "to": start,
            "reverse": len(graph[start]),
            "capacity": 0,
            "cost": -cost,
            "candidate": None,
        }
        graph[start].append(forward)
        graph[end].append(backward)
        return len(graph[start]) - 1

    for reference_index in reference_indices:
        add_edge(source, reference_nodes[reference_index], 1, 0.0)
    for ocr_index in ocr_indices:
        add_edge(ocr_nodes[ocr_index], sink, 1, 0.0)

    candidate_edges = []
    for candidate in sorted(candidates, key=lambda match: (match["reference_index"], match["ocr_index"])):
        # Favor text evidence deterministically when two maximum-cardinality
        # solutions have numerically equal primary match scores.
        weight = (
            candidate["match_score"]
            + candidate["text_token_jaccard"] * 1e-6
            + (candidate["bbox_iou"] or 0.0) * 1e-9
        )
        start = reference_nodes[candidate["reference_index"]]
        edge_index = add_edge(start, ocr_nodes[candidate["ocr_index"]], 1, -weight, candidate)
        candidate_edges.append((start, edge_index, candidate))

    while True:
        distances = [float("inf")] * len(graph)
        previous = [None] * len(graph)
        distances[source] = 0.0
        queue = deque([source])
        queued = [False] * len(graph)
        queued[source] = True
        while queue:
            node = queue.popleft()
            queued[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if not edge["capacity"]:
                    continue
                candidate_distance = distances[node] + edge["cost"]
                if candidate_distance + 1e-12 < distances[edge["to"]]:
                    distances[edge["to"]] = candidate_distance
                    previous[edge["to"]] = (node, edge_index)
                    if not queued[edge["to"]]:
                        queue.append(edge["to"])
                        queued[edge["to"]] = True
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            previous_node, edge_index = previous[node]
            edge = graph[previous_node][edge_index]
            edge["capacity"] -= 1
            graph[node][edge["reverse"]]["capacity"] += 1
            node = previous_node

    return [
        candidate
        for start, edge_index, candidate in candidate_edges
        if graph[start][edge_index]["capacity"] == 0
    ]


def one_to_one_block_matches(reference_blocks, ocr_blocks, *, require_same_type):
    """Select evidence-backed, one-to-one reference/OCR block matches.

    A candidate needs token Jaccard or bbox IoU at the documented threshold.
    Structure scoring additionally requires equivalent canonical semantic types;
    reading-order scoring does not, because segmentation engines commonly use
    different labels for otherwise corresponding textual regions.
    """
    candidates = []

    for reference_index, reference in enumerate(reference_blocks):
        if not isinstance(reference, dict):
            continue
        reference_type = canonical_block_type(reference)

        for ocr_index, ocr in enumerate(ocr_blocks):
            if not isinstance(ocr, dict):
                continue
            ocr_type = canonical_block_type(ocr)
            if require_same_type and (
                not reference_type or reference_type != ocr_type
            ):
                continue

            text_overlap = token_jaccard(
                reference.get("text", ""),
                ocr.get("text", ""),
            )
            geometry_compatible, geometry_status = geometry_compatibility(reference, ocr)
            spatial_overlap = (
                bbox_iou(bbox_of(reference), bbox_of(ocr))
                if geometry_compatible
                else None
            )
            text_match = text_overlap >= TEXT_TOKEN_JACCARD_THRESHOLD
            bbox_match = (
                spatial_overlap is not None
                and spatial_overlap >= BBOX_IOU_THRESHOLD
            )
            if not text_match and not bbox_match:
                continue

            match_basis = []
            if text_match:
                match_basis.append("text_token_jaccard")
            if bbox_match:
                match_basis.append("bbox_iou")

            candidates.append(
                {
                    "reference_index": reference_index,
                    "ocr_index": ocr_index,
                    "reference_type": reference_type,
                    "ocr_type": ocr_type,
                    "reference_reading_order": declared_reading_order(
                        reference,
                        reference_index,
                    ),
                    "ocr_reading_order": declared_reading_order(
                        ocr,
                        ocr_index,
                    ),
                    "text_token_jaccard": text_overlap,
                    "bbox_iou": spatial_overlap,
                    "geometry_evidence_status": geometry_status,
                    # Max evidence is intentionally simple and auditable. The
                    # sort below then breaks ties in favor of text evidence.
                    "match_score": max(text_overlap, spatial_overlap or 0.0),
                    "match_basis": match_basis,
                }
            )

    return maximum_cardinality_weight_matching(candidates)


def public_match(match):
    """Keep match evidence serializable without copying potentially sensitive text."""
    return {
        "reference_block_index": match["reference_index"],
        "ocr_block_index": match["ocr_index"],
        "reference_reading_order": match["reference_reading_order"],
        "ocr_reading_order": match["ocr_reading_order"],
        "reference_type": match["reference_type"],
        "ocr_type": match["ocr_type"],
        # Preserve the original public field for consumers that grouped
        # structure matches by their shared semantic type.
        "type": match["reference_type"],
        "text_token_jaccard": match["text_token_jaccard"],
        "bbox_iou": match["bbox_iou"],
        "geometry_evidence_status": match["geometry_evidence_status"],
        "match_score": match["match_score"],
        "match_basis": match["match_basis"],
    }


def order_similarity_details(reference_blocks, ocr_blocks):
    """Return coverage-aware reading-order evidence and its inversion score.

    ``observed_inversion_rate`` remains available for diagnosis, but the
    primary ``inversion_rate`` is ``None`` when too few blocks matched or when
    either side's block coverage is below ``READING_ORDER_MIN_COVERAGE``.
    """
    matches = one_to_one_block_matches(
        reference_blocks,
        ocr_blocks,
        require_same_type=False,
    )
    by_ocr_order = sorted(
        matches,
        key=lambda match: (match["ocr_reading_order"], match["ocr_index"]),
    )
    mapped_reference_orders = [
        match["reference_reading_order"]
        for match in by_ocr_order
    ]

    inversions = 0
    pairs = 0
    for left_index, mapped_reference in enumerate(mapped_reference_orders):
        for later_reference in mapped_reference_orders[left_index + 1 :]:
            pairs += 1
            if mapped_reference > later_reference:
                inversions += 1

    observed_inversion_rate = (
        inversions / pairs
        if pairs
        else None
    )
    matched_blocks = len(matches)
    reference_coverage = (
        matched_blocks / len(reference_blocks)
        if reference_blocks
        else None
    )
    ocr_coverage = (
        matched_blocks / len(ocr_blocks)
        if ocr_blocks
        else None
    )

    inconclusive_reasons = []
    if matched_blocks < READING_ORDER_MIN_MATCHES:
        inconclusive_reasons.append("fewer_than_two_matched_blocks")
    if reference_coverage is None:
        inconclusive_reasons.append("no_reference_blocks")
    elif reference_coverage < READING_ORDER_MIN_COVERAGE:
        inconclusive_reasons.append("low_reference_match_coverage")
    if ocr_coverage is None:
        inconclusive_reasons.append("no_ocr_blocks")
    elif ocr_coverage < READING_ORDER_MIN_COVERAGE:
        inconclusive_reasons.append("low_ocr_match_coverage")

    status = "inconclusive" if inconclusive_reasons else "measured"
    return {
        "status": status,
        "inconclusive_reasons": inconclusive_reasons,
        "inversion_rate": (
            observed_inversion_rate
            if status == "measured"
            else None
        ),
        "observed_inversion_rate": observed_inversion_rate,
        "matched_blocks": matched_blocks,
        "inversions": inversions,
        "reference_blocks": len(reference_blocks),
        "ocr_blocks": len(ocr_blocks),
        "reference_match_coverage": reference_coverage,
        "ocr_match_coverage": ocr_coverage,
        "minimum_matches": READING_ORDER_MIN_MATCHES,
        "minimum_coverage": READING_ORDER_MIN_COVERAGE,
        "matches": [public_match(match) for match in by_ocr_order],
    }


def order_similarity(reference_blocks, ocr_blocks):
    """Backward-compatible tuple form of :func:`order_similarity_details`.

    The first value is the observed inversion rate, matching the historical
    behavior: it is available whenever at least two blocks match even if the
    coverage-aware report marks the result inconclusive.
    """
    details = order_similarity_details(reference_blocks, ocr_blocks)
    return (
        details["observed_inversion_rate"],
        details["matched_blocks"],
        details["inversions"],
    )


def structure_scores(reference_blocks, ocr_blocks):
    """Score semantic structure from evidence-backed one-to-one matches."""
    ref_types = Counter(
        canonical_block_type(block)
        for block in reference_blocks
        if canonical_block_type(block)
    )
    ocr_types = Counter(
        canonical_block_type(block)
        for block in ocr_blocks
        if canonical_block_type(block)
    )
    matches = one_to_one_block_matches(
        reference_blocks,
        ocr_blocks,
        require_same_type=True,
    )
    matches_by_type = {}
    for match in matches:
        matches_by_type.setdefault(match["reference_type"], []).append(match)

    types = sorted(set(ref_types) | set(ocr_types))
    per_type = {}

    for block_type in types:
        type_matches = matches_by_type.get(block_type, [])
        matched = len(type_matches)
        precision = matched / ocr_types[block_type] if ocr_types[block_type] else 0.0
        recall = matched / ref_types[block_type] if ref_types[block_type] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        text_scores = [match["text_token_jaccard"] for match in type_matches]
        bbox_scores = [
            match["bbox_iou"]
            for match in type_matches
            if match["bbox_iou"] is not None
        ]

        per_type[block_type] = {
            "reference": ref_types[block_type],
            "ocr": ocr_types[block_type],
            "matched": matched,
            "unmatched_reference": ref_types[block_type] - matched,
            "unmatched_ocr": ocr_types[block_type] - matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_text_token_jaccard": (
                sum(text_scores) / len(text_scores)
                if text_scores
                else None
            ),
            "mean_bbox_iou": (
                sum(bbox_scores) / len(bbox_scores)
                if bbox_scores
                else None
            ),
        }

    relevant_types = [
        result for result in per_type.values()
        if result["reference"] > 0 or result["ocr"] > 0
    ]
    macro_f1 = (
        sum(result["f1"] for result in relevant_types) / len(relevant_types)
        if relevant_types
        else None
    )

    return {
        "per_type": per_type,
        "macro_f1": macro_f1,
        "matching": {
            "method": "maximum_cardinality_maximum_evidence_one_to_one_same_type_token_jaccard_or_bbox_iou",
            "criteria": {
                "canonical_semantic_type": "must_match",
                "text_token_jaccard_threshold": TEXT_TOKEN_JACCARD_THRESHOLD,
                "bbox_iou_threshold": BBOX_IOU_THRESHOLD,
                "bbox_iou_coordinate_requirement": (
                    "matching explicit normalized pdf_points coordinate frame "
                    "and page envelope"
                ),
            },
            "matched_blocks": len(matches),
            "reference_blocks": sum(ref_types.values()),
            "ocr_blocks": sum(ocr_types.values()),
            "reference_match_coverage": (
                len(matches) / sum(ref_types.values())
                if ref_types
                else None
            ),
            "ocr_match_coverage": (
                len(matches) / sum(ocr_types.values())
                if ocr_types
                else None
            ),
            "unlabeled_ocr_blocks": sum(
                1 for block in ocr_blocks if not canonical_block_type(block)
            ),
            "matches": [
                public_match(match)
                for match in sorted(matches, key=lambda match: match["reference_index"])
            ],
        },
    }


def read_performance(path):
    if not path or not Path(path).exists():
        return None
    return load_json(Path(path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument(
        "--engine",
        required=True,
        choices=["surya", "chandra", "cascade"],
        help=(
            "OCR artifact adapter; cascade accepts either <stem>_cascade.json "
            "or <stem>_rich.json"
        ),
    )
    parser.add_argument("--ocr-json", required=True)
    parser.add_argument("--benchmark-summary")
    parser.add_argument("--out-dir", default="artifacts/evaluation")

    args = parser.parse_args()

    reference_pages = load_reference(
        Path(args.reference)
    )

    structure_pages = load_structure(
        Path(args.structure)
    )

    ocr_pages = (
        load_cascade(Path(args.ocr_json))
        if args.engine == "cascade"
        else load_surya(Path(args.ocr_json))
    )

    aligned = aligned_pages(reference_pages, structure_pages, ocr_pages)
    n = len(aligned)

    rows = []

    aggregate_reference_chars = 0
    aggregate_ocr_chars = 0
    aggregate_reference_words = 0
    aggregate_ocr_words = 0
    aggregate_character_edit_distance = 0
    aggregate_word_edit_distance = 0

    order_values = []
    order_results = []
    structure_results = []
    visual_description_counts = Counter()

    for page_number, _reference_page, struct_page, ocr_page in aligned:

        ref_text = trusted_reference_text(struct_page)
        ocr_blocks = page_ocr_blocks(
            ocr_page,
            args.engine,
        )
        page_visual_labels = visual_description_labels(
            ocr_page,
            args.engine,
        )
        visual_description_counts.update(page_visual_labels)
        ocr_text = "\n".join(
            block["text"]
            for block in ocr_blocks
        )

        character_distance, reference_chars, ocr_chars = character_error_stats(
            ref_text,
            ocr_text,
        )
        word_distance, reference_words, ocr_words = word_error_stats(
            ref_text,
            ocr_text,
        )
        page_cer = error_rate(
            character_distance,
            reference_chars,
            ocr_chars,
        )
        page_wer = error_rate(
            word_distance,
            reference_words,
            ocr_words,
        )

        ref_blocks = reference_blocks_for_page(
            struct_page
        )

        order = order_similarity_details(
            ref_blocks,
            ocr_blocks,
        )

        struct = structure_scores(
            ref_blocks,
            ocr_blocks,
        )

        if order["inversion_rate"] is not None:
            order_values.append(order["inversion_rate"])

        order_results.append({"page": page_number, **order})
        structure_results.append(
            struct
        )

        aggregate_reference_chars += reference_chars
        aggregate_ocr_chars += ocr_chars
        aggregate_reference_words += reference_words
        aggregate_ocr_words += ocr_words
        aggregate_character_edit_distance += character_distance
        aggregate_word_edit_distance += word_distance

        rows.append(
            {
                "page": page_number,
                "reference_chars": reference_chars,
                "ocr_chars": ocr_chars,
                "reference_words": reference_words,
                "ocr_words": ocr_words,
                "CER": page_cer,
                "WER": page_wer,
                "character_edit_distance": character_distance,
                "word_edit_distance": word_distance,
                # The primary rate is deliberately null when matching coverage
                # is too low to support a reading-order conclusion. Keep the
                # observed value separately for diagnostic review.
                "reading_order_inversion_rate": order["inversion_rate"],
                "reading_order_observed_inversion_rate": order[
                    "observed_inversion_rate"
                ],
                "reading_order_status": order["status"],
                "reading_order_inconclusive_reasons": ";".join(
                    order["inconclusive_reasons"]
                ),
                "reading_order_matched_blocks": order["matched_blocks"],
                "reading_order_reference_match_coverage": order[
                    "reference_match_coverage"
                ],
                "reading_order_ocr_match_coverage": order[
                    "ocr_match_coverage"
                ],
                "reading_order_inversions": order["inversions"],
                "structure_macro_f1": struct["macro_f1"],
                "excluded_visual_description_blocks": len(page_visual_labels),
            }
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{args.engine}_per_page.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate exact page edit distances, weighted by the corresponding
    # reference units.  This preserves page boundaries (so an error cannot be
    # aligned away by text on a later page) and avoids a corpus-sized O(n*m)
    # Python dynamic-programming pass after the per-page work has completed.

    structure_f1_values = [
        x["macro_f1"]
        for x in structure_results
        if x["macro_f1"] is not None
    ]

    evaluation = {
        "engine": args.engine,
        "pages_reference": len(reference_pages),
        "pages_structure": len(structure_pages),
        "pages_ocr": len(ocr_pages),
        "pages_evaluated": n,
        "reference": {
            "type": "conservative_silver_reference",
            "text_source": "PDF embedded text layer",
            "diagram_internal_text_excluded_when_not_trusted": True,
        },
        "aggregate": {
            "method": "exact per-page Levenshtein distances / total reference units",
            "character_edit_distance": aggregate_character_edit_distance,
            "word_edit_distance": aggregate_word_edit_distance,
            "reference_characters": aggregate_reference_chars,
            "ocr_characters": aggregate_ocr_chars,
            "reference_words": aggregate_reference_words,
            "ocr_words": aggregate_ocr_words,
            "weighted_CER": error_rate(
                aggregate_character_edit_distance,
                aggregate_reference_chars,
                aggregate_ocr_chars,
            ),
            "weighted_WER": error_rate(
                aggregate_word_edit_distance,
                aggregate_reference_words,
                aggregate_ocr_words,
            ),
            "mean_page_CER": (
                sum(r["CER"] for r in rows) / len(rows)
                if rows else None
            ),
            "mean_page_WER": (
                sum(r["WER"] for r in rows) / len(rows)
                if rows else None
            ),
            "mean_reading_order_inversion_rate": (
                sum(order_values) / len(order_values)
                if order_values
                else None
            ),
            "reading_order_pages_measured": sum(
                result["status"] == "measured"
                for result in order_results
            ),
            "reading_order_pages_inconclusive": sum(
                result["status"] == "inconclusive"
                for result in order_results
            ),
            "mean_reading_order_reference_match_coverage": (
                sum(
                    result["reference_match_coverage"]
                    for result in order_results
                    if result["reference_match_coverage"] is not None
                )
                / sum(
                    result["reference_match_coverage"] is not None
                    for result in order_results
                )
                if any(
                    result["reference_match_coverage"] is not None
                    for result in order_results
                )
                else None
            ),
            "mean_reading_order_ocr_match_coverage": (
                sum(
                    result["ocr_match_coverage"]
                    for result in order_results
                    if result["ocr_match_coverage"] is not None
                )
                / sum(
                    result["ocr_match_coverage"] is not None
                    for result in order_results
                )
                if any(
                    result["ocr_match_coverage"] is not None
                    for result in order_results
                )
                else None
            ),
            "mean_structure_macro_f1": (
                sum(structure_f1_values)
                / len(structure_f1_values)
                if structure_f1_values
                else None
            ),
        },
        "structure": structure_results,
        "reading_order": {
            "method": "maximum_cardinality_maximum_evidence_one_to_one_token_jaccard_or_bbox_iou",
            "text_token_jaccard_threshold": TEXT_TOKEN_JACCARD_THRESHOLD,
            "bbox_iou_threshold": BBOX_IOU_THRESHOLD,
            "bbox_iou_coordinate_requirement": (
                "matching explicit normalized pdf_points coordinate frame "
                "and page envelope"
            ),
            "minimum_matched_blocks": READING_ORDER_MIN_MATCHES,
            "minimum_coverage": READING_ORDER_MIN_COVERAGE,
            "per_page": order_results,
        },
        "visual_description_blocks": {
            "text_scoring": "excluded",
            "count": sum(visual_description_counts.values()),
            "by_type": dict(sorted(visual_description_counts.items())),
        },
        "benchmark_performance": read_performance(
            args.benchmark_summary
        ),
        "per_page_csv": str(csv_path),
    }

    json_path = out / f"{args.engine}_evaluation.json"

    json_path.write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            evaluation,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
