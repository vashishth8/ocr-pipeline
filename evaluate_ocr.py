#!/usr/bin/env python3
"""
Evaluate OCR output against STTL silver references.

Important:
- CER/WER use reference.json, but only for text that belongs to
  trusted structural regions when structure.json is available.
- Reading order is evaluated at block level, not word-level bbox guesses.
- Structure classification is reported as per-type precision/recall/F1
  when OCR output supplies block labels.
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
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"\S+", re.UNICODE)

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
        if not text:
            continue

        result.append(
            {
                "text": text,
                # ``block_type`` is the semantic label preserved by the
                # normalized schema.  ``type`` is often merely "text".
                "label": semantic_block_label(block),
                "bbox": bbox_of(block),
                "reading_order": block.get("reading_order"),
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
        [{"text": text, "label": None, "bbox": None, "reading_order": 1}]
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
            if not text:
                continue

            result.append(
                {
                    "text": text,
                    "label": semantic_block_label(block),
                    "bbox": bbox_of(block),
                    "reading_order": block.get("reading_order"),
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
            if text:
                result.append(
                    {
                        "text": text,
                        "label": semantic_block_label(block),
                        "bbox": bbox_of(block),
                        "reading_order": block.get("reading_order"),
                    }
                )
        if result:
            return result
        if raw:
            return []

    text = generic_page_text(page)
    return (
        [{"text": text, "label": None, "bbox": None, "reading_order": i}]
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

    for block in structure_page.get("blocks", []):
        if block.get("type") in REFERENCE_TEXT_TYPES:
            blocks.append(
                {
                    "type": block.get("type"),
                    "text": normalize_text(block.get("text", "")),
                    "bbox": bbox_of(block),
                    "reading_order": block.get("reading_order"),
                }
            )

    return blocks


def order_similarity(reference_blocks, ocr_blocks):
    """
    Block-level order score.

    Match OCR blocks to reference blocks by normalized text token overlap
    (greedy, one-to-one), then calculate the fraction of matched pairs whose
    relative order is preserved.

    Returns:
      inversion_rate, matched_blocks, inversions
    """
    ref = [tokens(b["text"]) for b in reference_blocks]
    hyp = [tokens(b["text"]) for b in ocr_blocks]

    ref_sets = [set(x) for x in ref]
    used = set()
    mapped = []

    for h_index, h_tokens in enumerate(hyp):
        if not h_tokens:
            continue

        best = None
        best_score = 0.0

        for r_index, r_set in enumerate(ref_sets):
            if r_index in used or not r_set:
                continue

            overlap = len(set(h_tokens) & r_set) / max(
                1,
                len(set(h_tokens) | r_set),
            )

            if overlap > best_score:
                best = r_index
                best_score = overlap

        if best is not None and best_score >= 0.25:
            used.add(best)
            mapped.append(best)

    if len(mapped) < 2:
        return None, len(mapped), 0

    inversions = 0
    pairs = 0

    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            pairs += 1
            if mapped[i] > mapped[j]:
                inversions += 1

    return inversions / pairs if pairs else 0.0, len(mapped), inversions


def structure_scores(reference_blocks, ocr_blocks):
    ref_types = Counter(
        b["type"] for b in reference_blocks
    )

    ocr_types = Counter()

    for block in ocr_blocks:
        label = canonical_label(block.get("label"))
        if label:
            ocr_types[label] += 1

    types = sorted(
        set(ref_types) | set(ocr_types)
    )

    per_type = {}

    for t in types:
        tp = min(ref_types[t], ocr_types[t])
        precision = (
            tp / ocr_types[t]
            if ocr_types[t]
            else 0.0
        )
        recall = (
            tp / ref_types[t]
            if ref_types[t]
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_type[t] = {
            "reference": ref_types[t],
            "ocr": ocr_types[t],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    relevant_types = [
        x for x in per_type.values()
        if x["reference"] > 0
        or x["ocr"] > 0
    ]

    macro_f1 = (
        sum(x["f1"] for x in relevant_types)
        / len(relevant_types)
        if relevant_types
        else None
    )

    return {
        "per_type": per_type,
        "macro_f1": macro_f1,
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

        order_rate, matched, inversions = (
            order_similarity(
                ref_blocks,
                ocr_blocks,
            )
        )

        struct = structure_scores(
            ref_blocks,
            ocr_blocks,
        )

        if order_rate is not None:
            order_values.append(order_rate)

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
                "reading_order_inversion_rate": order_rate,
                "reading_order_matched_blocks": matched,
                "reading_order_inversions": inversions,
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
            "mean_structure_macro_f1": (
                sum(structure_f1_values)
                / len(structure_f1_values)
                if structure_f1_values
                else None
            ),
        },
        "structure": structure_results,
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
