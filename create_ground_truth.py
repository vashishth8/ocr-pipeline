#!/usr/bin/env python3
"""
Create a conservative silver reference for a PDF OCR evaluation.

Outputs:
  <output-dir>/reference.json
      Raw word-level PDF text layer + coordinates. This is the source for
      text accuracy measurements.

  <output-dir>/structure.json
      Conservative structural reference derived from PDF geometry. Only
      high-confidence structures are labeled; uncertain content is "text".

This deliberately avoids treating text extracted from large diagram regions
as trustworthy textual ground truth. Source PDFs and generated references are
local artifacts; keep them out of version control unless they are cleared for
redistribution.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup


def norm(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip()


def bbox_union(boxes):
    boxes = [b for b in boxes if b and len(b) == 4]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def extract_word(node):
    text = node.get_text(" ", strip=True)
    if not text:
        return None
    return {
        "text": text,
        "x0": float(node.get("xmin", 0)),
        "y0": float(node.get("ymin", 0)),
        "x1": float(node.get("xmax", 0)),
        "y1": float(node.get("ymax", 0)),
    }


def block_text(words):
    return norm(" ".join(w["text"] for w in words))


def parse_pdf(pdf_path: Path, xml_path: Path):
    subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf_path), str(xml_path)],
        check=True,
    )

    soup = BeautifulSoup(
        xml_path.read_text(encoding="utf-8", errors="replace"),
        "html.parser",
    )

    page_nodes = soup.find_all("page")
    if not page_nodes:
        raise RuntimeError("pdftotext returned no <page> nodes")

    pages = []

    for page_no, page in enumerate(page_nodes, 1):
        width = float(page.get("width", 0))
        height = float(page.get("height", 0))

        all_words = []
        native_blocks = []

        for native_index, raw_block in enumerate(page.find_all("block")):
            words = []
            for raw_word in raw_block.find_all("word"):
                w = extract_word(raw_word)
                if w:
                    words.append(w)
                    all_words.append(w)

            if words:
                native_blocks.append(
                    {
                        "native_index": native_index,
                        "words": words,
                        "text": block_text(words),
                        "bbox": bbox_union(
                            [[w["x0"], w["y0"], w["x1"], w["y1"]] for w in words]
                        ),
                    }
                )

        pages.append(
            {
                "page": page_no,
                "width": width,
                "height": height,
                "words": all_words,
                "native_blocks": native_blocks,
            }
        )

    return pages


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+", text.strip()))


def is_section_heading(text: str) -> bool:
    text = norm(text)
    return bool(
        re.fullmatch(r"\d{3,4}\s+[A-Z][A-Z0-9 /&()'.,\-]*", text)
        or re.match(r"^\d{3,4}(?:\.\d+)+\s+[A-Z]", text)
    )


def is_caption(text: str) -> bool:
    text = norm(text)
    return bool(
        re.match(r"^(?:Fig\.|Figure)\s*[\w.\-]+", text, re.I)
        or re.match(r"^Table\s*[\w.\-]+", text, re.I)
    )


def is_table_like(text: str) -> bool:
    # High-confidence textual table row signal.
    if re.match(r"^[A-Z][A-Z0-9]{1,10}\s*:", text):
        return True
    # Common table item numbering + measurement/content.
    if re.match(r"^(?:\d+\)|[A-Za-z]\))\s+", text):
        return True
    return False


def build_visual_lines(words):
    """Cluster words into visual lines using y-center proximity."""
    words = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2, w["x0"]))
    lines = []

    for word in words:
        cy = (word["y0"] + word["y1"]) / 2
        h = max(word["y1"] - word["y0"], 1.0)

        best = None
        best_distance = None

        for line in reversed(lines[-8:]):
            tolerance = max(3.5, min(7.0, max(line["avg_h"], h) * 0.55))
            d = abs(line["cy"] - cy)
            if d <= tolerance and (best_distance is None or d < best_distance):
                best = line
                best_distance = d

        if best is None:
            lines.append({"words": [word], "cy": cy, "avg_h": h})
        else:
            best["words"].append(word)
            centers = [(w["y0"] + w["y1"]) / 2 for w in best["words"]]
            heights = [max(w["y1"] - w["y0"], 1.0) for w in best["words"]]
            best["cy"] = sum(centers) / len(centers)
            best["avg_h"] = sum(heights) / len(heights)

    out = []
    for idx, line in enumerate(sorted(lines, key=lambda x: x["cy"])):
        ws = sorted(line["words"], key=lambda w: w["x0"])
        out.append(
            {
                "line": idx,
                "words": ws,
                "text": block_text(ws),
                "bbox": bbox_union(
                    [[w["x0"], w["y0"], w["x1"], w["y1"]] for w in ws]
                ),
            }
        )
    return out


def merge_heading_fragments(lines):
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            combined = norm(cur["text"] + " " + nxt["text"])
            vertical = abs(cur["bbox"][1] - nxt["bbox"][1])

            if (
                vertical <= 16
                and len(cur["text"]) <= 30
                and len(nxt["text"]) <= 100
                and is_section_heading(combined)
            ):
                ws = cur["words"] + nxt["words"]
                ws = sorted(ws, key=lambda w: (w["y0"], w["x0"]))
                out.append(
                    {
                        "line": cur["line"],
                        "words": ws,
                        "text": block_text(ws),
                        "bbox": bbox_union(
                            [[w["x0"], w["y0"], w["x1"], w["y1"]] for w in ws]
                        ),
                    }
                )
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def detect_repeated_headers(all_pages):
    counts = Counter()

    for page in all_pages:
        seen = set()
        for line in build_visual_lines(page["words"]):
            if line["bbox"][1] <= page["height"] * 0.15:
                t = norm(line["text"]).lower()
                if t and t not in seen:
                    counts[t] += 1
                    seen.add(t)

    return {t for t, n in counts.items() if n >= 2}


def build_structure_page(page, repeated_headers):
    lines = merge_heading_fragments(
        build_visual_lines(page["words"])
    )

    # Identify high-confidence page furniture.
    header_idxs = set()
    footer_idxs = set()

    for i, line in enumerate(lines):
        text = norm(line["text"])
        y0, y1 = line["bbox"][1], line["bbox"][3]

        if (
            text.lower() in repeated_headers
            and y0 <= page["height"] * 0.15
        ):
            header_idxs.add(i)

        if (
            is_page_number(text)
            and y1 >= page["height"] * 0.90
        ):
            footer_idxs.add(i)

    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        text = norm(line["text"])

        if i in header_idxs:
            blocks.append({
                "type": "page_header",
                "text": text,
                "bbox": line["bbox"],
            })
            i += 1
            continue

        if i in footer_idxs:
            blocks.append({
                "type": "page_footer",
                "text": text,
                "bbox": line["bbox"],
            })
            i += 1
            continue

        if is_section_heading(text):
            blocks.append({
                "type": "section_heading",
                "text": text,
                "bbox": line["bbox"],
            })
            i += 1
            continue

        if is_caption(text):
            blocks.append({
                "type": (
                    "table_caption"
                    if text.lower().startswith("table")
                    else "figure_caption"
                ),
                "text": text,
                "bbox": line["bbox"],
            })
            i += 1
            continue

        # Detect a run of table-like rows. We require >= 3 rows.
        run = [line]
        j = i + 1
        while j < len(lines):
            t = norm(lines[j]["text"])
            if is_section_heading(t) or is_caption(t):
                break
            gap = lines[j]["bbox"][1] - run[-1]["bbox"][3]
            if gap > 20:
                break
            if is_table_like(t):
                run.append(lines[j])
                j += 1
            elif run and len(run) >= 3 and len(t) < 140 and ":" in t:
                run.append(lines[j])
                j += 1
            else:
                break

        if len(run) >= 3:
            all_words = [w for r in run for w in r["words"]]
            blocks.append({
                "type": "table",
                "text": block_text(all_words),
                "bbox": bbox_union(
                    [[w["x0"], w["y0"], w["x1"], w["y1"]] for w in all_words]
                ),
            })
            i = j
            continue

        # Paragraph: collect nearby wrapped lines until a strong boundary.
        para = [line]
        j = i + 1

        while j < len(lines):
            candidate = lines[j]
            t = norm(candidate["text"])

            if t in {"",}:
                break
            if j in header_idxs or j in footer_idxs:
                break
            if is_section_heading(t) or is_caption(t):
                break
            if is_table_like(t):
                break

            prev = para[-1]
            gap = candidate["bbox"][1] - prev["bbox"][3]
            prev_h = max(prev["bbox"][3] - prev["bbox"][1], 1.0)

            if gap <= prev_h * 1.65:
                para.append(candidate)
                j += 1
            else:
                break

        all_words = [w for r in para for w in r["words"]]
        blocks.append({
            "type": "paragraph",
            "text": block_text(all_words),
            "bbox": bbox_union(
                [[w["x0"], w["y0"], w["x1"], w["y1"]] for w in all_words]
            ),
        })
        i = j

    # Remove paragraph blocks that are clearly diagram debris:
    # very short isolated fragments in the middle of the page, while
    # keeping ordinary short text. We only use a conservative rule.
    cleaned = []
    for block in blocks:
        if block["type"] == "paragraph":
            t = block["text"]
            h = block["bbox"][3] - block["bbox"][1]
            if (
                len(t) <= 3
                and page["height"] * 0.18 < block["bbox"][1] < page["height"] * 0.88
                and h > 0
            ):
                continue
        cleaned.append(block)

    for order, block in enumerate(cleaned):
        block["id"] = f'p{page["page"]}-b{order}'
        block["reading_order"] = order

    summary = Counter(b["type"] for b in cleaned)

    return {
        "page": page["page"],
        "width": page["width"],
        "height": page["height"],
        "blocks": cleaned,
        "reading_order": [b["id"] for b in cleaned],
        "structure_summary": dict(summary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="source PDF; it is never copied into the output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ground_truth"),
        help="directory for local reference artifacts (default: artifacts/ground_truth)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdf_path = args.pdf.expanduser()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF does not exist: {pdf_path}")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "reference.json"
    structure_path = output_dir / "structure.json"

    with tempfile.TemporaryDirectory(prefix="sttl-bbox-") as temporary:
        pages = parse_pdf(pdf_path, Path(temporary) / "layout.html")
    print(f"PDF pages detected: {len(pages)}")

    reference_pages = []
    for page in pages:
        reference_pages.append({
            "page": page["page"],
            "width": page["width"],
            "height": page["height"],
            "words": page["words"],
            "text": " ".join(w["text"] for w in page["words"]),
        })

    reference_path.write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "type": "pdf_text_layer_silver_ground_truth",
                "page_count": len(reference_pages),
                "pages": reference_pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    headers = detect_repeated_headers(pages)

    structure_pages = [
        build_structure_page(page, headers)
        for page in pages
    ]

    structure_path.write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "type": "conservative_geometry_derived_silver_structure",
                "classification": [
                    "page_header",
                    "page_footer",
                    "section_heading",
                    "heading",
                    "paragraph",
                    "table",
                    "table_caption",
                    "figure_caption",
                ],
                "notes": [
                    "reference.json is raw PDF text-layer data.",
                    "structure.json is conservative silver structure.",
                    "Diagram-internal PDF text is not trusted as text ground truth.",
                    "Unknown regions are not forced into semantic classes.",
                ],
                "page_count": len(structure_pages),
                "pages": structure_pages,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    total_words = sum(len(p["words"]) for p in reference_pages)
    total_blocks = sum(len(p["blocks"]) for p in structure_pages)

    print(f"Created: {reference_path}")
    print(f"Created: {structure_path}")
    print(f"Pages:   {len(reference_pages)}")
    print(f"Words:   {total_words}")
    print(f"Blocks:  {total_blocks}")

    print("\nStructure by page:")
    for p in structure_pages:
        print(
            f'  Page {p["page"]}: '
            f'{len(p["blocks"])} blocks '
            f'{p["structure_summary"]}'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
