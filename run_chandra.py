#!/usr/bin/env python3
"""Run an optional hosted OCR adapter and save evaluation-friendly outputs.

Outputs:
  <stem>_datalab_response.json  raw completed API response
  <stem>_chandra.json           normalized page-list for the existing evaluator
  <stem>_chandra.md              markdown output (page-paginated when possible)
  <stem>_timing.json             client-side submission/poll/total timing

Usage:
  # Set DATALAB_API_KEY in the environment first.
  python run_chandra.py input.pdf --output-dir artifacts/chandra --mode balanced

Reuse a completed response without making an API request:
  python run_chandra.py --from-response artifacts/chandra/response.json --output-dir artifacts/chandra_offline

The adapter uploads the supplied document to an external service. Review the
provider's current terms, pricing, and data-handling requirements before use.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter, Retry

API_URL = "https://www.datalab.to/api/v1/convert"
TERMINAL_STATUSES = {"complete", "failed", "error"}


class _HTMLTextExtractor(HTMLParser):
    """Turn Datalab's block HTML into readable, page-ordered text."""

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
        value = html_lib.unescape("".join(self.parts))
        lines = [" ".join(line.split()) for line in value.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def html_to_text(value: Any) -> str:
    """Extract readable text while retaining source HTML separately in output."""
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def request_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def require_api_key() -> str:
    key = os.environ.get("DATALAB_API_KEY")
    if not key:
        raise SystemExit(
            "DATALAB_API_KEY is not set. Set it in the environment before rerunning."
        )
    return key


def post_convert(
    session: requests.Session,
    api_key: str,
    pdf_path: Path,
    mode: str,
    skip_cache: bool,
    paginate: bool,
) -> tuple[dict[str, Any], float]:
    headers = {"X-API-Key": api_key}
    data: dict[str, str] = {
        "output_format": "json,markdown",
        "mode": mode,
        "skip_cache": str(skip_cache).lower(),
        "paginate": str(paginate).lower(),
    }

    started = time.perf_counter()
    with pdf_path.open("rb") as fh:
        response = session.post(
            API_URL,
            headers=headers,
            files={"file": (pdf_path.name, fh, "application/pdf")},
            data=data,
            timeout=(30, 180),
        )

    response.raise_for_status()
    body = response.json()
    elapsed = time.perf_counter() - started

    if not body.get("success", True) or not body.get("request_check_url"):
        raise RuntimeError(f"Chandra submission failed: {json.dumps(body, indent=2)}")

    return body, elapsed


def poll_result(
    session: requests.Session,
    api_key: str,
    check_url: str,
    poll_interval: float,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float, int]:
    headers = {"X-API-Key": api_key}
    started = time.perf_counter()
    polls = 0

    while True:
        if time.perf_counter() - started > timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s while polling {check_url}"
            )

        response = session.get(check_url, headers=headers, timeout=(30, 120))
        response.raise_for_status()
        result = response.json()
        polls += 1
        status = str(result.get("status", "")).lower()

        print(f"  poll {polls:>3}: status={status or 'unknown'}", flush=True)

        if status in TERMINAL_STATUSES:
            elapsed = time.perf_counter() - started
            return result, elapsed, polls

        time.sleep(poll_interval)


def scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "content", "raw_text", "markdown", "html"):
            if isinstance(value.get(key), str):
                return value[key]
    return ""


def first_text(node: Any) -> str:
    """Recursively find the most plausible text field in an arbitrary block."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""

    # Prefer explicit text-bearing fields over nested children.
    for key in ("text", "markdown", "content", "raw_text"):
        text = scalar_text(node.get(key))
        if text:
            return text

    # Datalab's structured response puts the actual page/block text in HTML
    # for many documents.  The prior adapter omitted this field, which made a
    # successful Chandra result normalize to empty pages and invalidated CER /
    # WER comparisons.  Keep the original HTML on normalized blocks too, but
    # expose readable text to generic consumers and the evaluator.
    text = html_to_text(node.get("html"))
    if text:
        return text

    for key in ("children", "blocks", "items", "words", "text_lines", "lines"):
        value = node.get(key)
        if isinstance(value, list):
            parts = [first_text(item) for item in value]
            parts = [p for p in parts if p]
            if parts:
                return " ".join(parts)
        elif isinstance(value, dict):
            text = first_text(value)
            if text:
                return text
    return ""


def normalize_chandra_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve Datalab child-block HTML, text, labels, and page order."""
    children = page.get("children")
    if not isinstance(children, list):
        return []

    blocks: list[dict[str, Any]] = []
    for source_order, child in enumerate(children, start=1):
        if not isinstance(child, dict):
            continue
        text = first_text(child)
        if not text:
            continue
        label = child.get("block_type") or child.get("type") or "Text"
        block: dict[str, Any] = {
            "block_type": label,
            "type": str(child.get("type") or label).lower(),
            "text": text,
            "source_reading_order": child.get("reading_order", source_order),
            "reading_order": len(blocks) + 1,
        }
        bbox = bbox_of(child)
        if bbox:
            block["bbox"] = bbox
        if isinstance(child.get("html"), str):
            block["html"] = child["html"]
        blocks.append(block)
    return blocks


def bbox_of(node: Any) -> list[float] | None:
    if not isinstance(node, dict):
        return None
    for key in ("bbox", "box", "bounding_box", "coordinates"):
        value = node.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
            except (TypeError, ValueError):
                return None
    return None


def word_from_node(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    text = scalar_text(node)
    if not text:
        return None
    bbox = bbox_of(node)
    out: dict[str, Any] = {"text": text}
    if bbox:
        out["bbox"] = bbox
    # Datalab may expose confidence at word-level in current API responses.
    for key in ("confidence", "score"):
        value = node.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
            break
    return out


def extract_words(node: Any) -> list[dict[str, Any]]:
    """Recursively collect explicit word/line nodes without fabricating boxes."""
    words: list[dict[str, Any]] = []
    if not isinstance(node, (dict, list)):
        return words

    if isinstance(node, list):
        for item in node:
            words.extend(extract_words(item))
        return words

    # Don't treat arbitrary paragraphs as individual words.
    for key in ("words", "text_lines", "lines"):
        value = node.get(key)
        if isinstance(value, list):
            for item in value:
                candidate = word_from_node(item)
                if candidate:
                    words.append(candidate)
                else:
                    words.extend(extract_words(item))

    for key in ("children", "blocks", "items"):
        value = node.get(key)
        if isinstance(value, (dict, list)):
            words.extend(extract_words(value))

    return words


def collect_page_like_nodes(obj: Any) -> list[dict[str, Any]]:
    """Extract only Datalab's top-level Page nodes.

    Datalab's JSON document structure is:
        result.json.children = [
            {"block_type": "Page", ...},
            ...
        ]

    Nested blocks also contain a `page` field, so we must NOT recursively
    treat every dictionary with a `page` key as a page.
    """
    if not isinstance(obj, dict):
        return []

    children = obj.get("children")
    if not isinstance(children, list):
        return []

    pages = [
        node
        for node in children
        if isinstance(node, dict)
        and node.get("block_type") == "Page"
    ]

    return pages

def split_paginated_markdown(markdown: str) -> list[str]:
    """Best-effort split for Datalab's paginate=True markdown.

    Datalab documents pagination as page delimiters. We support both the common
    horizontal-rule form and an explicit page-number delimiter when present.
    """
    if not markdown.strip():
        return []

    # First split on horizontal rules that are on their own line.
    parts = re.split(r"\n\s*[-*_]{3,}\s*\n", markdown)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [markdown.strip()]


def normalize_chandra(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Prefer structured JSON when available.
    raw_json = result.get("json")
    pages = collect_page_like_nodes(raw_json)

    if pages:
        normalized: list[dict[str, Any]] = []
        pages_sorted = sorted(
            pages,
            key=lambda p: (
                p.get("page", p.get("page_number", 10**9))
                if isinstance(p.get("page", p.get("page_number", 10**9)), (int, float))
                else 10**9
            ),
        )
        for idx, page in enumerate(pages_sorted, start=1):
            blocks = normalize_chandra_blocks(page)
            entry: dict[str, Any] = {
                "page": page.get("page", page.get("page_number", idx)),
                # Prefer ordered child blocks: this retains labels and lets the
                # evaluator score reading order and structure.  A page-level
                # HTML/text field remains the fallback for sparse API output.
                "text": "\n".join(block["text"] for block in blocks) if blocks else first_text(page),
            }
            if blocks:
                entry["blocks"] = blocks
            words = extract_words(page)
            if words:
                entry["words"] = words
            if bbox_of(page):
                entry["bbox"] = bbox_of(page)
            # Keep a small amount of structure useful to debugging/evaluation.
            for key in ("block_type", "type", "reading_order"):
                if key in page:
                    entry[key] = page[key]
            normalized.append(entry)
        return normalized

    markdown = scalar_text(result.get("markdown"))
    chunks = split_paginated_markdown(markdown)
    if chunks:
        return [{"page": idx, "text": text} for idx, text in enumerate(chunks, start=1)]

    # Last-resort single-page output.
    return [{"page": 1, "text": first_text(raw_json) or markdown}]


def normalized_document(result: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Build the evaluator-compatible Chandra document without writing files."""
    pages = normalize_chandra(result)
    return {
        "engine": "chandra",
        "source": source_name,
        "page_count": len(pages),
        "pages": pages,
        "metadata": {
            "checkpoint_id": result.get("checkpoint_id"),
            "runtime": result.get("runtime"),
            "total_cost": result.get("total_cost"),
            "cost_breakdown": result.get("cost_breakdown"),
            "parse_quality_score": result.get("parse_quality_score"),
            "versions": result.get("versions"),
        },
    }


def write_normalized_output(
    output_dir: Path,
    stem: str,
    source_name: str,
    result: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Write only the normalized JSON used by the evaluator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalized_document(result, source_name)
    output_path = output_dir / f"{stem}_chandra.json"
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path, normalized


def load_saved_response(response_path: Path) -> dict[str, Any]:
    """Load a completed Datalab result, accepting this script's raw wrapper too."""
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Saved response not found: {response_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Saved response is not valid JSON: {response_path}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Saved response must be a JSON object: {response_path}")

    # Live runs save {"submission": ..., "result": ...}.  Supporting a raw
    # completed result as well keeps this mode useful for API responses saved
    # outside this script.
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise ValueError(
            f"Saved response has a non-object 'result' field: {response_path}"
        )
    return result


def inferred_response_stem(response_path: Path) -> str:
    """Recover the original PDF stem from this script's raw response name."""
    suffix = "_datalab_response.json"
    if response_path.name.endswith(suffix):
        return response_path.name[: -len(suffix)]
    return response_path.stem


def normalize_saved_response(
    response_path: Path,
    output_dir: Path,
    source_pdf: Path | None = None,
    write_markdown: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Normalize an existing response locally, without credentials or network I/O.

    The raw response is read only.  Client submission/poll timings cannot be
    reconstructed, so this mode intentionally does not emit a timing file.
    Markdown is copied only when explicitly requested and present in the result.
    """
    result = load_saved_response(response_path)
    stem = source_pdf.stem if source_pdf is not None else inferred_response_stem(response_path)
    source_name = source_pdf.name if source_pdf is not None else f"{stem}.pdf"
    output_path, normalized = write_normalized_output(
        output_dir, stem, source_name, result
    )

    if write_markdown and isinstance(result.get("markdown"), str):
        (output_dir / f"{stem}_chandra.md").write_text(
            result["markdown"], encoding="utf-8"
        )

    return output_path, normalized


def write_outputs(
    output_dir: Path,
    pdf_path: Path,
    submit_response: dict[str, Any],
    result: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    (output_dir / f"{stem}_datalab_response.json").write_text(
        json.dumps({"submission": submit_response, "result": result}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_normalized_output(output_dir, stem, pdf_path.name, result)

    markdown = result.get("markdown")
    if isinstance(markdown, str):
        (output_dir / f"{stem}_chandra.md").write_text(markdown, encoding="utf-8")

    (output_dir / f"{stem}_timing.json").write_text(
        json.dumps(timing, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the hosted OCR adapter")
    ap.add_argument("pdf", type=Path, nargs="?", help="PDF to submit to Datalab")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/chandra"))
    ap.add_argument(
        "--from-response",
        type=Path,
        metavar="JSON",
        help="Normalize a saved completed Datalab response locally; makes no API request",
    )
    ap.add_argument(
        "--source-pdf",
        type=Path,
        help="Original PDF name to use for offline output naming and metadata",
    )
    ap.add_argument(
        "--write-markdown",
        action="store_true",
        help="With --from-response, copy embedded markdown when it is available",
    )
    ap.add_argument("--mode", choices=["balanced", "accurate"], default="balanced")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--skip-cache", action="store_true", default=True,
                    help="Force a fresh run (enabled by default for benchmarking)")
    args = ap.parse_args()

    if args.from_response is not None:
        if args.pdf is not None:
            ap.error("pdf cannot be used with --from-response; use --source-pdf to name it")
        response_path = args.from_response.resolve()
        try:
            output_path, normalized = normalize_saved_response(
                response_path,
                args.output_dir,
                source_pdf=args.source_pdf,
                write_markdown=args.write_markdown,
            )
        except ValueError as error:
            ap.error(str(error))
        print(f"Offline:    {len(normalized['pages'])} pages")
        print(f"Saved to:   {output_path.resolve()}")
        return 0

    if args.pdf is None:
        ap.error("pdf is required unless --from-response is used")
    if args.source_pdf is not None:
        ap.error("--source-pdf can only be used with --from-response")
    if args.write_markdown:
        ap.error("--write-markdown can only be used with --from-response")

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a PDF, got: {pdf_path}")

    api_key = require_api_key()
    session = request_session()

    print(f"Input:      {pdf_path}")
    print(f"Mode:       {args.mode}")
    print(f"Cache:      {'disabled' if args.skip_cache else 'enabled'}")
    print("Submitting to Datalab...")

    total_started = time.perf_counter()
    submit_response, submit_elapsed = post_convert(
        session,
        api_key,
        pdf_path,
        args.mode,
        args.skip_cache,
        paginate=True,
    )

    print(f"Submitted:  request_id={submit_response.get('request_id')}")
    print("Polling...")
    result, poll_elapsed, polls = poll_result(
        session,
        api_key,
        submit_response["request_check_url"],
        args.poll_interval,
        args.timeout,
    )

    total_elapsed = time.perf_counter() - total_started
    status = str(result.get("status", "unknown")).lower()

    timing = {
        "client_total_seconds": total_elapsed,
        "client_submission_seconds": submit_elapsed,
        "client_polling_seconds": poll_elapsed,
        "poll_count": polls,
        "datalab_runtime_seconds": result.get("runtime"),
        "request_id": submit_response.get("request_id"),
        "status": status,
        "mode": args.mode,
        "skip_cache": args.skip_cache,
        "source_pdf": pdf_path.name,
    }

    write_outputs(args.output_dir, pdf_path, submit_response, result, timing)

    if status != "complete" or not result.get("success", False):
        print(json.dumps({"timing": timing, "error": result.get("error")}, indent=2), file=sys.stderr)
        return 1

    normalized = normalize_chandra(result)
    print(f"Complete:   {len(normalized)} pages")
    print(f"Client time:{total_elapsed:.2f}s")
    if isinstance(result.get("runtime"), (int, float)):
        print(f"API runtime:{result['runtime']:.2f}s")
    if result.get("total_cost") is not None:
        print(f"Cost:       {result['total_cost']}")
    print(f"Saved to:   {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
