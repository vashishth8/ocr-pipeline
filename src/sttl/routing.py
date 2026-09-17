#!/usr/bin/env python3
"""Evaluate an STTL rich artifact against reviewed text by routing outcome.

Unlike ``evaluate_ocr.py``, this tool is deliberately text-only and requires
human-adjudicated transcription. It is designed to answer whether native,
Tesseract, and Surya pages are appropriate for the document types actually in
scope, without treating one OCR system as ground truth for another.

The truth file uses the existing ``sttl-human-transcription/v1`` page schema.
Only its reviewed pages are evaluated, but each must be present in the rich
artifact and the truth source digest must match the artifact's sibling job.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any

from sttl.evaluate import character_error_stats, error_rate, word_error_stats
from sttl.jsonio import load_json_object_bytes

HUMAN_TRANSCRIPTION_SCHEMA_VERSION = "sttl-human-transcription/v1"
ROUTING_EVALUATION_SCHEMA_VERSION = "sttl-routing-evaluation/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    """Load one UTF-8 JSON object and return its exact-byte digest."""
    return load_json_object_bytes(path)


def positive_page_number(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def page_map(pages: Any, context: str) -> dict[int, dict[str, Any]]:
    """Index a non-empty page array, rejecting ambiguous page identities."""
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{context} must be a non-empty list")
    result: dict[int, dict[str, Any]] = {}
    for index, entry in enumerate(pages, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"{context}[{index}] must be an object")
        page = positive_page_number(entry.get("page"), f"{context}[{index}].page")
        if page in result:
            raise ValueError(f"{context} contains duplicate page {page}")
        result[page] = entry
    return result


def source_digest(job: dict[str, Any], job_path: Path) -> str:
    source = job.get("source")
    digest = source.get("content_sha256") if isinstance(source, dict) else None
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{job_path} source.content_sha256 must be a lowercase SHA-256 digest")
    return digest


def reviewed_truth_pages(truth: dict[str, Any], truth_path: Path) -> dict[int, str]:
    if truth.get("schema_version") != HUMAN_TRANSCRIPTION_SCHEMA_VERSION:
        raise ValueError(
            f"{truth_path}.schema_version must be {HUMAN_TRANSCRIPTION_SCHEMA_VERSION!r}"
        )
    result: dict[int, str] = {}
    for page, entry in page_map(truth.get("pages"), f"{truth_path}.pages").items():
        text = entry.get("text")
        review = entry.get("review")
        if not isinstance(text, str):
            raise ValueError(f"{truth_path}.pages[{page}].text must be a string")
        if not isinstance(review, dict) or review.get("status") != "adjudicated":
            raise ValueError(f"{truth_path}.pages[{page}].review.status must be 'adjudicated'")
        result[page] = text
    return result


def rich_page_metadata(
    page: dict[str, Any], page_number: int
) -> tuple[str, str, str, str | None, str]:
    """Return authoritative engine/outcome and audited routing fields."""
    authoritative = page.get("authoritative")
    routing = page.get("routing")
    if not isinstance(authoritative, dict):
        raise ValueError(f"rich.pages[{page_number}].authoritative must be an object")
    if not isinstance(routing, dict):
        raise ValueError(f"rich.pages[{page_number}].routing must be an object")
    engine = authoritative.get("engine")
    outcome = authoritative.get("outcome")
    route = routing.get("route")
    classifier_route = routing.get("classifier_route")
    text = authoritative.get("text")
    if not isinstance(engine, str) or not engine:
        raise ValueError(
            f"rich.pages[{page_number}].authoritative.engine must be a non-empty string"
        )
    if not isinstance(outcome, str) or not outcome:
        raise ValueError(
            f"rich.pages[{page_number}].authoritative.outcome must be a non-empty string"
        )
    if not isinstance(route, str) or not route:
        raise ValueError(f"rich.pages[{page_number}].routing.route must be a non-empty string")
    if classifier_route is not None and not isinstance(classifier_route, str):
        raise ValueError(
            f"rich.pages[{page_number}].routing.classifier_route must be a string when present"
        )
    if not isinstance(text, str):
        raise ValueError(f"rich.pages[{page_number}].authoritative.text must be a string")
    return engine, route, outcome, classifier_route, text


def metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact per-page edit-distance metrics without joining pages."""
    values = list(rows)
    character_distance = sum(row["character_edit_distance"] for row in values)
    word_distance = sum(row["word_edit_distance"] for row in values)
    reference_characters = sum(row["reference_chars"] for row in values)
    ocr_characters = sum(row["ocr_chars"] for row in values)
    reference_words = sum(row["reference_words"] for row in values)
    ocr_words = sum(row["ocr_words"] for row in values)
    return {
        "pages": len(values),
        "character_edit_distance": character_distance,
        "word_edit_distance": word_distance,
        "reference_characters": reference_characters,
        "ocr_characters": ocr_characters,
        "reference_words": reference_words,
        "ocr_words": ocr_words,
        "weighted_CER": error_rate(character_distance, reference_characters, ocr_characters),
        "weighted_WER": error_rate(word_distance, reference_words, ocr_words),
        "mean_page_CER": fmean(row["CER"] for row in values) if values else None,
        "mean_page_WER": fmean(row["WER"] for row in values) if values else None,
    }


def stratify(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: metrics(groups[key]) for key in sorted(groups)}


def evaluate_routing(
    rich_path: Path,
    truth_path: Path,
    *,
    job_path: Path | None = None,
) -> dict[str, Any]:
    """Score reviewed pages and expose their authoritative routing strata.

    The strata describe the deployed cascade: they are useful for finding a
    weak route, but they are not a causal engine comparison because routing
    conditions determine which pages reach each engine.
    """
    job_path = job_path or rich_path.parent / "job.json"
    rich, rich_sha256 = load_json_bytes(rich_path)
    truth, truth_sha256 = load_json_bytes(truth_path)
    job, job_sha256 = load_json_bytes(job_path)

    if rich.get("schema_version") != "cascade-ocr/rich-v1":
        raise ValueError(f"{rich_path} must be a cascade rich artifact")
    truth_digest = truth.get("source_sha256")
    if not isinstance(truth_digest, str) or not SHA256_RE.fullmatch(truth_digest):
        raise ValueError(f"{truth_path}.source_sha256 must be a lowercase SHA-256 digest")
    if truth_digest != source_digest(job, job_path):
        raise ValueError(
            f"{truth_path}.source_sha256 does not match {job_path} source.content_sha256"
        )

    rich_pages = page_map(rich.get("pages"), f"{rich_path}.pages")
    truth_pages = reviewed_truth_pages(truth, truth_path)
    unavailable = sorted(set(truth_pages) - set(rich_pages))
    if unavailable:
        raise ValueError(f"{truth_path} contains pages absent from {rich_path}: {unavailable}")

    rows: list[dict[str, Any]] = []
    for page in sorted(truth_pages):
        engine, route, outcome, classifier_route, hypothesis = rich_page_metadata(
            rich_pages[page], page
        )
        reference = truth_pages[page]
        character_distance, reference_chars, ocr_chars = character_error_stats(
            reference, hypothesis
        )
        word_distance, reference_words, ocr_words = word_error_stats(reference, hypothesis)
        rows.append(
            {
                "page": page,
                "engine": engine,
                "route": route,
                "classifier_route": classifier_route or "",
                "outcome": outcome,
                "reference_chars": reference_chars,
                "ocr_chars": ocr_chars,
                "reference_words": reference_words,
                "ocr_words": ocr_words,
                "character_edit_distance": character_distance,
                "word_edit_distance": word_distance,
                "CER": error_rate(character_distance, reference_chars, ocr_chars),
                "WER": error_rate(word_distance, reference_words, ocr_words),
            }
        )

    return {
        "schema_version": ROUTING_EVALUATION_SCHEMA_VERSION,
        "evaluation": {
            "reference_type": "human_adjudicated_transcription",
            "method": "exact per-page Levenshtein distances / total reference units",
            "evaluated_page_ids": [row["page"] for row in rows],
        },
        "provenance": {
            "rich_sha256": rich_sha256,
            "human_truth_sha256": truth_sha256,
            "job_sha256": job_sha256,
            "source_sha256": truth_digest,
            "rich_pipeline_version": rich.get("metadata", {}).get("pipeline_version")
            if isinstance(rich.get("metadata"), dict)
            else None,
        },
        "coverage": {
            "rich_pages_available": len(rich_pages),
            "human_reviewed_pages": len(truth_pages),
            "pages_evaluated": len(rows),
            "unreviewed_rich_pages": sorted(set(rich_pages) - set(truth_pages)),
        },
        "aggregate": metrics(rows),
        "by_engine": stratify(rows, "engine"),
        "by_route": stratify(rows, "route"),
        "by_outcome": stratify(rows, "outcome"),
        "by_engine_route": stratify(
            [{**row, "engine_route": f"{row['engine']} / {row['route']}"} for row in rows],
            "engine_route",
        ),
        "per_page": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich-json", type=Path, required=True, help="STTL *_rich.json artifact")
    parser.add_argument(
        "--human-truth", type=Path, required=True, help="Reviewed sttl-human-transcription/v1 JSON"
    )
    parser.add_argument(
        "--job-json", type=Path, help="Sibling job.json; inferred from --rich-json when omitted"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/routing-evaluation"))
    args = parser.parse_args()

    report = evaluate_routing(args.rich_json, args.human_truth, job_path=args.job_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "routing_per_page.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "page",
            "engine",
            "route",
            "classifier_route",
            "outcome",
            "reference_chars",
            "ocr_chars",
            "reference_words",
            "ocr_words",
            "character_edit_distance",
            "word_edit_distance",
            "CER",
            "WER",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["per_page"])
    report["per_page_csv"] = str(csv_path)
    output_path = args.out_dir / "routing_evaluation.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
