#!/usr/bin/env python3
"""Calibrate STTL's Tesseract acceptance gate against reviewed transcription.

This intentionally evaluates the raw Tesseract evidence layer rather than the
authoritative output. In compact-only runs, a rejected Tesseract result is
correctly absent from authoritative output but is exactly the evidence needed
to replay candidate gate settings without OCRing the corpus again.

Keep development and test manifests document-disjoint. A calibration report
helps choose a frozen candidate on development data; it does not edit pipeline
defaults or claim that a small sample proves clinical safety.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from sttl.evaluate import (
    cer,
    character_error_stats,
    error_rate,
    wer,
    word_error_stats,
)
from sttl.jsonio import load_json_object_bytes
from sttl.pipeline import PipelineConfig, tesseract_quality

CORPUS_SCHEMA_VERSION = "sttl-gate-calibration-corpus/v1"
HUMAN_TRANSCRIPTION_SCHEMA_VERSION = "sttl-human-transcription/v1"
REPORT_SCHEMA_VERSION = "sttl-tesseract-gate-calibration/v1"
DEFAULT_CANDIDATE_ID = "current"
DEFAULT_CANDIDATE = (
    f"{DEFAULT_CANDIDATE_ID}:{PipelineConfig().min_mean_confidence:g}:"
    f"{PipelineConfig().min_confident_word_ratio:.2f}:"
    f"{PipelineConfig().confident_word_threshold:g}"
)
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GateCandidate:
    """The confidence controls intentionally swept by this calibration tool."""

    identifier: str
    min_mean_confidence: float
    min_confident_word_ratio: float
    confident_word_threshold: float


@dataclass(frozen=True)
class CalibrationPage:
    """Reviewed reference and one fixed raw Tesseract result for a page."""

    document_id: str
    page: int
    reference_text: str
    tesseract_text: str
    confidences: tuple[float, ...]
    expected_auto_accept: bool
    review_reasons: tuple[str, ...]


def load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    """Compatibility wrapper for the shared byte-exact JSON loader."""
    return load_json_object_bytes(path)


def relative_path(manifest_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def validated_page_map(pages: Any, context: str) -> dict[int, dict[str, Any]]:
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"{context}.pages must be a non-empty list")
    result: dict[int, dict[str, Any]] = {}
    for entry in pages:
        if not isinstance(entry, dict):
            raise ValueError(f"{context}.pages entries must be objects")
        page = entry.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError(f"{context}.pages[].page must be a positive integer")
        if page in result:
            raise ValueError(f"{context}.pages has duplicate page {page}")
        result[page] = entry
    return result


def reviewed_page(entry: dict[str, Any], context: str) -> tuple[str, bool, tuple[str, ...]]:
    text = entry.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{context}.text must be a string")
    review = entry.get("review")
    if not isinstance(review, dict):
        raise ValueError(f"{context}.review must be an object")
    if review.get("status") != "adjudicated":
        raise ValueError(f"{context}.review.status must be 'adjudicated'")
    auto_accept = review.get("auto_accept")
    if not isinstance(auto_accept, bool):
        raise ValueError(f"{context}.review.auto_accept must be a boolean")
    reasons = review.get("reasons", [])
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError(f"{context}.review.reasons must be a list of strings")
    return text, auto_accept, tuple(reasons)


def tesseract_evidence(entry: dict[str, Any], context: str) -> tuple[str, tuple[float, ...]]:
    layers = entry.get("layers")
    if not isinstance(layers, dict):
        raise ValueError(f"{context}.layers must be an object")
    layer = layers.get("tesseract5")
    if not isinstance(layer, dict) or layer.get("attempted") is not True:
        raise ValueError(
            f"{context} has no raw attempted Tesseract evidence; "
            "run the corpus through Tesseract before calibration"
        )
    text = layer.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{context}.layers.tesseract5.text must be a string")
    words = layer.get("words")
    if not isinstance(words, list):
        raise ValueError(f"{context}.layers.tesseract5.words must be a list")
    confidences: list[float] = []
    for word_index, word in enumerate(words, 1):
        if not isinstance(word, dict):
            raise ValueError(f"{context}.tesseract5.words[{word_index}] must be an object")
        confidence = word.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"{context}.tesseract5.words[{word_index}].confidence must be numeric")
        confidences.append(float(confidence))
    return text, tuple(confidences)


def load_calibration_pages(
    corpus_path: Path,
) -> tuple[dict[str, Any], str, list[CalibrationPage], list[dict[str, Any]]]:
    """Load and strictly align corpus, reviewed truth, raw OCR, and job hashes."""
    manifest, manifest_sha256 = load_json_bytes(corpus_path)
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"{corpus_path}.schema_version must be {CORPUS_SCHEMA_VERSION!r}")
    split = manifest.get("split")
    if split not in {"development", "test"}:
        raise ValueError(f"{corpus_path}.split must be 'development' or 'test'")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{corpus_path}.documents must be a non-empty list")

    page_samples: list[CalibrationPage] = []
    provenance: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for position, document in enumerate(documents, 1):
        context = f"{corpus_path}.documents[{position}]"
        if not isinstance(document, dict):
            raise ValueError(f"{context} must be an object")
        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{context}.id must be a non-empty string")
        if document_id in identifiers:
            raise ValueError(f"{corpus_path} has duplicate document id {document_id!r}")
        identifiers.add(document_id)

        ocr_path = relative_path(corpus_path, document.get("ocr_json"), f"{context}.ocr_json")
        truth_path = relative_path(
            corpus_path, document.get("human_truth_json"), f"{context}.human_truth_json"
        )
        ocr, ocr_sha256 = load_json_bytes(ocr_path)
        truth, _ = load_json_bytes(truth_path)

        if truth.get("schema_version") != HUMAN_TRANSCRIPTION_SCHEMA_VERSION:
            raise ValueError(
                f"{truth_path}.schema_version must be {HUMAN_TRANSCRIPTION_SCHEMA_VERSION!r}"
            )
        if truth.get("document_id") != document_id:
            raise ValueError(f"{truth_path}.document_id does not match manifest id {document_id!r}")
        source_sha256 = truth.get("source_sha256")
        if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
            raise ValueError(f"{truth_path}.source_sha256 must be a lowercase SHA-256 digest")
        evidence_sha256 = truth.get("tesseract_evidence_sha256")
        if not isinstance(evidence_sha256, str) or not SHA256_RE.fullmatch(evidence_sha256):
            raise ValueError(
                f"{truth_path}.tesseract_evidence_sha256 must be a lowercase SHA-256 digest"
            )
        if evidence_sha256 != ocr_sha256:
            raise ValueError(
                f"{truth_path} does not bind its reviewed Tesseract evidence to {ocr_path}"
            )

        job_path = ocr_path.parent / "job.json"
        job, _ = load_json_bytes(job_path)
        job_source = job.get("source")
        job_sha256 = job_source.get("content_sha256") if isinstance(job_source, dict) else None
        if job_sha256 != source_sha256:
            raise ValueError(
                f"{truth_path}.source_sha256 does not match {job_path} source.content_sha256"
            )

        ocr_pages = validated_page_map(ocr.get("pages"), str(ocr_path))
        truth_pages = validated_page_map(truth.get("pages"), str(truth_path))
        if set(ocr_pages) != set(truth_pages):
            raise ValueError(
                f"{truth_path} page IDs do not exactly match raw OCR evidence in {ocr_path}"
            )

        for page in sorted(ocr_pages):
            reference_text, auto_accept, reasons = reviewed_page(
                truth_pages[page], f"{truth_path}.pages[{page}]"
            )
            tesseract_text, confidences = tesseract_evidence(
                ocr_pages[page], f"{ocr_path}.pages[{page}]"
            )
            page_samples.append(
                CalibrationPage(
                    document_id=document_id,
                    page=page,
                    reference_text=reference_text,
                    tesseract_text=tesseract_text,
                    confidences=confidences,
                    expected_auto_accept=auto_accept,
                    review_reasons=reasons,
                )
            )

        provenance.append(
            {
                "id": document_id,
                "source_sha256": source_sha256,
                "tesseract_evidence_sha256": ocr_sha256,
                "page_ids": sorted(ocr_pages),
            }
        )
    return manifest, manifest_sha256, page_samples, provenance


def parse_candidate(value: str) -> GateCandidate:
    """Parse id:min-mean-confidence:min-confident-ratio:word-threshold."""
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--candidate must be id:min_mean_confidence:min_confident_word_ratio:"
            "confident_word_threshold"
        )
    identifier, mean, ratio, threshold = parts
    if not CANDIDATE_ID_RE.fullmatch(identifier):
        raise argparse.ArgumentTypeError(
            "candidate id may contain letters, numbers, '.', '_' and '-' only"
        )
    try:
        candidate = GateCandidate(
            identifier=identifier,
            min_mean_confidence=float(mean),
            min_confident_word_ratio=float(ratio),
            confident_word_threshold=float(threshold),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"candidate values must be numeric: {value!r}") from exc
    if not 0 <= candidate.min_mean_confidence <= 100:
        raise argparse.ArgumentTypeError("candidate min_mean_confidence must be between 0 and 100")
    if not 0 <= candidate.min_confident_word_ratio <= 1:
        raise argparse.ArgumentTypeError(
            "candidate min_confident_word_ratio must be between 0 and 1"
        )
    if not 0 <= candidate.confident_word_threshold <= 100:
        raise argparse.ArgumentTypeError(
            "candidate confident_word_threshold must be between 0 and 100"
        )
    return candidate


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def fixed_quality_config(args: argparse.Namespace, candidate: GateCandidate) -> PipelineConfig:
    """Build the exact gate semantics to replay from one candidate."""
    return PipelineConfig(
        min_tesseract_chars=args.min_tesseract_chars,
        min_tesseract_words=args.min_tesseract_words,
        min_mean_confidence=candidate.min_mean_confidence,
        min_confident_word_ratio=candidate.min_confident_word_ratio,
        confident_word_threshold=candidate.confident_word_threshold,
        max_tesseract_garbage_ratio=args.max_tesseract_garbage_ratio,
        min_plausible_word_ratio=args.min_plausible_word_ratio,
    )


def aggregate_accuracy(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate error evidence without rounding before the final report."""
    rows = list(results)
    char_distance = char_reference_length = char_hypothesis_length = 0
    word_distance = word_reference_length = word_hypothesis_length = 0
    page_cers: list[float] = []
    page_wers: list[float] = []
    for row in rows:
        reference = row["reference_text"]
        hypothesis = row["tesseract_text"]
        distance, reference_length, hypothesis_length = character_error_stats(reference, hypothesis)
        char_distance += distance
        char_reference_length += reference_length
        char_hypothesis_length += hypothesis_length
        distance, reference_length, hypothesis_length = word_error_stats(reference, hypothesis)
        word_distance += distance
        word_reference_length += reference_length
        word_hypothesis_length += hypothesis_length
        page_cers.append(cer(reference, hypothesis))
        page_wers.append(wer(reference, hypothesis))
    return {
        "pages": len(rows),
        "weighted_CER": round(
            error_rate(char_distance, char_reference_length, char_hypothesis_length), 6
        )
        if rows
        else None,
        "weighted_WER": round(
            error_rate(word_distance, word_reference_length, word_hypothesis_length), 6
        )
        if rows
        else None,
        "mean_page_CER": round(fmean(page_cers), 6) if page_cers else None,
        "mean_page_WER": round(fmean(page_wers), 6) if page_wers else None,
        "reference_characters": char_reference_length,
        "hypothesis_characters": char_hypothesis_length,
        "reference_words": word_reference_length,
        "hypothesis_words": word_hypothesis_length,
    }


def evaluate_candidate(
    candidate: GateCandidate,
    pages: list[CalibrationPage],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = fixed_quality_config(args, candidate)
    page_results: list[dict[str, Any]] = []
    safe_accepted = unsafe_accepted = safe_rejected = unsafe_rejected = 0
    for sample in pages:
        quality = tesseract_quality(sample.tesseract_text, sample.confidences, config)
        accepted = bool(quality["accepted"])
        if sample.expected_auto_accept and accepted:
            outcome = "safe_accepted"
            safe_accepted += 1
        elif not sample.expected_auto_accept and accepted:
            outcome = "unsafe_accepted_false_accept"
            unsafe_accepted += 1
        elif sample.expected_auto_accept:
            outcome = "safe_rejected_false_reject"
            safe_rejected += 1
        else:
            outcome = "unsafe_rejected"
            unsafe_rejected += 1
        page_results.append(
            {
                "document_id": sample.document_id,
                "page": sample.page,
                "expected_auto_accept": sample.expected_auto_accept,
                "gate_accepted": accepted,
                "outcome": outcome,
                "review_reasons": list(sample.review_reasons),
                "CER": round(cer(sample.reference_text, sample.tesseract_text), 6),
                "WER": round(wer(sample.reference_text, sample.tesseract_text), 6),
                "quality": quality,
                # Keep source text out of this report. The reviewed transcript
                # and raw OCR remain in their separately access-controlled files.
                "reference_text": sample.reference_text,
                "tesseract_text": sample.tesseract_text,
            }
        )

    # Text is needed only locally to calculate aggregates. Do not retain it in
    # the persisted calibration report, where it would duplicate sensitive OCR.
    all_accuracy = aggregate_accuracy(page_results)
    accepted_accuracy = aggregate_accuracy(row for row in page_results if row["gate_accepted"])
    for row in page_results:
        row.pop("reference_text")
        row.pop("tesseract_text")

    total = len(page_results)
    safe_total = safe_accepted + safe_rejected
    unsafe_total = unsafe_accepted + unsafe_rejected
    accepted_total = safe_accepted + unsafe_accepted
    summary = {
        "pages": total,
        "confusion_matrix": {
            "safe_accepted": safe_accepted,
            "unsafe_accepted_false_accepts": unsafe_accepted,
            "safe_rejected_false_rejects": safe_rejected,
            "unsafe_rejected": unsafe_rejected,
        },
        "rates": {
            "acceptance_rate": rate(accepted_total, total),
            "unsafe_escape_rate": rate(unsafe_accepted, unsafe_total),
            "unsafe_among_accepted": rate(unsafe_accepted, accepted_total),
            "safe_rejection_rate": rate(safe_rejected, safe_total),
        },
        "all_pages_accuracy": all_accuracy,
        "accepted_only_accuracy": accepted_accuracy,
    }
    return {
        "candidate": asdict(candidate),
        "fixed_gate_settings": {
            "min_tesseract_chars": args.min_tesseract_chars,
            "min_tesseract_words": args.min_tesseract_words,
            "max_tesseract_garbage_ratio": args.max_tesseract_garbage_ratio,
            "min_plausible_word_ratio": args.min_plausible_word_ratio,
        },
        "summary": summary,
        "page_results": page_results,
    }


def mark_pareto_frontier(candidates: list[dict[str, Any]]) -> None:
    """Mark candidates not dominated on false accepts and coverage."""
    for candidate in candidates:
        current = candidate["summary"]["confusion_matrix"]
        current_false_accepts = current["unsafe_accepted_false_accepts"]
        current_accepted = current["safe_accepted"] + current_false_accepts
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            comparison = other["summary"]["confusion_matrix"]
            other_false_accepts = comparison["unsafe_accepted_false_accepts"]
            other_accepted = comparison["safe_accepted"] + other_false_accepts
            if (
                other_false_accepts <= current_false_accepts
                and other_accepted >= current_accepted
                and (
                    other_false_accepts < current_false_accepts or other_accepted > current_accepted
                )
            ):
                dominated = True
                break
        candidate["summary"]["pareto_optimal"] = not dominated


def selection_summary(
    candidates: list[dict[str, Any]], max_false_accepts: int | None
) -> dict[str, Any]:
    """Report a predeclared development constraint without mutating defaults."""
    eligible = [
        candidate
        for candidate in candidates
        if max_false_accepts is None
        or candidate["summary"]["confusion_matrix"]["unsafe_accepted_false_accepts"]
        <= max_false_accepts
    ]
    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -(
                candidate["summary"]["confusion_matrix"]["safe_accepted"]
                + candidate["summary"]["confusion_matrix"]["unsafe_accepted_false_accepts"]
            ),
            candidate["summary"]["confusion_matrix"]["unsafe_accepted_false_accepts"],
            candidate["candidate"]["identifier"],
        ),
    )
    return {
        "max_false_accepts": max_false_accepts,
        "eligible_candidate_ids": [candidate["candidate"]["identifier"] for candidate in eligible],
        "highest_coverage_eligible_candidate": (
            ranked[0]["candidate"]["identifier"] if ranked else None
        ),
        "note": (
            "Development-only guidance. Freeze one candidate before evaluating "
            "it once on a document-disjoint test corpus; this tool never edits "
            "production pipeline defaults."
        ),
    }


def flat_candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = candidate["summary"]
    matrix = summary["confusion_matrix"]
    rates = summary["rates"]
    accuracy = summary["accepted_only_accuracy"]
    return {
        "candidate": candidate["candidate"]["identifier"],
        "min_mean_confidence": candidate["candidate"]["min_mean_confidence"],
        "min_confident_word_ratio": candidate["candidate"]["min_confident_word_ratio"],
        "confident_word_threshold": candidate["candidate"]["confident_word_threshold"],
        "pages": summary["pages"],
        **matrix,
        **rates,
        "accepted_only_weighted_CER": accuracy["weighted_CER"],
        "accepted_only_weighted_WER": accuracy["weighted_WER"],
        "pareto_optimal": summary["pareto_optimal"],
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tesseract_gate_calibration.json"
    csv_path = output_dir / "tesseract_gate_candidates.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [flat_candidate_row(candidate) for candidate in report["candidates"]]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay reviewed raw Tesseract pages against candidate confidence "
            "gates. It is for development calibration, not a production switch."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Development or held-out test manifest using sttl-gate-calibration-corpus/v1.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        metavar="ID:MEAN:RATIO:WORD",
        help=(
            "Candidate confidence gate. Repeat to compare several. Defaults to "
            f"{DEFAULT_CANDIDATE!r} when omitted."
        ),
    )
    parser.add_argument(
        "--max-false-accepts",
        type=int,
        default=None,
        help="Predeclared maximum unsafe accepted pages for development ranking.",
    )
    parser.add_argument(
        "--min-tesseract-chars",
        type=int,
        default=PipelineConfig.min_tesseract_chars,
    )
    parser.add_argument(
        "--min-tesseract-words",
        type=int,
        default=PipelineConfig.min_tesseract_words,
    )
    parser.add_argument(
        "--max-tesseract-garbage-ratio",
        type=float,
        default=PipelineConfig.max_tesseract_garbage_ratio,
    )
    parser.add_argument(
        "--min-plausible-word-ratio",
        type=float,
        default=PipelineConfig.min_plausible_word_ratio,
    )
    return parser


def validate_args(args: argparse.Namespace) -> list[GateCandidate]:
    if args.max_false_accepts is not None and args.max_false_accepts < 0:
        raise ValueError("--max-false-accepts must be zero or greater")
    if args.min_tesseract_chars < 0 or args.min_tesseract_words < 0:
        raise ValueError("minimum character and word counts must be zero or greater")
    for name in ("max_tesseract_garbage_ratio", "min_plausible_word_ratio"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    candidates = args.candidate or [parse_candidate(DEFAULT_CANDIDATE)]
    identifiers = [candidate.identifier for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("--candidate identifiers must be unique")
    return candidates


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    candidates = validate_args(args)
    corpus, corpus_sha256, pages, provenance = load_calibration_pages(args.corpus)
    evaluated = [evaluate_candidate(candidate, pages, args) for candidate in candidates]
    mark_pareto_frontier(evaluated)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "corpus": {
            "schema_version": corpus["schema_version"],
            "split": corpus["split"],
            "manifest_sha256": corpus_sha256,
            "documents": provenance,
            "pages": len(pages),
        },
        "selection_policy": selection_summary(evaluated, args.max_false_accepts),
        "candidates": evaluated,
    }
    json_path, csv_path = write_report(args.out_dir, report)
    print(
        json.dumps(
            {
                "report": str(json_path),
                "candidate_table": str(csv_path),
                "split": corpus["split"],
                "pages": len(pages),
                "candidates": [
                    {
                        "id": candidate["candidate"]["identifier"],
                        **candidate["summary"]["confusion_matrix"],
                    }
                    for candidate in evaluated
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def cli(argv: list[str] | None = None) -> int:
    """Run the report-producing calibration command with a shell exit status."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
