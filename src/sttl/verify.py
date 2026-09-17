#!/usr/bin/env python3
"""Verify that an STTL OCR job's manifest and published artifacts agree.

This is an integrity and provenance verifier, not an OCR-accuracy scorer. It
checks the last completed manifest record for each page, the document summary,
normalized/rich exports, combined text, routing contracts, and retained Surya
raw-batch evidence. Use ``evaluate_routing.py`` with human-reviewed truth to
measure accuracy after this verifier succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from sttl.pipeline import PAGE_STATES, PipelineConfig, primary_ocr_engine

VERIFIER_SCHEMA_VERSION = "sttl-ocr-artifact-verification/v1"


def blank_report(job_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "job_dir": str(job_dir),
        "valid": False,
        "errors": [],
        "warnings": [],
        "checks": {},
    }


def issue(
    report: dict[str, Any],
    severity: str,
    code: str,
    message: str,
    location: str | None = None,
) -> None:
    entry: dict[str, Any] = {"code": code, "message": message}
    if location is not None:
        entry["location"] = location
    report["errors" if severity == "error" else "warnings"].append(entry)


def read_json(path: Path, report: dict[str, Any], label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issue(report, "error", "missing_file", f"Required {label} is missing", str(path))
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issue(report, "error", "invalid_json", f"Could not read {label}: {exc}", str(path))
        return None
    if not isinstance(value, dict):
        issue(report, "error", "invalid_json_object", f"{label} must be a JSON object", str(path))
        return None
    return value


def effective_manifest(path: Path, report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return the last completed record for each page, preserving reprocess history."""
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        issue(report, "error", "missing_file", "Required page manifest is missing", str(path))
        return result
    records = superseded = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        issue(
            report, "error", "invalid_manifest", f"Could not read page manifest: {exc}", str(path)
        )
        return result
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issue(report, "error", "invalid_manifest_json", str(exc), f"{path}:{line_number}")
            continue
        if not isinstance(record, dict):
            issue(
                report,
                "error",
                "invalid_manifest_record",
                "Record must be an object",
                f"{path}:{line_number}",
            )
            continue
        if record.get("status") != "complete":
            issue(
                report,
                "warning",
                "ignored_noncomplete_record",
                "Ignoring non-complete manifest record",
                f"{path}:{line_number}",
            )
            continue
        page = record.get("page")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            issue(
                report,
                "error",
                "invalid_manifest_page",
                "Completed record has no positive integer page",
                f"{path}:{line_number}",
            )
            continue
        records += 1
        if page in result:
            superseded += 1
        result[page] = record
    report["checks"]["manifest"] = {
        "records": records,
        "effective_completed_pages": len(result),
        "superseded_completed_records": superseded,
    }
    return result


def resolved_config(config: Any, report: dict[str, Any]) -> PipelineConfig | None:
    if not isinstance(config, dict):
        issue(report, "error", "invalid_config", "job.config must be an object", "job.json")
        return None
    defaults = asdict(PipelineConfig())
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        issue(
            report,
            "warning",
            "unknown_config_fields",
            f"Ignoring unknown config fields: {', '.join(unknown)}",
            "job.json",
        )
    merged = {key: config.get(key, default) for key, default in defaults.items()}
    try:
        return PipelineConfig(**merged)
    except (TypeError, ValueError) as exc:
        issue(
            report,
            "error",
            "invalid_config",
            f"Cannot construct pipeline config: {exc}",
            "job.json",
        )
        return None


def expected_summary(records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(str(record.get("classification")) for record in records.values())
    outcomes = Counter(str(record.get("outcome")) for record in records.values())
    quality_escalations = sum(
        record.get("outcome") == "surya_escalated"
        and record.get("escalation_reason") != "structure"
        for record in records.values()
    )
    structure_escalations = sum(
        record.get("outcome") == "surya_escalated"
        and record.get("escalation_reason") == "structure"
        for record in records.values()
    )
    hybrid_pages = sum(
        isinstance(record.get("hybrid_text"), dict) and bool(record["hybrid_text"].get("applied"))
        for record in records.values()
    )
    confidences = [
        float(record["tesseract_quality"]["mean_word_confidence"])
        for record in records.values()
        if isinstance(record.get("tesseract_quality"), dict)
        and isinstance(record["tesseract_quality"].get("mean_word_confidence"), (int, float))
        and not isinstance(record["tesseract_quality"].get("mean_word_confidence"), bool)
    ]
    return {
        # ``summarise_document`` deliberately preserves zero-valued states so
        # dashboards can compare runs without adding missing keys.  Mirror
        # that canonical shape rather than comparing a sparse Counter.
        "classification_counts": {state: classifications.get(state, 0) for state in PAGE_STATES},
        "outcome_counts": dict(outcomes),
        "native_pages": outcomes["native_text_accepted"],
        "tesseract_accepted_pages": outcomes["tesseract_accepted"],
        "tesseract_rejected_no_fallback_pages": outcomes["tesseract_rejected_no_fallback"],
        "surya_primary_pages": outcomes["surya_primary"],
        "surya_escalated_pages": outcomes["surya_escalated"],
        "surya_quality_escalated_pages": quality_escalations,
        "surya_structure_escalated_pages": structure_escalations,
        "surya_hybrid_text_pages": hybrid_pages,
        "mean_tesseract_confidence": round(fmean(confidences), 3) if confidences else None,
    }


def finite_bbox(value: Any, page_bbox: Any) -> bool:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and isinstance(page_bbox, list)
        and len(page_bbox) == 4
    ):
        return False
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
        px0, py0, px1, py1 = (float(item) for item in page_bbox)
    except (TypeError, ValueError):
        return False
    tolerance = 0.01
    return (
        all(math.isfinite(item) for item in (x0, y0, x1, y1, px0, py0, px1, py1))
        and x1 >= x0
        and y1 >= y0
        and x0 >= px0 - tolerance
        and y0 >= py0 - tolerance
        and x1 <= px1 + tolerance
        and y1 <= py1 + tolerance
    )


def verify_blocks(
    blocks: Any,
    page_bbox: Any,
    report: dict[str, Any],
    location: str,
) -> None:
    if not isinstance(blocks, list):
        issue(report, "error", "invalid_blocks", "blocks must be a list", location)
        return
    orders: list[int] = []
    for index, block in enumerate(blocks, 1):
        block_location = f"{location}.blocks[{index}]"
        if not isinstance(block, dict):
            issue(report, "error", "invalid_block", "Block must be an object", block_location)
            continue
        order = block.get("reading_order")
        if isinstance(order, bool) or not isinstance(order, int):
            issue(
                report,
                "error",
                "invalid_block_order",
                "Block reading_order must be an integer",
                block_location,
            )
        else:
            orders.append(order)
        if "bbox" in block and not finite_bbox(block["bbox"], page_bbox):
            issue(
                report,
                "error",
                "invalid_block_bbox",
                "Block bbox must be finite and within the page bbox",
                block_location,
            )
        if block.get("bbox") is not None and block.get("coordinate_space") != "pdf_points":
            issue(
                report,
                "error",
                "invalid_block_coordinate_space",
                "Block bbox must be in pdf_points",
                block_location,
            )
    if orders and orders != list(range(1, len(orders) + 1)):
        issue(
            report,
            "error",
            "noncontiguous_block_order",
            "Block reading_order must be contiguous from 1",
            location,
        )


def verify_record_contract(
    record: dict[str, Any],
    config: PipelineConfig | None,
    report: dict[str, Any],
    location: str,
) -> None:
    """Check the route fields that define one selected outcome.

    This validates persisted routing/provenance consistency, not OCR accuracy.
    Raw-prediction and export checks are deliberately separate so failures can
    report which evidence boundary was broken.
    """
    outcome = record.get("outcome")
    engine = record.get("engine")
    route = record.get("route")
    quality = record.get("tesseract_quality")
    candidate = record.get("tesseract_candidate")
    if outcome == "native_text_accepted":
        if engine != "pymupdf" or route != "native_text":
            issue(
                report,
                "error",
                "native_route_contract",
                "Native outcome requires pymupdf/native_text",
                location,
            )
    elif outcome == "tesseract_accepted":
        if engine != "tesseract5" or route != "tesseract":
            issue(
                report,
                "error",
                "tesseract_route_contract",
                "Accepted Tesseract outcome requires tesseract5/tesseract",
                location,
            )
        if not isinstance(quality, dict) or quality.get("accepted") is not True:
            issue(
                report,
                "error",
                "tesseract_quality_contract",
                "Accepted Tesseract page needs accepted quality evidence",
                location,
            )
    elif outcome == "tesseract_rejected_no_fallback":
        if engine != "none" or route != "tesseract":
            issue(
                report,
                "error",
                "compact_route_contract",
                "Compact rejection requires none/tesseract",
                location,
            )
        if not isinstance(candidate, dict) or record.get("text") != "":
            issue(
                report,
                "error",
                "compact_evidence_contract",
                "Compact rejection needs raw Tesseract evidence and empty authoritative text",
                location,
            )
        if config is not None and config.fallback_engine != "none":
            issue(
                report,
                "error",
                "compact_config_contract",
                "Compact rejection requires fallback_engine='none'",
                location,
            )
    elif outcome == "surya_primary":
        if engine != "surya" or route != "surya":
            issue(
                report,
                "error",
                "surya_primary_route_contract",
                "Surya primary requires surya/surya",
                location,
            )
        if (
            record.get("classifier_route") != "tesseract"
            or record.get("escalation_reason") != "primary"
        ):
            issue(
                report,
                "error",
                "surya_primary_provenance_contract",
                "Surya primary needs classifier_route='tesseract' and escalation_reason='primary'",
                location,
            )
        if quality is not None or candidate is not None:
            issue(
                report,
                "error",
                "surya_primary_evidence_contract",
                "Surya primary must not contain Tesseract evidence",
                location,
            )
        if not isinstance(record.get("surya_batch"), int):
            issue(
                report,
                "error",
                "surya_batch_contract",
                "Surya primary requires a numeric Surya batch",
                location,
            )
        if config is not None and primary_ocr_engine(config) != "surya":
            issue(
                report,
                "error",
                "surya_primary_config_contract",
                "Surya primary requires resolved primary engine Surya",
                location,
            )
    elif outcome == "surya_escalated":
        if engine != "surya" or route != "tesseract":
            issue(
                report,
                "error",
                "surya_fallback_route_contract",
                "Surya fallback requires surya/tesseract",
                location,
            )
        if not isinstance(quality, dict) or not isinstance(candidate, dict):
            issue(
                report,
                "error",
                "surya_fallback_evidence_contract",
                "Surya fallback requires Tesseract quality and candidate evidence",
                location,
            )
        if record.get("escalation_reason") not in {"quality", "structure"}:
            issue(
                report,
                "error",
                "surya_fallback_reason_contract",
                "Surya fallback requires quality or structure escalation",
                location,
            )
        if not isinstance(record.get("surya_batch"), int):
            issue(
                report,
                "error",
                "surya_batch_contract",
                "Surya fallback requires a numeric Surya batch",
                location,
            )
        if config is not None and config.fallback_engine != "surya":
            issue(
                report,
                "error",
                "surya_fallback_config_contract",
                "Surya fallback requires fallback_engine='surya'",
                location,
            )
    else:
        issue(report, "error", "unknown_outcome", f"Unknown outcome {outcome!r}", location)


def page_map(pages: Any, report: dict[str, Any], location: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(pages, list):
        issue(report, "error", "invalid_pages", "pages must be a list", location)
        return result
    for index, page in enumerate(pages, 1):
        item_location = f"{location}[{index}]"
        if not isinstance(page, dict):
            issue(report, "error", "invalid_page", "Page must be an object", item_location)
            continue
        number = page.get("page")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            issue(
                report,
                "error",
                "invalid_page_number",
                "Page number must be a positive integer",
                item_location,
            )
            continue
        if number in result:
            issue(
                report, "error", "duplicate_page_number", f"Duplicate page {number}", item_location
            )
            continue
        result[number] = page
    if list(result) != sorted(result):
        issue(
            report,
            "error",
            "nonascending_pages",
            "Page array must be ascending by page number",
            location,
        )
    return result


def verify_surya_raw_batches(
    job_dir: Path,
    records: dict[int, dict[str, Any]],
    report: dict[str, Any],
    *,
    required: bool,
) -> None:
    for page, record in records.items():
        if record.get("engine") != "surya":
            continue
        batch = record.get("surya_batch")
        if not isinstance(batch, int):
            continue
        results_files = list((job_dir / "surya" / f"batch-{batch:06d}").glob("**/results.json"))
        if not results_files:
            issue(
                report,
                "error" if required else "warning",
                "missing_surya_raw_batch",
                f"No raw Surya results.json found for batch {batch}",
                f"pages.jsonl page {page}",
            )
            continue
        found_page = False
        for results_path in results_files:
            raw = read_json(results_path, report, "Surya results")
            if raw is None:
                continue
            expected = f"page-{page:06d}"
            if any(Path(str(key)).stem == expected for key in raw):
                found_page = True
                break
        if not found_page:
            issue(
                report,
                "error" if required else "warning",
                "missing_surya_raw_page",
                f"Surya batch {batch} contains no raw prediction for page {page}",
                f"pages.jsonl page {page}",
            )


def verify_published_artifact(
    artifact: dict[str, Any],
    kind: str,
    records: dict[int, dict[str, Any]],
    document: dict[str, Any],
    expected_pages: list[int],
    config: PipelineConfig | None,
    report: dict[str, Any],
) -> None:
    expected_schema = "cascade-ocr/rich-v1" if kind == "rich" else "cascade-ocr/v1"
    if artifact.get("schema_version") != expected_schema:
        issue(
            report,
            "error",
            "artifact_schema",
            f"{kind} artifact has unexpected schema version",
            kind,
        )
    if artifact.get("engine") != "cascade":
        issue(report, "error", "artifact_engine", f"{kind} artifact engine must be cascade", kind)
    if artifact.get("source") != document.get("source"):
        issue(
            report,
            "error",
            "artifact_source_mismatch",
            f"{kind} artifact source does not match document",
            kind,
        )
    if kind == "rich" and artifact.get("coordinate_space") != "pdf_points":
        issue(
            report,
            "error",
            "artifact_coordinate_space",
            "rich artifact coordinate_space must be pdf_points",
            kind,
        )
    if artifact.get("page_count") != len(expected_pages):
        issue(
            report,
            "error",
            "artifact_page_count",
            f"{kind} artifact page_count does not match document",
            kind,
        )
    pages = page_map(artifact.get("pages"), report, f"{kind}.pages")
    if sorted(pages) != expected_pages:
        issue(
            report,
            "error",
            "artifact_page_coverage",
            f"{kind} artifact page IDs do not match document",
            kind,
        )
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        issue(report, "error", "artifact_metadata", f"{kind} artifact lacks metadata", kind)
    else:
        if metadata.get("pipeline_version") != document.get("pipeline_version"):
            issue(
                report,
                "error",
                "artifact_pipeline_version_mismatch",
                f"{kind} artifact pipeline version does not match document",
                kind,
            )
        routing = metadata.get("routing")
        if not isinstance(routing, dict):
            issue(
                report,
                "error",
                "artifact_routing_metadata",
                f"{kind} artifact lacks routing metadata",
                kind,
            )
        else:
            if config is not None:
                expected_primary = primary_ocr_engine(config)
                if routing.get("primary_ocr_engine") != expected_primary:
                    issue(
                        report,
                        "error",
                        "artifact_primary_engine_mismatch",
                        f"{kind} routing primary engine does not match job config",
                        kind,
                    )
                if routing.get("fallback_engine") != config.fallback_engine:
                    issue(
                        report,
                        "error",
                        "artifact_fallback_engine_mismatch",
                        f"{kind} routing fallback engine does not match job config",
                        kind,
                    )
            for key in (
                "classification_counts",
                "outcome_counts",
                "tesseract_rejected_no_fallback_pages",
                "surya_primary_pages",
                "surya_quality_escalated_pages",
                "surya_structure_escalated_pages",
                "surya_hybrid_text_pages",
            ):
                if routing.get(key) != document.get(key):
                    issue(
                        report,
                        "error",
                        "artifact_summary_mismatch",
                        f"{kind} routing.{key} does not match document summary",
                        kind,
                    )
    for page_number, record in records.items():
        page = pages.get(page_number)
        if page is None:
            continue
        if not finite_bbox(page.get("bbox"), page.get("bbox")):
            issue(
                report,
                "error",
                "invalid_page_bbox",
                "Page bbox must be a finite ordered rectangle",
                f"{kind}.pages[{page_number}]",
            )
        if page.get("coordinate_space") != "pdf_points":
            issue(
                report,
                "error",
                "invalid_page_coordinate_space",
                "Page coordinate_space must be pdf_points",
                f"{kind}.pages[{page_number}]",
            )
        if kind == "rich":
            authoritative = page.get("authoritative") if isinstance(page, dict) else None
            routing = page.get("routing") if isinstance(page, dict) else None
            if not isinstance(authoritative, dict) or not isinstance(routing, dict):
                issue(
                    report,
                    "error",
                    "rich_page_provenance",
                    "Rich page lacks authoritative or routing object",
                    f"rich.pages[{page_number}]",
                )
                continue
            comparisons = {
                "authoritative.engine": (authoritative.get("engine"), record.get("engine")),
                "authoritative.outcome": (authoritative.get("outcome"), record.get("outcome")),
                "authoritative.text": (authoritative.get("text"), record.get("text")),
                "routing.route": (routing.get("route"), record.get("route")),
            }
            if record.get("classifier_route") is not None:
                comparisons["routing.classifier_route"] = (
                    routing.get("classifier_route"),
                    record.get("classifier_route"),
                )
            for key, (actual, expected) in comparisons.items():
                if actual != expected:
                    issue(
                        report,
                        "error",
                        "rich_manifest_mismatch",
                        f"{key} does not match manifest",
                        f"rich.pages[{page_number}]",
                    )
            verify_blocks(
                authoritative.get("blocks"),
                page.get("bbox"),
                report,
                f"rich.pages[{page_number}].authoritative",
            )
        else:
            metadata_page = page.get("metadata") if isinstance(page, dict) else None
            if not isinstance(metadata_page, dict):
                issue(
                    report,
                    "error",
                    "normalized_page_metadata",
                    "Normalized page lacks metadata",
                    f"cascade.pages[{page_number}]",
                )
                continue
            comparisons = {
                "metadata.engine": (metadata_page.get("engine"), record.get("engine")),
                "metadata.outcome": (metadata_page.get("outcome"), record.get("outcome")),
                "metadata.route": (metadata_page.get("route"), record.get("route")),
                "text": (page.get("text"), record.get("text")),
            }
            if record.get("classifier_route") is not None:
                comparisons["metadata.classifier_route"] = (
                    metadata_page.get("classifier_route"),
                    record.get("classifier_route"),
                )
            for key, (actual, expected) in comparisons.items():
                if actual != expected:
                    issue(
                        report,
                        "error",
                        "cascade_manifest_mismatch",
                        f"{key} does not match manifest",
                        f"cascade.pages[{page_number}]",
                    )
            verify_blocks(
                page.get("blocks"), page.get("bbox"), report, f"cascade.pages[{page_number}]"
            )


def verify_combined_text(
    path: Path, records: dict[int, dict[str, Any]], report: dict[str, Any]
) -> None:
    expected_lines: list[str] = []
    for page in sorted(records):
        expected_lines.extend(
            [f"--- Page {page} ---", str(records[page].get("text", "")).strip(), ""]
        )
    expected = "\n".join(expected_lines)
    try:
        actual = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issue(report, "error", "missing_file", "Required combined text is missing", str(path))
        return
    except (OSError, UnicodeDecodeError) as exc:
        issue(report, "error", "invalid_combined_text", str(exc), str(path))
        return
    if actual != expected:
        issue(
            report,
            "error",
            "combined_text_mismatch",
            "combined.txt does not match the effective manifest",
            str(path),
        )


def verify_artifact(
    job_dir: str | Path,
    *,
    require_complete: bool = True,
    require_raw_surya: bool = True,
) -> dict[str, Any]:
    """Return a structured integrity report; malformed artifacts never raise."""
    directory = Path(job_dir)
    report = blank_report(directory)
    job = read_json(directory / "job.json", report, "job metadata")
    document = read_json(directory / "document.json", report, "document summary")
    records = effective_manifest(directory / "pages.jsonl", report)
    if job is None or document is None:
        report["valid"] = not report["errors"]
        return report

    config = resolved_config(job.get("config"), report)
    source = job.get("source")
    if not isinstance(source, dict):
        issue(report, "error", "job_source", "job.source must be an object", "job.json")
    else:
        for key in ("file_name", "content_sha256"):
            if document.get("source" if key == "file_name" else "source_sha256") != source.get(key):
                issue(
                    report,
                    "error",
                    "source_identity_mismatch",
                    f"document summary does not match job source {key}",
                    "document.json",
                )
    if document.get("pipeline_version") != job.get("pipeline_version"):
        issue(
            report,
            "error",
            "pipeline_version_mismatch",
            "document and job pipeline versions differ",
            "document.json",
        )
    page_count = document.get("source_page_count")
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
        issue(
            report,
            "error",
            "invalid_page_count",
            "document source_page_count must be positive",
            "document.json",
        )
        page_count = 0
    expected_pages = list(range(1, page_count + 1))
    if sorted(records) != expected_pages:
        issue(
            report,
            "error" if require_complete else "warning",
            "manifest_page_coverage",
            "Effective completed manifest pages do not match source pages",
            "pages.jsonl",
        )
    if require_complete and document.get("state") != "complete":
        issue(
            report,
            "error",
            "document_not_complete",
            "Document summary is not complete",
            "document.json",
        )
    for page, record in records.items():
        if record.get("classification") not in PAGE_STATES:
            issue(
                report,
                "error",
                "unknown_classification",
                f"Unknown page classification {record.get('classification')!r}",
                f"pages.jsonl page {page}",
            )
        verify_record_contract(record, config, report, f"pages.jsonl page {page}")
        if isinstance(record.get("blocks"), list):
            raster = record.get("raster")
            page_bbox = raster.get("pdf_bbox") if isinstance(raster, dict) else None
            verify_blocks(record["blocks"], page_bbox, report, f"pages.jsonl page {page}")

    summary = expected_summary(records)
    for key, expected in summary.items():
        actual = document.get(key)
        if (
            key == "mean_tesseract_confidence"
            and isinstance(expected, float)
            and isinstance(actual, (int, float))
        ):
            matches = abs(float(actual) - expected) < 0.001
        else:
            matches = actual == expected
        if not matches:
            issue(
                report,
                "error",
                "summary_mismatch",
                f"document.{key} does not match effective manifest",
                "document.json",
            )

    expected_state = "complete" if len(records) == page_count and page_count else "incomplete"
    for key, expected in {
        "completed_pages": len(records),
        "pending_pages": max(0, page_count - len(records)),
        "state": expected_state,
    }.items():
        if document.get(key) != expected:
            issue(
                report,
                "error",
                "document_progress_mismatch",
                f"document.{key} does not match effective manifest coverage",
                "document.json",
            )

    if document.get("state") == "complete" and sorted(records) == expected_pages:
        verify_combined_text(directory / "combined.txt", records, report)
        normalized_name = document.get("normalized_output")
        rich_name = document.get("rich_output")
        if not isinstance(normalized_name, str) or not normalized_name:
            issue(
                report,
                "error",
                "missing_output_name",
                "document.normalized_output is required",
                "document.json",
            )
        else:
            artifact = read_json(directory / normalized_name, report, "normalized artifact")
            if artifact is not None:
                verify_published_artifact(
                    artifact, "cascade", records, document, expected_pages, config, report
                )
        if not isinstance(rich_name, str) or not rich_name:
            issue(
                report,
                "error",
                "missing_output_name",
                "document.rich_output is required",
                "document.json",
            )
        else:
            artifact = read_json(directory / rich_name, report, "rich artifact")
            if artifact is not None:
                verify_published_artifact(
                    artifact, "rich", records, document, expected_pages, config, report
                )
    elif not require_complete:
        issue(
            report,
            "warning",
            "skipped_published_outputs",
            "Skipping published-output checks for incomplete artifact",
            str(directory),
        )

    verify_surya_raw_batches(directory, records, report, required=require_raw_surya)
    report["checks"]["effective_summary"] = summary
    report["valid"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="One STTL document output directory")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report incomplete page coverage as a warning",
    )
    parser.add_argument(
        "--allow-missing-raw-surya",
        action="store_true",
        help="Report missing raw Surya batch files as warnings",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path for the verification report")
    args = parser.parse_args()
    report = verify_artifact(
        args.job_dir,
        require_complete=not args.allow_incomplete,
        require_raw_surya=not args.allow_missing_raw_surya,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
