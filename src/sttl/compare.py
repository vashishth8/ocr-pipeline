#!/usr/bin/env python3
"""Compare saved cascade (or legacy Surya) and Chandra evaluations safely.

The two inputs are only meaningfully comparable when they score the same
reference and structure artifacts using the same matching evidence. New
evaluations record that provenance in ``comparison_protocol``; older output is
still readable, but its apparent winners are withheld by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

METRICS = {
    # comparison result key: (aggregate JSON key, optimization direction)
    "weighted_CER": ("weighted_CER", "lower"),
    "weighted_WER": ("weighted_WER", "lower"),
    "reading_order_inversion_rate": (
        "mean_reading_order_inversion_rate",
        "lower",
    ),
    "structure_macro_f1": ("mean_structure_macro_f1", "higher"),
}
LAYOUT_METRICS = {"reading_order_inversion_rate", "structure_macro_f1"}

# Keep this in lock-step with evaluate_ocr.py. A partial protocol is treated
# as legacy/unverified rather than silently lending it a false equivalence.
COMPARISON_PROTOCOL_KEYS = (
    "version",
    "reference_sha256",
    "structure_sha256",
    "matching_evidence",
)


def load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation JSON must be an object: {path}")
    return value


def lower_is_better(
    left: float | int | None,
    chandra: float | int | None,
    left_engine: str = "surya",
) -> str:
    """Return the winning engine for a lower-is-better metric.

    ``left_engine`` defaults to ``surya`` so callers using the old helper
    signature retain its original result labels.
    """
    if left is None or chandra is None:
        return "unavailable"
    if left == chandra:
        return "tie"
    return left_engine if left < chandra else "chandra"


def higher_is_better(
    left: float | int | None,
    chandra: float | int | None,
    left_engine: str = "surya",
) -> str:
    """Return the winning engine for a higher-is-better metric."""
    if left is None or chandra is None:
        return "unavailable"
    if left == chandra:
        return "tie"
    return left_engine if left > chandra else "chandra"


def _require_engine(evaluation: dict[str, Any], expected: str, input_name: str) -> None:
    declared = evaluation.get("engine")
    if declared != expected:
        raise ValueError(
            f"{input_name} evaluation declares engine {declared!r}; expected {expected!r}"
        )


def _page_count(evaluation: dict[str, Any], input_name: str) -> int:
    value = evaluation.get("pages_evaluated")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{input_name} evaluation has invalid pages_evaluated {value!r}; "
            "expected a non-negative integer"
        )
    return value


def require_matching_page_counts(
    left: dict[str, Any], chandra: dict[str, Any], left_engine: str
) -> int:
    """Require that both evaluations score exactly the same number of pages."""
    left_count = _page_count(left, left_engine)
    chandra_count = _page_count(chandra, "chandra")
    if left_count != chandra_count:
        raise ValueError(
            "Evaluation page counts do not match: "
            f"{left_engine}={left_count}, chandra={chandra_count}"
        )
    return left_count


def _complete_comparison_protocol(
    evaluation: dict[str, Any], input_name: str
) -> dict[str, Any] | None:
    """Return a usable protocol or ``None`` for a legacy/partial evaluation."""
    protocol = evaluation.get("comparison_protocol")
    if not isinstance(protocol, dict):
        return None

    # Empty fingerprints are no stronger than absent fingerprints. Do not
    # mistake partial newly-written JSON for a verified comparison.
    if any(
        key not in protocol or protocol[key] is None or protocol[key] == ""
        for key in COMPARISON_PROTOCOL_KEYS
    ):
        return None

    page_alignment = protocol.get("page_alignment")
    if not isinstance(page_alignment, dict):
        return None
    page_ids = page_alignment.get("evaluated_page_ids")
    if (
        not isinstance(page_ids, list)
        or any(
            isinstance(page, bool) or not isinstance(page, int) or page <= 0 for page in page_ids
        )
        or len(set(page_ids)) != len(page_ids)
    ):
        return None
    if len(page_ids) != _page_count(evaluation, input_name):
        return None

    result = {key: protocol[key] for key in COMPARISON_PROTOCOL_KEYS}
    # Page count alone cannot establish a paired comparison: two equally
    # sized evaluations might score disjoint document pages. New evaluator
    # reports carry the deterministic, exact page-ID alignment needed here.
    result["evaluated_page_ids"] = page_ids
    return result


def comparison_status(
    left: dict[str, Any], chandra: dict[str, Any], left_engine: str
) -> tuple[str, str | None]:
    """Validate provenance and return ``(status, reason)``.

    Missing protocol data is deliberately non-fatal for historical output, but
    a complete protocol mismatch is unsafe and therefore rejected.
    """
    left_protocol = _complete_comparison_protocol(left, left_engine)
    chandra_protocol = _complete_comparison_protocol(chandra, "chandra")
    if left_protocol is None or chandra_protocol is None:
        return (
            "unverified",
            "one or both evaluations lack a complete comparison_protocol",
        )

    if left_protocol != chandra_protocol:
        raise ValueError(
            "Evaluation comparison_protocol values do not match; rerun both "
            "evaluations against the same reference, structure, and matching "
            "evidence"
        )
    return "verified", None


def _text_only_matching_evidence(protocol: dict[str, Any]) -> bool:
    """Return whether a verified report scored layout using text evidence only."""
    evidence = protocol.get("matching_evidence")
    return isinstance(evidence, str) and evidence.endswith("/text-only")


def _finite_order_rate(value: Any, input_name: str, page: int) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _measured_reading_order_rates(
    evaluation: dict[str, Any], input_name: str
) -> dict[int, float | int]:
    """Return safely measured per-page inversion rates keyed by page ID."""
    reading_order = evaluation.get("reading_order")
    per_page = reading_order.get("per_page") if isinstance(reading_order, dict) else None
    if not isinstance(per_page, list):
        return {}

    rates: dict[int, float | int] = {}
    for entry in per_page:
        if not isinstance(entry, dict) or entry.get("status") != "measured":
            continue
        page = entry.get("page")
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            continue
        rate = _finite_order_rate(entry.get("inversion_rate"), input_name, page)
        if rate is not None:
            rates[page] = rate
    return rates


def paired_reading_order_comparison(
    left: dict[str, Any], chandra: dict[str, Any], left_engine: str
) -> dict[str, Any]:
    """Compare reading order only on pages measured by both engines."""
    left_rates = _measured_reading_order_rates(left, left_engine)
    chandra_rates = _measured_reading_order_rates(chandra, "chandra")
    common_pages = sorted(set(left_rates) & set(chandra_rates))
    if not common_pages:
        return {
            "status": "inconclusive",
            "reason": "no_pages_measured_by_both_engines",
            "pages": [],
        }

    left_mean = sum(left_rates[page] for page in common_pages) / len(common_pages)
    chandra_mean = sum(chandra_rates[page] for page in common_pages) / len(common_pages)
    return {
        "status": "measured",
        "pages": common_pages,
        f"{left_engine}_mean_inversion_rate": left_mean,
        "chandra_mean_inversion_rate": chandra_mean,
        "winner": lower_is_better(left_mean, chandra_mean, left_engine),
    }


def _finite_metric(evaluation: dict[str, Any], metric: str, input_name: str) -> float | int | None:
    aggregate = evaluation.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{input_name} evaluation has no aggregate object")

    value = aggregate.get(metric)
    if value is None:
        return None
    # ``bool`` is an ``int`` subclass, but it is not a meaningful OCR metric.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{input_name} aggregate.{metric} must be a finite numeric scalar "
            f"or null, got {value!r}"
        )
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # JSON permits arbitrarily large integer literals. They cannot be a
        # finite scalar rate once represented as a numeric metric.
        finite = False
    if not finite:
        raise ValueError(
            f"{input_name} aggregate.{metric} must be a finite numeric scalar "
            f"or null, got {value!r}"
        )
    return value


def _metrics(evaluation: dict[str, Any], input_name: str) -> dict[str, float | int | None]:
    return {
        result_key: _finite_metric(evaluation, aggregate_key, input_name)
        for result_key, (aggregate_key, _direction) in METRICS.items()
    }


def build_comparison(
    left: dict[str, Any],
    chandra: dict[str, Any],
    *,
    left_engine: str,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Build one defensively validated comparison result."""
    _require_engine(left, left_engine, left_engine)
    _require_engine(chandra, "chandra", "chandra")
    pages_compared = require_matching_page_counts(left, chandra, left_engine)
    comparability, comparability_reason = comparison_status(left, chandra, left_engine)

    left_metrics = _metrics(left, left_engine)
    chandra_metrics = _metrics(chandra, "chandra")

    winners_suppressed = comparability == "unverified" and not allow_unverified
    paired_reading_order = None
    left_protocol = _complete_comparison_protocol(left, left_engine)
    if winners_suppressed:
        # Preserve the original result shape while making it impossible to
        # mistake an unverified score for a verdict.
        winners: dict[str, str | None] = {metric: None for metric in METRICS}
    else:
        winners = {}
        # ``--allow-unverified`` is an explicit legacy escape hatch. Preserve
        # its documented behavior of emitting all historical winner fields;
        # verified reports otherwise need the symmetric text-only policy for
        # layout claims.
        text_only_layout = (
            comparability == "unverified" and allow_unverified
        ) or _text_only_matching_evidence(left_protocol or {})
        for metric, (_aggregate_key, direction) in METRICS.items():
            if metric in LAYOUT_METRICS and not text_only_layout:
                winners[metric] = "inconclusive"
                continue
            if metric == "reading_order_inversion_rate":
                paired_reading_order = paired_reading_order_comparison(left, chandra, left_engine)
                winners[metric] = (
                    paired_reading_order["winner"]
                    if paired_reading_order["status"] == "measured"
                    else "inconclusive"
                )
                continue
            comparator = lower_is_better if direction == "lower" else higher_is_better
            winners[metric] = comparator(left_metrics[metric], chandra_metrics[metric], left_engine)

    result: dict[str, Any] = {
        "left_engine": left_engine,
        left_engine: left,
        "chandra": chandra,
        "pages_compared": pages_compared,
        "comparability": comparability,
        "winner_by_metric": winners,
    }
    if comparability_reason:
        result["comparability_reason"] = comparability_reason
    if paired_reading_order is not None:
        result["paired_reading_order"] = paired_reading_order
    if comparability == "verified" and not _text_only_matching_evidence(left_protocol or {}):
        result["layout_metric_caveat"] = (
            "Structural and reading-order winners are inconclusive because "
            "text-and-geometry matching can admit geometry for only one engine. "
            "Rerun both evaluations with --matching-evidence text-only."
        )
    if winners_suppressed:
        result["winner_suppression_reason"] = "unverified_comparability"
    elif comparability == "unverified":
        result["unverified_comparison_allowed"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    left = parser.add_mutually_exclusive_group(required=True)
    left.add_argument(
        "--cascade",
        help="Evaluation JSON produced with evaluate_ocr.py --engine cascade",
    )
    left.add_argument(
        "--surya",
        help="Deprecated legacy left input; use --cascade for the pipeline result",
    )
    parser.add_argument(
        "--chandra",
        required=True,
        help="Evaluation JSON produced with evaluate_ocr.py --engine chandra",
    )
    parser.add_argument("--out", default="artifacts/evaluation/comparison.json")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help=(
            "Emit metric winners for legacy evaluations without matching "
            "comparison_protocol fingerprints"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    left_engine = "cascade" if args.cascade is not None else "surya"
    left_path = args.cascade if args.cascade is not None else args.surya

    try:
        result = build_comparison(
            load(left_path),
            load(args.chandra),
            left_engine=left_engine,
            allow_unverified=args.allow_unverified,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    return result


def cli(argv: list[str] | None = None) -> int:
    """Run the report-producing comparison command with a shell exit status."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
