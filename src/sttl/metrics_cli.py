"""Compatibility command for STTL's simple standalone CER/WER calculator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sttl.metrics import levenshtein


def normalize(text: str) -> str:
    """Preserve the legacy CLI's whitespace-only normalization policy."""
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Return legacy character error rate semantics."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Return legacy whitespace-token word error rate semantics."""
    reference_words = normalize(reference).split()
    hypothesis_words = normalize(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein(reference_words, hypothesis_words) / len(reference_words)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("ocr_output", type=Path)
    args = parser.parse_args(argv)
    reference = args.ground_truth.read_text(encoding="utf-8")
    hypothesis = args.ocr_output.read_text(encoding="utf-8")
    print(f"CER: {cer(reference, hypothesis):.4%}")
    print(f"WER: {wer(reference, hypothesis):.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
