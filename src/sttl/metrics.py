"""Metric primitives with no policy-specific text normalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

Item = TypeVar("Item")


def levenshtein(left: Sequence[Item], right: Sequence[Item]) -> int:
    """Return the exact unit-cost edit distance for two sequences.

    Normalization is intentionally left to callers: the legacy metrics CLI
    and the evaluation pipeline have different, documented Unicode policies.
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[right_index - 1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]
