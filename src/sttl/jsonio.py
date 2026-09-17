"""JSON artifact loading with byte-exact provenance digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact artifact bytes."""
    return hashlib.sha256(value).hexdigest()


def load_json_object_bytes(path: Path) -> tuple[dict[str, Any], str]:
    """Load a UTF-8 JSON object and return its exact-byte SHA-256 digest."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data, sha256_bytes(raw)
