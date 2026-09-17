"""Durable JSON artifacts and resume-manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

COMPLETE = "complete"
HASH_CHUNK_BYTES = 1_048_576


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically publish a JSON artifact once its complete content is ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and fsync one completed page before derived exports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed_pages(path: Path) -> dict[int, dict[str, Any]]:
    """Return the final complete record for each page in an append-only manifest."""
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                page_number = int(record["page"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid manifest at {path}:{line_number}") from exc
            if record.get("status") == COMPLETE:
                completed[page_number] = record
    return completed


def source_identity(pdf_path: Path) -> dict[str, Any]:
    """Return a path-free, content-based resume identity for a source PDF."""
    stat = pdf_path.stat()
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return {
        "file_name": pdf_path.name,
        "size_bytes": stat.st_size,
        "content_sha256": digest.hexdigest(),
    }


def job_output_dir(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    """Map an input PDF to its stable, root-relative output directory."""
    try:
        relative = pdf_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(pdf_path.name)
    return output_root / relative.with_suffix("")
