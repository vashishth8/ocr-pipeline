"""Pipeline configuration and deterministic OCR-route selection."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from sttl.text import normalize_whitespace


@dataclass(frozen=True)
class PipelineConfig:
    """Every setting that can change a routed page's result."""

    language: str = "eng"
    # Optional workspace-local Tesseract language data. It avoids a global
    # package install while keeping the model location explicit and resumable.
    tessdata_dir: str | None = None
    render_dpi: int = 200
    tesseract_psm: int = 3
    tesseract_timeout_seconds: int = 180
    surya_timeout_seconds: int = 1_800
    surya_batch_size: int = 4
    surya_keep_server: bool = False
    # Keep Surya's established fallback behavior by default. ``none`` is a
    # compact-only mode: a rejected Tesseract candidate remains audit evidence
    # instead of starting a heavyweight fallback or becoming authoritative.
    fallback_engine: str = "surya"
    # ``auto`` makes Surya the primary OCR engine for Hindi/Devanagari only
    # when the heavyweight route is enabled. Compact-only Hindi jobs therefore
    # remain explicitly Tesseract-first without silently loading Surya.
    ocr_engine: str = "auto"
    # Opt in because a structurally complex page costs a full Surya pass even
    # when its plain Tesseract text is high-confidence.
    structure_aware: bool = False
    # A structure escalation already has a quality-accepted Tesseract pass.
    # Retain Surya's semantic regions while preferring high-confidence
    # Tesseract words inside those regions. This never applies to a quality
    # rejection, where Surya remains the text authority.
    structure_hybrid_text: bool = True
    min_native_chars: int = 80
    min_native_words: int = 12
    max_native_garbage_ratio: float = 0.05
    dominant_image_ratio: float = 0.55
    min_tesseract_chars: int = 20
    min_tesseract_words: int = 4
    min_mean_confidence: float = 65.0
    min_confident_word_ratio: float = 0.70
    confident_word_threshold: float = 60.0
    max_tesseract_garbage_ratio: float = 0.08
    min_plausible_word_ratio: float = 0.70


def language_requests_hindi(language: str) -> bool:
    """Return whether a Tesseract language setting requests Devanagari OCR."""
    configured = {value.strip().lower() for value in language.split("+") if value.strip()}
    return bool(configured & {"hin", "devanagari", "script/devanagari"})


def primary_ocr_engine(config: PipelineConfig) -> str:
    """Resolve the page OCR engine without changing native-text routing."""
    if config.ocr_engine != "auto":
        return config.ocr_engine
    if language_requests_hindi(config.language) and config.fallback_engine == "surya":
        return "surya"
    return "tesseract"


def garbage_ratio(value: str) -> float:
    """Return the share of non-whitespace characters unsuitable as OCR text."""
    characters = [char for char in value if not char.isspace()]
    if not characters:
        return 0.0
    garbage = sum(
        char == "\ufffd" or not char.isprintable() or ord(char) < 32 for char in characters
    )
    return garbage / len(characters)


def private_use_character_count(value: str) -> int:
    """Count non-whitespace private-use glyphs as a routing diagnostic."""
    return sum(unicodedata.category(char) == "Co" for char in value if not char.isspace())


def _unicode_word_char(char: str) -> bool:
    """Return whether a character belongs to a Unicode word token."""
    return char == "_" or char.isalnum() or unicodedata.category(char).startswith("M")


def _unicode_word_tokens(value: str) -> list[str]:
    """Split text into word tokens without discarding combining marks."""
    tokens: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFC", value):
        if _unicode_word_char(char):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def plausible_word_ratio(value: str) -> float:
    """Return a language-neutral plausibility ratio for OCR routing."""
    tokens = re.findall(r"\S+", value)
    if not tokens:
        return 0.0
    plausible = 0
    for token in tokens:
        symbols = sum(
            not (_unicode_word_char(char) or char in "'_-.,:/()[]{}%+*=#।॥") for char in token
        )
        if (
            any(char.isalnum() for char in token)
            and len(token) <= 64
            and symbols / len(token) <= 0.30
        ):
            plausible += 1
    return plausible / len(tokens)


def classify_page_signals(
    *,
    text: str,
    text_block_count: int,
    image_count: int,
    image_area_ratio: float,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Classify extracted page evidence without rendering the page."""
    native_text = normalize_whitespace(text)
    text_chars = len(native_text)
    word_count = len(re.findall(r"\S+", native_text))
    native_garbage = garbage_ratio(native_text)
    native_private_use = private_use_character_count(native_text)
    usable = (
        text_chars >= config.min_native_chars
        and word_count >= config.min_native_words
        and native_garbage <= config.max_native_garbage_ratio
    )
    image_dominant = image_area_ratio >= config.dominant_image_ratio
    if usable and not image_dominant:
        classification, route = "DIGITAL", "native_text"
    elif usable:
        classification, route = "MIXED", "tesseract"
    elif image_count or image_dominant:
        classification, route = "SCANNED", "tesseract"
    else:
        classification, route = "OCR_NEEDED", "tesseract"
    return {
        "classification": classification,
        "route": route,
        "native_text": native_text,
        "signals": {
            "native_text_chars": text_chars,
            "native_word_count": word_count,
            "native_text_block_count": text_block_count,
            "native_garbage_ratio": round(native_garbage, 6),
            "native_private_use_count": native_private_use,
            "image_count": image_count,
            "image_area_ratio": round(image_area_ratio, 6),
        },
    }


def tesseract_quality(
    text: str, confidences: Sequence[float], config: PipelineConfig
) -> dict[str, Any]:
    """Decide whether cheap OCR is safe enough to accept."""
    text = normalize_whitespace(text)
    character_count = len(text)
    word_count = len(re.findall(r"\S+", text))
    mean_confidence = fmean(confidences) if confidences else None
    confident_ratio = (
        sum(value >= config.confident_word_threshold for value in confidences) / len(confidences)
        if confidences
        else 0.0
    )
    ocr_garbage = garbage_ratio(text)
    plausibility = plausible_word_ratio(text)
    rejected: list[str] = []
    if character_count < config.min_tesseract_chars:
        rejected.append("too_few_characters")
    if word_count < config.min_tesseract_words:
        rejected.append("too_few_words")
    if not confidences:
        rejected.append("no_word_confidences")
    elif mean_confidence is not None and mean_confidence < config.min_mean_confidence:
        rejected.append("low_mean_confidence")
    if confident_ratio < config.min_confident_word_ratio:
        rejected.append("low_confident_word_ratio")
    if ocr_garbage > config.max_tesseract_garbage_ratio:
        rejected.append("high_garbage_ratio")
    if plausibility < config.min_plausible_word_ratio:
        rejected.append("low_word_plausibility")
    return {
        "accepted": not rejected,
        "rejection_reasons": rejected,
        "text_chars": character_count,
        "word_count": word_count,
        "word_confidence_count": len(confidences),
        "mean_word_confidence": round(mean_confidence, 3) if mean_confidence is not None else None,
        "confident_word_ratio": round(confident_ratio, 6),
        "garbage_ratio": round(ocr_garbage, 6),
        "plausible_word_ratio": round(plausibility, 6),
    }


def pipeline_description(config: PipelineConfig) -> str:
    """Describe the configured route without implying an unused fallback."""
    if primary_ocr_engine(config) == "surya":
        return "PyMuPDF -> Surya OCR (primary route)"
    fallback = "Surya fallback" if config.fallback_engine == "surya" else "no heavyweight fallback"
    return f"PyMuPDF -> Tesseract 5 -> quality / optional structure gate -> {fallback}"
