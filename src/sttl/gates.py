"""Pipeline configuration and deterministic OCR-route selection."""

from __future__ import annotations

from dataclasses import dataclass


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
