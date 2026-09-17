"""Typed, non-throwing coercion for untrusted OCR bounding boxes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

PDF_POINT_FRAME = "pdf_page_points_top_left_unrotated"


def coerce_bbox(value: Any, *, exact_length: bool = False) -> list[float] | None:
    """Return four numeric coordinates or ``None`` for malformed input.

    Callers deliberately choose whether a provider format accepts extra
    coordinates. This primitive preserves the existing non-throwing behavior
    used for external OCR payloads without making different schemas identical.
    """
    if not isinstance(value, (list, tuple)):
        return None
    if (exact_length and len(value) != 4) or (not exact_length and len(value) < 4):
        return None
    try:
        return [float(value[index]) for index in range(4)]
    except (TypeError, ValueError, OverflowError):
        return None


def bbox_from_mapping(
    value: Any,
    *,
    keys: Sequence[str] = ("bbox",),
    exact_length: bool = False,
) -> list[float] | None:
    """Extract a bbox from a mapping using the caller's accepted key policy."""
    if not isinstance(value, dict):
        return None
    for key in keys:
        bbox = coerce_bbox(value.get(key), exact_length=exact_length)
        if bbox is not None:
            return bbox
    return None


def derotation_matrix(
    raster: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    """Return a JSON-safe affine derotation matrix, if one was persisted."""
    raw = raster.get("derotation_matrix")
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        return tuple(float(value) for value in raw)  # type: ignore[return-value]
    except (TypeError, ValueError, OverflowError):
        return None


def raster_point_to_pdf_point(x: Any, y: Any, raster: dict[str, Any]) -> list[float] | None:
    """Map a rendered-image point to PyMuPDF's unrotated PDF-point space."""
    rendered_bbox = coerce_bbox(raster.get("rendered_pdf_bbox")) or coerce_bbox(
        raster.get("pdf_bbox")
    )
    try:
        raster_width = float(raster["raster_width"])
        raster_height = float(raster["raster_height"])
        source_x = float(x)
        source_y = float(y)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if rendered_bbox is None or raster_width <= 0 or raster_height <= 0:
        return None
    point_x = rendered_bbox[0] + source_x * (rendered_bbox[2] - rendered_bbox[0]) / raster_width
    point_y = rendered_bbox[1] + source_y * (rendered_bbox[3] - rendered_bbox[1]) / raster_height
    matrix = derotation_matrix(raster)
    if matrix is None:
        return [point_x, point_y]
    a, b, c, d, e, f = matrix
    return [a * point_x + c * point_y + e, b * point_x + d * point_y + f]


def scale_bbox_to_pdf_points(bbox: Any, raster: dict[str, Any]) -> list[float] | None:
    """Map a raster bbox into PDF points, including any recorded derotation."""
    source = coerce_bbox(bbox)
    if source is None:
        return None
    points = [
        raster_point_to_pdf_point(source[0], source[1], raster),
        raster_point_to_pdf_point(source[2], source[1], raster),
        raster_point_to_pdf_point(source[2], source[3], raster),
        raster_point_to_pdf_point(source[0], source[3], raster),
    ]
    if any(point is None for point in points):
        return None
    mapped = [point for point in points if point is not None]
    return [
        min(point[0] for point in mapped),
        min(point[1] for point in mapped),
        max(point[0] for point in mapped),
        max(point[1] for point in mapped),
    ]


def scale_polygon_to_pdf_points(value: Any, raster: dict[str, Any]) -> list[list[float]] | None:
    """Map a polygon from raster pixels into PDF points, retaining its shape."""
    if not isinstance(value, (list, tuple)):
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        mapped = raster_point_to_pdf_point(point[0], point[1], raster)
        if mapped is None:
            return None
        points.append(mapped)
    return points


def scale_ocr_items_to_pdf_points(
    items: Sequence[dict[str, Any]], raster: dict[str, Any]
) -> list[dict[str, Any]]:
    """Copy OCR items into a common PDF-point coordinate system."""
    scaled: list[dict[str, Any]] = []
    for item in items:
        output = dict(item)
        bbox = coerce_bbox(item.get("bbox"))
        if bbox is not None:
            output["source_bbox"] = bbox
            mapped_bbox = scale_bbox_to_pdf_points(bbox, raster)
            if mapped_bbox is not None:
                output["bbox"] = mapped_bbox
        polygon = item.get("polygon")
        mapped_polygon = scale_polygon_to_pdf_points(polygon, raster)
        if mapped_polygon is not None and isinstance(polygon, (list, tuple)):
            output["source_polygon"] = [list(point[:2]) for point in polygon]
            output["polygon"] = mapped_polygon
        output["coordinate_space"] = "pdf_points"
        output["coordinate_frame"] = PDF_POINT_FRAME
        scaled.append(output)
    return scaled
