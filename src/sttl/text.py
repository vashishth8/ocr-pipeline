"""Text normalization and conservative HTML-to-text conversion helpers."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any


class HTMLTextExtractor(HTMLParser):
    """Turn block HTML into readable text while retaining row boundaries."""

    BREAKS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    CELLS = {"td", "th"}

    def __init__(self, *, collapse_unicode_whitespace: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.collapse_unicode_whitespace = collapse_unicode_whitespace
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BREAKS:
            self.parts.append("\n")
        elif tag.lower() in self.CELLS:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        if self.collapse_unicode_whitespace:
            lines = [" ".join(line.split()) for line in value.splitlines()]
        else:
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def html_to_text(value: Any, *, collapse_unicode_whitespace: bool = False) -> str:
    """Extract readable text from a possibly absent HTML block."""
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = HTMLTextExtractor(collapse_unicode_whitespace=collapse_unicode_whitespace)
    parser.feed(value)
    parser.close()
    return parser.text()


def normalize_whitespace(value: str) -> str:
    """Collapse arbitrary Unicode whitespace to one ASCII space."""
    return " ".join(value.split())
