from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sttl.geometry import coerce_bbox
from sttl.jsonio import load_json_object_bytes
from sttl.metrics import levenshtein
from sttl.text import html_to_text, normalize_whitespace


class SharedUtilityContractTests(unittest.TestCase):
    def test_html_text_preserves_row_boundaries(self) -> None:
        self.assertEqual(
            html_to_text(
                "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>10</td></tr></table>"
            ),
            "Item Value\nA 10",
        )
        self.assertEqual(normalize_whitespace(" one\n\t two "), "one two")

    def test_bbox_coercion_is_non_throwing_and_policy_driven(self) -> None:
        self.assertEqual(coerce_bbox([1, "2", 3.0, 4]), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(coerce_bbox([1, 2, 3, 4, 5]), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(coerce_bbox([1, 2, 3, 4, 5], exact_length=True))
        self.assertIsNone(coerce_bbox([1, 2, "invalid", 4]))

    def test_json_loader_hashes_exact_bytes(self) -> None:
        raw = b'{"item": "value"}\n'
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "artifact.json"
            path.write_bytes(raw)
            data, digest = load_json_object_bytes(path)
        self.assertEqual(data, json.loads(raw))
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_levenshtein_handles_text_and_token_sequences(self) -> None:
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein(["दे", "वनागरी"], ["दे", "नागरी"]), 1)


if __name__ == "__main__":
    unittest.main()
