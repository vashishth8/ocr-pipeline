"""Regression tests for legacy root command compatibility."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from sttl_compat import load_sttl_module, run_sttl_main

ROOT = Path(__file__).resolve().parent


class RootShimTests(unittest.TestCase):
    def test_loader_replaces_legacy_namespace_package(self) -> None:
        original_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "sttl" or name.startswith("sttl.")
        }
        legacy_namespace = types.ModuleType("sttl")
        legacy_namespace.__path__ = [str(ROOT / "sttl")]
        sys.modules["sttl"] = legacy_namespace

        try:
            version = load_sttl_module("version")
            self.assertEqual(Path(version.__file__).resolve(), ROOT / "src/sttl/version.py")
        finally:
            for name in list(sys.modules):
                if name == "sttl" or name.startswith("sttl."):
                    del sys.modules[name]
            sys.modules.update(original_modules)

    def test_pipeline_shim_reexports_geometry_helpers(self) -> None:
        import pdf_pipeline
        from sttl.geometry import scale_bbox_to_pdf_points, scale_polygon_to_pdf_points

        self.assertIs(pdf_pipeline.scale_bbox_to_pdf_points, scale_bbox_to_pdf_points)
        self.assertIs(pdf_pipeline.scale_polygon_to_pdf_points, scale_polygon_to_pdf_points)

    def test_report_result_is_not_used_as_a_process_exit_status(self) -> None:
        report_module = types.ModuleType("report_module")
        report_module.main = lambda: {"status": "ok"}

        self.assertEqual(run_sttl_main(report_module), 0)

    def test_report_cli_adapters_return_zero(self) -> None:
        from unittest.mock import patch

        from sttl import calibrate, compare

        with patch.object(calibrate, "main", return_value={"status": "ok"}):
            self.assertEqual(calibrate.cli(), 0)
        with patch.object(compare, "main", return_value={"status": "ok"}):
            self.assertEqual(compare.cli(), 0)
