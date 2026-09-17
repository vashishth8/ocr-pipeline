#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.pipeline`.

The implementation moved into the importable package during the incremental
layout migration. Replacing this module object preserves existing imports and
test monkeypatch targets until downstream callers can use ``sttl.pipeline``.
"""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_pipeline = load_sttl_module("pipeline")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_pipeline))

sys.modules[__name__] = _pipeline
