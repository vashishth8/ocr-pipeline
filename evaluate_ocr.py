#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.evaluate`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_evaluate = load_sttl_module("evaluate")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_evaluate))

sys.modules[__name__] = _evaluate
