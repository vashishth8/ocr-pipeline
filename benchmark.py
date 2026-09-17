#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.benchmark`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_benchmark = load_sttl_module("benchmark")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_benchmark))

sys.modules[__name__] = _benchmark
