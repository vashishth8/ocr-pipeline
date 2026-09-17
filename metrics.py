#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.metrics_cli`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_metrics_cli = load_sttl_module("metrics_cli")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_metrics_cli))

sys.modules[__name__] = _metrics_cli
