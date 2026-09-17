#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.compare`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_compare = load_sttl_module("compare")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_compare))

sys.modules[__name__] = _compare
