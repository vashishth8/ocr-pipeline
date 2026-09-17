#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.calibrate`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_calibrate = load_sttl_module("calibrate")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_calibrate))

sys.modules[__name__] = _calibrate
