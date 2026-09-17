#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.ground_truth`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_ground_truth = load_sttl_module("ground_truth")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_ground_truth))

sys.modules[__name__] = _ground_truth
