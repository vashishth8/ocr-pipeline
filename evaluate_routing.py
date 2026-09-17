#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.routing`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_routing = load_sttl_module("routing")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_routing))

sys.modules[__name__] = _routing
