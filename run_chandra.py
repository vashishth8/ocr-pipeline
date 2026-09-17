#!/usr/bin/env python3
"""Backward-compatible CLI shim for :mod:`sttl.chandra`."""

from __future__ import annotations

import sys

from sttl_compat import load_sttl_module, run_sttl_main

_chandra = load_sttl_module("chandra")


if __name__ == "__main__":
    raise SystemExit(run_sttl_main(_chandra))

sys.modules[__name__] = _chandra
